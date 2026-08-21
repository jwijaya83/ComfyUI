"""The render worker: one fully-resolved job -> an MP4 -> a status report.

This is the Python port of ai-chat/services/render-worker (render.js processJob +
server.js), bundled in the same container as the ComfyUI it drives. It is STATELESS —
no database, no personality files; everything it needs arrives in the job descriptor —
so any number of these can run anywhere and share one queue.

THREE INTAKES, ONE RENDER PATH (`QUEUE_DRIVER`, see config.py):

    redis   drain ai-chat's main render queue (render:high / render:low Redis Streams)
            as part of the shared `renderers` consumer group. Point REDIS_URL at
            production and this box joins the render pool — that is the scale story.
    runpod  RunPod Serverless: runpod.serverless.start polls RunPod's queue outbound.
    http    chat-api POSTs one job to /render (the no-broker local profile).

All three call process_job(), and all three report progress/terminal status to chat-api's
/internal/render-events webhook — chat-api's render_jobs row stays the source of truth,
never the queue.
"""
import os
import sys
import traceback

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import gcs  # noqa: E402
import storage  # noqa: E402
from comfy_client import (  # noqa: E402
    collect_outputs,
    download_output,
    free_comfy_vram,
    get_history,
    submit_prompt,
    upload_image,
    upload_video,
    wait_for_ready,
    watch_prompt,
)
from gpu_lease import free_ollama, gpu_lease
from mock import render_mock  # noqa: E402
from reporter import log_config, report  # noqa: E402
from workflow_builder import build_workflow  # noqa: E402

COMFY_READY_TIMEOUT = int(os.environ.get("COMFY_READY_TIMEOUT", "300"))


def _download(url, timeout=300, headers=None):
    r = requests.get(url, timeout=timeout, headers=headers or {})
    if not r.ok:
        raise RuntimeError(f"asset download failed {r.status_code}: {url[:80]}")
    return r.content


def _chat_api_asset(pathname):
    """Fetch an asset from chat-api's header-guarded /internal routes.

    The FALLBACK path. chat-api hands us signed GCS urls for the seed clip whenever GCS
    is on (renderQueue.js resolves them at enqueue), but with GCS off — or for an older
    un-mirrored seed, or a persona reference, which is never mirrored — the queue payload
    carries only an id/slug and the bytes live on chat-api's disk. Same endpoints
    render-worker used, same x-internal-token guard.
    """
    return _download(
        f"{config.CHAT_API_URL}{pathname}",
        headers={"x-internal-token": config.INTERNAL_TOKEN},
    )


def _resolve_assets(job):
    """Pick THE ONE conditioning input, by render.js's precedence: seed reuse > persona
    reference (seed generation) > img2video reference. Each may arrive either as a URL
    chat-api pre-resolved (the remote/runpod shape) or as a bare id/slug we fetch from
    chat-api ourselves (the redis/http shape)."""
    reference_image = job.get("referenceImage")
    source_video = None
    use_ref = job.get("useReferenceImage", True) is not False
    job_id = job.get("jobId")

    if job.get("sourceVideoUrl") or job.get("sourceSeedId"):
        # Latent injection (reuse): the seed clip becomes the denoise base. This path
        # uses the source video, NOT a reference image, so the i2v upload is skipped.
        if job.get("sourceVideoUrl"):
            data = _download(job["sourceVideoUrl"])
            origin = "gcs"
        else:
            seed_id = job["sourceSeedId"]
            data = _chat_api_asset(f"/internal/seed-media/{seed_id}.mp4")
            origin = f"seed {seed_id}"
        source_video = upload_video(data, filename=f"seed_{job.get('sourceSeedId', 'src')}.mp4")
        print(f"↺ job {job_id} latent-injects {origin} -> {source_video}", flush=True)
    elif job.get("personaReferenceUrl") or job.get("personaReference"):
        # Seed generation: condition on the persona's own picture (the IC-LoRA sheet).
        slug = job.get("personaReference") or "persona"
        data = (
            _download(job["personaReferenceUrl"])
            if job.get("personaReferenceUrl")
            else _chat_api_asset(f"/internal/persona-reference/{slug}")
        )
        fname = f"{slug}.reference.png"
        reference_image = upload_image(data, filename=fname)
        report({"jobId": job_id, "status": "running", "referenceImage": fname})
    elif use_ref and (job.get("referenceImageUrl") or job.get("personalitySlug")):
        # img2video: condition on the persona's reference picture.
        if job.get("referenceImageUrl"):
            data = _download(job["referenceImageUrl"])
            fname = job.get("referenceImageFilename") or "reference.png"
        else:
            slug = job["personalitySlug"]
            data = _chat_api_asset(f"/internal/persona-reference/{slug}")
            fname = f"{slug}.reference.png"
        reference_image = upload_image(data, filename=fname)
        report({"jobId": job_id, "status": "running", "referenceImage": fname})

    return reference_image, source_video, use_ref


def _render_comfy(job, on_progress):
    job_id = job.get("jobId")
    reference_image, source_video, use_ref = _resolve_assets(job)

    workflow = build_workflow(
        name=job["workflow"],
        prompt=job.get("positive") or job.get("dialogue"),
        reference_image=reference_image,
        use_reference_image=use_ref,
        lora_name=job.get("loraName"),
        duration_seconds=(float(job["durationSeconds"]) if job.get("durationSeconds") else None),
        width=(int(job["width"]) if job.get("width") else None),
        height=(int(job["height"]) if job.get("height") else None),
        filename_prefix=(f"ltx23/chat_{job['chatId']}" if job.get("chatId") else None),
        source_video=source_video,
    )

    prompt_id, client_id = submit_prompt(workflow)
    report({"jobId": job_id, "status": "running", "comfyPromptId": prompt_id})

    def on_event(msg):
        if msg.get("type") != "progress":
            return
        d = msg.get("data") or {}
        on_progress(d.get("value"), d.get("max"), d.get("node"))

    watch_prompt(client_id, prompt_id, on_event=on_event)
    outputs = collect_outputs(get_history(prompt_id))
    file = next((o for o in outputs if o.get("kind") == "videos"), outputs[0] if outputs else None)
    if not file:
        raise RuntimeError("ComfyUI produced no outputs")
    return download_output(file)


def _render(job, on_progress):
    """The mock renders on CPU (ffmpeg) — no GPU, no lease. Real ComfyUI takes the single
    GPU: hold the cross-service lease so ai-chat's llm-worker (Ollama) can't run on the
    card at the same time, evict any model it left resident before we grow VRAM, and hand
    the card back clean on the way out. All three are no-ops unless GPU_LEASE=1 /
    OLLAMA_HOST is set (on RunPod the GPU is ours alone)."""
    if config.MOCK_COMFY:
        return render_mock(job, on_progress)

    with gpu_lease(label=f"render {job.get('jobId')}"):
        free_ollama()
        try:
            return _render_comfy(job, on_progress)
        finally:
            # Single GPU: unload ComfyUI so the next text turn's Ollama reloads into a
            # clean card instead of contending with ComfyUI's cached models (which spills
            # the LLM to CPU). Lease mode only — on a dedicated GPU this would needlessly
            # cold-load ComfyUI on the next render.
            if config.GPU_LEASE:
                free_comfy_vram()


def process_job(job, rp_event=None):
    """Render one job with internal retry, then report the terminal status. NEVER raises:
    on exhaustion it reports `failed` and returns, so an intake can ACK unconditionally."""
    job_id = job.get("jobId")
    report({"jobId": job_id, "status": "running"})

    def on_progress(value, maximum, node=None):
        report({
            "jobId": job_id,
            "status": "running",
            "progress": {"value": value, "max": maximum, "node": node},
        })
        if rp_event is not None:
            try:
                import runpod

                runpod.serverless.progress_update(rp_event, {"value": value, "max": maximum})
            except Exception:  # noqa: BLE001 - progress is best-effort
                pass

    last_err = None
    for attempt in range(1, config.JOB_MAX_ATTEMPTS + 1):
        try:
            buffer = _render(job, on_progress)
            # One naming scheme for every intake, matching render-worker/storage.js:
            # chat_<jobId>.mp4 in the RESPONSE bucket. chat-api owns what happens next —
            # for an admin seed render it server-side-copies the object into video-seed
            # and reclaims this one (seedService.mirrorCompletedSeedRender).
            output_url = storage.save_video(buffer, f"chat_{job_id}.mp4")
            report({"jobId": job_id, "status": "completed", "outputUrl": output_url})
            print(f"✓ job {job_id} completed -> {output_url}", flush=True)
            return {"jobId": job_id, "status": "completed", "outputUrl": output_url}
        except Exception as e:  # noqa: BLE001 - retry then report failed
            last_err = e
            kind = f" ({e.kind})" if getattr(e, "kind", None) else ""
            print(f"↻ job {job_id} attempt {attempt}/{config.JOB_MAX_ATTEMPTS} failed{kind}: {e}", flush=True)
            traceback.print_exc()
            # A deliberate cancel (manual interrupt / cleared queue) must NOT be
            # re-rendered — the operator stopped it on purpose. Fail fast.
            if getattr(e, "retriable", True) is False:
                print(f"⃠ job {job_id} not retriable{kind} — reporting failed without retry", flush=True)
                break

    msg = str(last_err)
    report({"jobId": job_id, "status": "failed", "error": msg})
    print(f"✗ job {job_id} failed permanently: {msg}", flush=True)
    return {"jobId": job_id, "status": "failed", "error": msg}


def handler(event):
    """RunPod Serverless entry point."""
    job = (event or {}).get("input") or {}
    if not job.get("jobId") or not job.get("workflow"):
        return {"error": "input must include jobId and workflow"}
    return process_job(job, event)


def _check_delivery():
    """Fail fast if a finished render would have nowhere to go — better to exit now than
    to burn a GPU render and discover it at upload time.

    GCS configured -> validate the credentials hard (gcs.log_config raises). Otherwise a
    shared MEDIA_DIR is a complete delivery path on its own (the in-compose profile:
    chat-api re-serves the same volume at /media)."""
    if gcs.enabled(gcs.response_bucket()):
        gcs.log_config()
    elif config.MEDIA_DIR:
        print(f"[gcs] not configured — delivering to {storage.describe()}", flush=True)
    else:
        raise RuntimeError(
            "no delivery configured: set MEDIA_DIR (a volume shared with chat-api) or a "
            "GCS response bucket + credentials, or renders cannot be returned"
        )


def main():
    print(config.summary(), flush=True)
    log_config()
    _check_delivery()

    if not config.MOCK_COMFY:
        print("[handler] waiting for ComfyUI to be ready ...", flush=True)
        wait_for_ready(timeout=COMFY_READY_TIMEOUT)
        print("[handler] ComfyUI ready", flush=True)
    else:
        print("[handler] MOCK_COMFY=1 — rendering with ffmpeg, ComfyUI not used", flush=True)

    driver = config.QUEUE_DRIVER
    if driver == "redis":
        # Health probe stays up alongside the consumer (the compose healthcheck).
        from http_server import serve

        serve(process_job)
        from queue_consumer import start_consumer

        start_consumer(process_job)
    elif driver == "http":
        from http_server import serve

        serve(process_job)
        print("[handler] http intake — waiting for chat-api to POST /render", flush=True)
        import threading

        threading.Event().wait()  # park forever; the server runs on its own thread
    elif driver == "runpod":
        import runpod

        print("[handler] starting RunPod job listener (serverless)", flush=True)
        runpod.serverless.start({"handler": handler})
    else:
        raise SystemExit(f"unknown QUEUE_DRIVER '{driver}' (expected redis | runpod | http)")


if __name__ == "__main__":
    main()
