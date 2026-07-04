"""RunPod serverless handler — the render-worker, ported to a RunPod job handler.

Lifecycle (per the design we agreed on):
  1. RunPod invokes handler(event); event["input"] is the fully-resolved render job
     chat-api built (same descriptor as enqueueRender today, with assets as signed URLs).
  2. We render via the in-container ComfyUI (127.0.0.1:8188), streaming progress.
  3. We upload the MP4 to GCS and report `completed` (with the URL) to chat-api's
     /internal/render-events webhook — chat-api stays the source of truth.
  4. We also return {outputUrl} so it lands in RunPod's job output / terminal webhook.

This is the Python port of render-worker/render.js processJob(). It does NOT use the
GPU-lease / Ollama-eviction machinery — on RunPod the GPU is ComfyUI-dedicated.
"""
import os
import sys
import traceback

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod  # noqa: E402

import gcs  # noqa: E402
from comfy_client import (  # noqa: E402
    collect_outputs,
    download_output,
    get_history,
    submit_prompt,
    upload_image,
    upload_video,
    wait_for_ready,
    watch_prompt,
)
from reporter import log_config, report  # noqa: E402
from workflow_builder import build_workflow  # noqa: E402

MAX_ATTEMPTS = int(os.environ.get("JOB_MAX_ATTEMPTS", "3"))
COMFY_READY_TIMEOUT = int(os.environ.get("COMFY_READY_TIMEOUT", "300"))


def _download(url, timeout=300):
    r = requests.get(url, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"asset download failed {r.status_code}: {url[:80]}")
    return r.content


def _resolve_assets(job):
    """Pick the conditioning input (mirror render.js). Assets arrive as signed URLs;
    the worker never calls back into chat-api for files."""
    reference_image = job.get("referenceImage")
    source_video = None
    use_ref = job.get("useReferenceImage", True) is not False
    job_id = job.get("jobId")

    if job.get("sourceVideoUrl"):
        # Latent injection (reuse): download the seed clip, upload to ComfyUI input/.
        data = _download(job["sourceVideoUrl"])
        source_video = upload_video(data, filename=f"seed_{job.get('sourceSeedId', 'src')}.mp4")
        print(f"↺ job {job_id} latent-injects -> {source_video}", flush=True)
    elif job.get("personaReferenceUrl"):
        # Seed generation: condition on the persona's own picture (IC-LoRA sheet).
        data = _download(job["personaReferenceUrl"])
        fname = f"{job.get('personalitySlug', 'persona')}.reference.png"
        reference_image = upload_image(data, filename=fname)
        report({"jobId": job_id, "status": "running", "referenceImage": fname})
    elif use_ref and job.get("referenceImageUrl"):
        # img2video: chat-api already picked the reference from the persona's pool.
        data = _download(job["referenceImageUrl"])
        fname = job.get("referenceImageFilename") or "reference.png"
        reference_image = upload_image(data, filename=fname)
        report({"jobId": job_id, "status": "running", "referenceImage": fname})

    return reference_image, source_video, use_ref


def _render(job, rp_event):
    job_id = job.get("jobId")
    reference_image, source_video, use_ref = _resolve_assets(job)

    workflow = build_workflow(
        name=job["workflow"],
        prompt=job.get("positive") or job.get("dialogue"),
        reference_image=reference_image,
        use_reference_image=use_ref,
        lora_name=job.get("loraName"),
        duration_seconds=(float(job["durationSeconds"]) if job.get("durationSeconds") else None),
        filename_prefix=(f"ltx23/chat_{job['chatId']}" if job.get("chatId") else None),
        source_video=source_video,
    )

    prompt_id, client_id = submit_prompt(workflow)
    report({"jobId": job_id, "status": "running", "comfyPromptId": prompt_id})

    def on_event(msg):
        if msg.get("type") != "progress":
            return
        d = msg.get("data") or {}
        report({
            "jobId": job_id,
            "status": "running",
            "progress": {"value": d.get("value"), "max": d.get("max"), "node": d.get("node")},
        })
        try:
            runpod.serverless.progress_update(rp_event, {"value": d.get("value"), "max": d.get("max")})
        except Exception:  # noqa: BLE001 - progress is best-effort
            pass

    watch_prompt(client_id, prompt_id, on_event=on_event)
    outputs = collect_outputs(get_history(prompt_id))
    file = next((o for o in outputs if o.get("kind") == "videos"), outputs[0] if outputs else None)
    if not file:
        raise RuntimeError("ComfyUI produced no outputs")
    return download_output(file)


def process_job(job, rp_event):
    job_id = job.get("jobId")
    report({"jobId": job_id, "status": "running"})

    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            buffer = _render(job, rp_event)
            # Seed clips land in the seed-video bucket; per-turn renders in the
            # response bucket. GCS_SEED_BUCKET unset -> falls back to GCS_BUCKET.
            is_seed = job.get("workflow") == "seed_workflow"
            bucket = gcs.GCS_SEED_BUCKET if is_seed else None
            filename = f"{'seed' if is_seed else 'chat'}_{job_id}.mp4"
            output_url = gcs.upload_video(buffer, filename, bucket=bucket)
            report({"jobId": job_id, "status": "completed", "outputUrl": output_url})
            print(f"✓ job {job_id} completed -> {output_url}", flush=True)
            return {"jobId": job_id, "status": "completed", "outputUrl": output_url}
        except Exception as e:  # noqa: BLE001 - retry then report failed
            last_err = e
            print(f"↻ job {job_id} attempt {attempt}/{MAX_ATTEMPTS} failed: {e}", flush=True)
            traceback.print_exc()

    msg = str(last_err)
    report({"jobId": job_id, "status": "failed", "error": msg})
    print(f"✗ job {job_id} failed permanently: {msg}", flush=True)
    return {"jobId": job_id, "status": "failed", "error": msg}


def handler(event):
    job = (event or {}).get("input") or {}
    if not job.get("jobId") or not job.get("workflow"):
        return {"error": "input must include jobId and workflow"}
    return process_job(job, event)


if __name__ == "__main__":
    log_config()
    # Fail fast if GCS is misconfigured — better to recycle now than to render on the
    # GPU and only discover we can't deliver the output at upload time.
    gcs.log_config()
    print("[handler] waiting for ComfyUI to be ready ...", flush=True)
    wait_for_ready(timeout=COMFY_READY_TIMEOUT)
    print("[handler] ComfyUI ready — starting RunPod job listener (serverless)", flush=True)
    runpod.serverless.start({"handler": handler})
