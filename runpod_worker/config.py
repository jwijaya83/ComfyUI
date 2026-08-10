"""Worker configuration — the Python twin of ai-chat/services/render-worker/src/config.js.

This worker is STATELESS: it owns no database and reads no personality files. A
fully-resolved render job arrives over one of three INTAKES (`QUEUE_DRIVER`), it drives
ComfyUI (or an ffmpeg mock), delivers the MP4 (GCS and/or a shared media dir), and
reports status back to chat-api's /internal/render-events webhook.

    runpod  the RunPod Serverless queue (runpod.serverless.start) — the original
            intake. Jobs arrive as {"input": <job>} on an outbound poll.
    redis   the MAIN render queue: per-lane Redis Streams (render:high / render:low)
            drained as part of a shared consumer group, exactly like render-worker.
            THIS is what lets any box (local, RunPod pod, another cloud) join the
            same pool by pointing REDIS_URL at production.
    http    chat-api POSTs one job to /render (the no-broker local profile).

Every env name here is IDENTICAL to render-worker's so this image is a drop-in
replacement for that service — the same docker-compose environment block works.
"""
import os


def _flag(name, default="0"):
    return str(os.environ.get(name, default)).strip().lower() in ("1", "true", "on", "yes")


def _int(name, default):
    try:
        return int(float(os.environ.get(name) or default))
    except (TypeError, ValueError):
        return int(default)


# --- identity ----------------------------------------------------------------
# MUST be unique per running instance when several workers share a consumer group;
# consumer.py appends hostname+pid (every container is PID 1, so pid alone collides).
WORKER_ID = os.environ.get("WORKER_ID") or "comfy-worker-1"

# --- intake ------------------------------------------------------------------
QUEUE_DRIVER = (os.environ.get("QUEUE_DRIVER") or "runpod").strip().lower()
REDIS_URL = os.environ.get("REDIS_URL") or "redis://localhost:6379"
# Streams are "<prefix>:<lane>", drained left-to-right (high before low). MUST match
# chat-api's QUEUE_STREAM_PREFIX + lane names or we drain the wrong queue silently.
QUEUE_STREAM_PREFIX = os.environ.get("QUEUE_STREAM_PREFIX") or "render"
RENDER_LANES = [s.strip() for s in (os.environ.get("RENDER_LANES") or "high,low").split(",") if s.strip()]
RENDER_GROUP = os.environ.get("RENDER_GROUP") or "renderers"
# Jobs processed concurrently PER WORKER. Keep at 1 per GPU; the GLOBAL render ceiling
# is (worker instances) x RENDER_CONCURRENCY, so scaling is "add another box".
RENDER_CONCURRENCY = _int("RENDER_CONCURRENCY", 1)
# How long to block waiting for new work when every lane is idle (busy-loop guard).
RENDER_BLOCK_MS = _int("RENDER_BLOCK_MS", 5000)
# Visibility timeout: if a worker dies mid-render another reclaims the job after it has
# been idle this long. MUST exceed the worst-case process_job (JOB_MAX_ATTEMPTS x a full
# render) or a still-rendering job gets reclaimed and rendered twice.
RENDER_VISIBILITY_MS = _int("RENDER_VISIBILITY_MS", 30 * 60 * 1000)
# Re-deliveries (worker crashes) before a poison job is dead-lettered.
RENDER_MAX_DELIVERIES = _int("RENDER_MAX_DELIVERIES", 3)
RENDER_DEAD_STREAM = os.environ.get("RENDER_DEAD_STREAM") or "render:dead"
# Health probe + the http intake listen here (always up, all drivers).
RENDER_PORT = _int("RENDER_PORT", _int("PORT", 8080))

# --- render ------------------------------------------------------------------
# MOCK_COMFY=1 renders a real playable MP4 with ffmpeg (no GPU, no models) — the way
# to exercise the whole queue -> render -> report path without the GPU.
MOCK_COMFY = _flag("MOCK_COMFY")
JOB_MAX_ATTEMPTS = _int("JOB_MAX_ATTEMPTS", 3)

# --- delivery ----------------------------------------------------------------
# Local shared media dir (chat-api re-serves it at /media). Set ONLY when this worker
# shares a volume with chat-api; a remote worker leaves it unset and delivers via GCS.
MEDIA_DIR = os.environ.get("MEDIA_DIR") or ""
# Public URL base for the local /media link — the chat-api the CLIENT reaches, not this
# worker (chat-api re-serves the file). Unused when delivery goes through GCS.
PUBLIC_BASE = (os.environ.get("PUBLIC_BASE") or "http://localhost:3000").rstrip("/")

# --- report / asset fallback -------------------------------------------------
# Where status goes, and where we fetch job assets from when they didn't arrive as
# signed GCS urls (see handler._resolve_assets). reporter.py owns the same default.
CHAT_API_URL = (os.environ.get("CHAT_API_INTERNAL_URL") or "https://api.justinwijaya.com").rstrip("/")
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN") or ""

# --- single-GPU lease --------------------------------------------------------
# A Redis mutex shared with ai-chat's llm-worker so ComfyUI and Ollama never sit on the
# one GPU together. GPU_LEASE_KEY + REDIS_URL must match across both. Leave off (the
# default) on a GPU-dedicated box — on RunPod the GPU is ours alone.
GPU_LEASE = _flag("GPU_LEASE")
GPU_LEASE_KEY = os.environ.get("GPU_LEASE_KEY") or "gpu:lease"
GPU_LEASE_TTL_MS = _int("GPU_LEASE_TTL_MS", 60000)
GPU_LEASE_POLL_MS = _int("GPU_LEASE_POLL_MS", 300)
# Evict a resident Ollama model before ComfyUI grows VRAM. No-op when unset.
OLLAMA_HOST = (os.environ.get("OLLAMA_HOST") or "").rstrip("/")


def summary():
    """One line for the boot log — the config that decides where work comes from and
    where the finished MP4 goes."""
    intake = QUEUE_DRIVER
    if QUEUE_DRIVER == "redis":
        intake += f" [{','.join(RENDER_LANES)}] group={RENDER_GROUP} {REDIS_URL}"
    return (
        f"[config] worker={WORKER_ID} intake={intake} concurrency={RENDER_CONCURRENCY} "
        f"mock={int(MOCK_COMFY)} lease={int(GPU_LEASE)} media={MEDIA_DIR or '(gcs only)'} "
        f"report -> {CHAT_API_URL}"
    )
