#!/usr/bin/env bash
# Container entrypoint: boot ComfyUI (its own prebuilt venv) in the background exactly
# as on the host (`source runComfy`), then hand off to the render worker, which waits
# for ComfyUI to be ready before it starts accepting jobs.
#
# The worker's INTAKE is QUEUE_DRIVER (see runpod_worker/config.py):
#   redis  -> drain ai-chat's render:<lane> streams (the shared pool; scale by running
#             more of this container anywhere that can reach REDIS_URL)
#   runpod -> RunPod Serverless (poll RunPod's own queue outbound)
#   http   -> chat-api POSTs one job to :RENDER_PORT/render
# MOCK_COMFY=1 renders with ffmpeg instead, so ComfyUI and the model fetch are skipped
# entirely — the way to smoke-test queue wiring on a box with no GPU.
set -uo pipefail
: "${COMFY_DIR:?COMFY_DIR not set}"

MOCK="${MOCK_COMFY:-0}"

echo "[entrypoint] boot: worker=${WORKER_ID:-comfy-worker} intake=${QUEUE_DRIVER:-runpod} mock=${MOCK} report-target=${CHAT_API_INTERNAL_URL:-https://api.justinwijaya.com}"
echo "[entrypoint] gcs: response=${GCS_BUCKET_RESPONSE:-${GCS_BUCKET:-video-response}} seed=${GCS_BUCKET_SEED:-${GCS_SEED_BUCKET:-video-seed}} creds=$( [ -n "${RUNPOD_SECRET_gcs_api_key:-}${GCS_SA_KEY_JSON:-}${GCS_KEY_FILE:-}${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && echo present || echo MISSING )"

if [ "$MOCK" = "1" ]; then
  echo "[entrypoint] MOCK_COMFY=1 — skipping model fetch + ComfyUI boot (ffmpeg renders)"
else
  # Resolve the foundation models from the HF cache (RunPod Model Caching pre-warms it)
  # and symlink them into models/ BEFORE ComfyUI boots, so ComfyUI sees them when it
  # scans model dirs at startup. Fatal on failure — no models, no renders. Set
  # FETCH_MODELS=0 when models/ is bind-mounted from a host tree that already has them.
  if [ "${FETCH_MODELS:-1}" = "0" ]; then
    echo "[entrypoint] FETCH_MODELS=0 — using models/ as mounted"
  else
    echo "[entrypoint] resolving foundation models from HF cache"
    if ! /opt/handler-venv/bin/python "$COMFY_DIR/runpod_worker/fetch_models.py"; then
      echo "[entrypoint] FATAL: model fetch failed" >&2
      exit 1
    fi
  fi

  echo "[entrypoint] booting ComfyUI from $COMFY_DIR"
  (
    cd "$COMFY_DIR"
    # Activate the prebuilt venv so `python3` resolves to it (runComfy calls python3).
    # shellcheck disable=SC1091
    source venv/bin/activate
    # runComfy = python3 main.py --listen 0.0.0.0 --reserve-vram 1 --use-sage-attention --disable-pinned-memory
    # shellcheck disable=SC1091
    source runComfy
  ) &
  COMFY_PID=$!

  # If ComfyUI exits, stop the whole container so the orchestrator recycles the worker
  # instead of leaving a worker that fails every job it claims off the queue.
  trap 'kill -TERM "$COMFY_PID" 2>/dev/null' EXIT
fi

echo "[entrypoint] starting render worker (handler-venv)"
exec /opt/handler-venv/bin/python -u "$COMFY_DIR/runpod_worker/handler.py"
