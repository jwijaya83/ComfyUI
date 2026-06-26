#!/usr/bin/env bash
# Container entrypoint: boot ComfyUI (its own prebuilt venv) in the background exactly
# as on the host (`source runComfy`), then hand off to the RunPod handler, which waits
# for ComfyUI to be ready before it starts accepting jobs.
set -uo pipefail
: "${COMFY_DIR:?COMFY_DIR not set}"

# Resolve the base checkpoint + text encoder from the HF cache (RunPod Model Caching
# pre-warms it) and symlink them into models/ BEFORE ComfyUI boots, so ComfyUI sees
# them when it scans model dirs at startup. Fatal on failure — no models, no renders.
echo "[entrypoint] resolving foundation models from HF cache"
if ! /opt/handler-venv/bin/python "$COMFY_DIR/runpod_worker/fetch_models.py"; then
  echo "[entrypoint] FATAL: model fetch failed" >&2
  exit 1
fi

echo "[entrypoint] booting ComfyUI from $COMFY_DIR"
(
  cd "$COMFY_DIR"
  # Activate the prebuilt venv so `python3` resolves to it (runComfy calls python3).
  # The venv lives at its original absolute path, so activate/shebangs are all valid.
  # shellcheck disable=SC1091
  source venv/bin/activate
  # runComfy = python3 main.py --reserve-vram 1 --enable-manager --use-sage-attention --disable-pinned-memory
  # shellcheck disable=SC1091
  source runComfy
) &
COMFY_PID=$!

# If ComfyUI exits, stop the whole container so RunPod recycles the worker instead of
# leaving a handler that fails every job.
trap 'kill -TERM "$COMFY_PID" 2>/dev/null' EXIT

echo "[entrypoint] starting RunPod handler (handler-venv)"
exec /opt/handler-venv/bin/python -u "$COMFY_DIR/runpod_worker/handler.py"
