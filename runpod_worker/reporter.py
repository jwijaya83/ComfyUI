"""Status callback: handler -> chat-api /internal/render-events.

The SAME webhook the existing render-worker uses (render-worker/callback.js), so
chat-api's row-update + SSE/notification flow is unchanged. Best-effort: a dropped
status update must never crash a render. RunPod's own terminal webhook (set on the
/run submit) is the backstop for a worker that dies before reporting `completed`.
"""
import os

import requests

# Where status updates go. Defaults to the prod chat-api front door
# (api.justinwijaya.com) so a bare endpoint still reports; override with
# CHAT_API_INTERNAL_URL for staging/local. The /internal/render-events route is
# public through Caddy but guarded by x-internal-token (must equal INTERNAL_TOKEN).
DEFAULT_CHAT_API_URL = "https://api.justinwijaya.com"
CHAT_API_URL = (os.environ.get("CHAT_API_INTERNAL_URL") or DEFAULT_CHAT_API_URL).rstrip("/")
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")
WORKER_ID = os.environ.get("WORKER_ID", "runpod-worker")


def log_config():
    """Print (once, at boot) where status reports go — visible in RunPod stdout."""
    if not CHAT_API_URL:
        print("[reporter] WARNING: CHAT_API_INTERNAL_URL empty — status reports DISABLED", flush=True)
        return
    print(f"[reporter] status -> {CHAT_API_URL}/internal/render-events (worker={WORKER_ID})", flush=True)
    if not INTERNAL_TOKEN:
        print("[reporter] WARNING: INTERNAL_TOKEN is empty — chat-api will reject reports (401)", flush=True)


def report(evt):
    if not CHAT_API_URL:
        return
    try:
        requests.post(
            f"{CHAT_API_URL}/internal/render-events",
            json={"workerId": WORKER_ID, **evt},
            headers={"content-type": "application/json", "x-internal-token": INTERNAL_TOKEN},
            timeout=10,
        )
    except Exception as e:  # noqa: BLE001 - status reporting is best-effort
        print(
            f"[reporter] status report failed ({evt.get('status')} {evt.get('jobId')}): {e}",
            flush=True,
        )
