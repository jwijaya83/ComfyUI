"""Status callback: handler -> chat-api /internal/render-events.

The SAME webhook the existing render-worker uses (render-worker/callback.js), so
chat-api's row-update + SSE/notification flow is unchanged. Best-effort: a dropped
status update must never crash a render. RunPod's own terminal webhook (set on the
/run submit) is the backstop for a worker that dies before reporting `completed`.
"""
import os

import requests

CHAT_API_URL = os.environ.get("CHAT_API_INTERNAL_URL", "").rstrip("/")
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")
WORKER_ID = os.environ.get("WORKER_ID", "runpod-worker")


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
