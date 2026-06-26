"""Thin ComfyUI HTTP + WebSocket client (Python port of render-worker/comfyui.js).

ComfyUI runs in the SAME container on 127.0.0.1:8188; we reach it over loopback.
"""
import json
import os
import time
import uuid
from urllib.parse import urlencode

import requests
from websocket import WebSocketTimeoutException, create_connection

HTTP_BASE = os.environ.get("COMFYUI_HTTP", "http://127.0.0.1:8188")
WS_BASE = os.environ.get("COMFYUI_WS", "ws://127.0.0.1:8188/ws")


def wait_for_ready(timeout=300, interval=2):
    """Block until ComfyUI answers /system_stats (cold-start boot can take a while)."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{HTTP_BASE}/system_stats", timeout=5)
            if r.ok:
                return True
        except Exception as e:  # noqa: BLE001 - boot not up yet
            last = e
        time.sleep(interval)
    raise RuntimeError(f"ComfyUI not ready at {HTTP_BASE} after {timeout}s ({last})")


def _upload_to_input(data, filename, subfolder=None):
    files = {"image": (filename, data)}
    form = {"overwrite": "true"}
    if subfolder:
        form["subfolder"] = subfolder
    r = requests.post(f"{HTTP_BASE}/upload/image", files=files, data=form, timeout=120)
    if not r.ok:
        raise RuntimeError(f"ComfyUI /upload/image failed: {r.status_code} {r.text}")
    d = r.json()
    return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]


# ComfyUI's /upload/image stores any posted file in input/ regardless of media type;
# VHS_LoadVideo lists video files there by name. Both uploads use the same endpoint.
def upload_image(data, filename, subfolder=None):
    return _upload_to_input(data, filename, subfolder)


def upload_video(data, filename, subfolder=None):
    return _upload_to_input(data, filename, subfolder)


def submit_prompt(workflow):
    client_id = str(uuid.uuid4())
    r = requests.post(
        f"{HTTP_BASE}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"ComfyUI /prompt failed: {r.status_code} {r.text}")
    return r.json()["prompt_id"], client_id


def get_history(prompt_id):
    r = requests.get(f"{HTTP_BASE}/history/{prompt_id}", timeout=60)
    if not r.ok:
        raise RuntimeError(f"ComfyUI /history failed: {r.status_code}")
    return r.json().get(prompt_id)


def collect_outputs(history_entry):
    if not history_entry or "outputs" not in history_entry:
        return []
    files = []
    for node_output in history_entry["outputs"].values():
        for key in ("videos", "gifs", "images"):
            for f in node_output.get(key, []) or []:
                files.append({**f, "kind": key})
    return files


def _view_url(file):
    params = {
        "filename": file.get("filename", ""),
        "subfolder": file.get("subfolder", ""),
        "type": file.get("type", "output"),
    }
    return f"{HTTP_BASE}/view?{urlencode(params)}"


def download_output(file):
    r = requests.get(_view_url(file), timeout=300)
    if not r.ok:
        raise RuntimeError(f"ComfyUI /view failed: {r.status_code}")
    return r.content


def watch_prompt(client_id, prompt_id, on_event=None, timeout=30 * 60):
    """Resolve when ComfyUI signals execution complete for our prompt.

    Mirrors the Node WS loop: forward every event to on_event; finish on the
    `executing` message whose node is null for our prompt_id. Binary frames
    (preview images) are skipped.
    """
    ws = create_connection(f"{WS_BASE}?clientId={client_id}", timeout=timeout)
    deadline = time.time() + timeout
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError("ComfyUI WS timed out")
            ws.settimeout(remaining)
            try:
                raw = ws.recv()
            except WebSocketTimeoutException:
                raise RuntimeError("ComfyUI WS timed out")
            if isinstance(raw, (bytes, bytearray)):
                continue  # preview frame, not a status event
            try:
                msg = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            data = msg.get("data") or {}
            if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                continue
            if on_event:
                on_event(msg)
            if (
                msg.get("type") == "executing"
                and data.get("node") is None
                and data.get("prompt_id") == prompt_id
            ):
                break
    finally:
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass
