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

# Active-watch tuning (see watch_prompt). Env names mirror render-worker's config.js so
# the two workers behave the same. Poll requests are time-boxed so a HUNG ComfyUI reads
# as unreachable rather than hanging the watch.
_POLL_MS = int(os.environ.get("COMFY_POLL_MS", "5000"))
_UNREACHABLE_TRIES = int(os.environ.get("COMFY_UNREACHABLE_TRIES", "3"))
_WATCH_TIMEOUT_S = int(os.environ.get("COMFY_WATCH_TIMEOUT_MS", str(30 * 60 * 1000))) / 1000.0
_POLL_TIMEOUT_S = int(os.environ.get("COMFY_POLL_TIMEOUT_MS", "8000")) / 1000.0


class ComfyWatchError(RuntimeError):
    """A render ComfyUI did NOT complete. `kind` is the cause (interrupted | node_error
    | vanished | unreachable | timeout); `retriable` tells process_job whether to
    re-render (a crash/error may succeed on retry; a deliberate manual cancel must not
    be re-rendered)."""

    def __init__(self, message, kind="error", retriable=True):
        super().__init__(message)
        self.kind = kind
        self.retriable = retriable


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
    # Time-boxed like /queue: on the watch poll path a HUNG ComfyUI (accepts the
    # connection but never answers) must read as unreachable, not block for a minute.
    r = requests.get(f"{HTTP_BASE}/history/{prompt_id}", timeout=_POLL_TIMEOUT_S)
    if not r.ok:
        raise RuntimeError(f"ComfyUI /history failed: {r.status_code}")
    return r.json().get(prompt_id)


def get_queue():
    """The live queue: {queue_running:[...], queue_pending:[...]}. Each item is
    [number, prompt_id, prompt, extra_data, outputs] — index 1 is the prompt_id."""
    r = requests.get(f"{HTTP_BASE}/queue", timeout=_POLL_TIMEOUT_S)
    if not r.ok:
        raise RuntimeError(f"ComfyUI /queue failed: {r.status_code}")
    return r.json()


def interrupt(prompt_id=None):
    """Best-effort: ask ComfyUI to stop a prompt we've given up on (our own timeout) so
    the GPU frees for the next render. Targeted by prompt_id — never interrupts another
    render."""
    try:
        requests.post(
            f"{HTTP_BASE}/interrupt",
            json=({"prompt_id": prompt_id} if prompt_id else {}),
            timeout=_POLL_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001 - we're already abandoning this render
        pass


def _summarize_status_messages(messages):
    """Pull an exception/interrupt note out of a history entry's status.messages (a list
    of [event, data] tuples) for a human-readable failure reason."""
    if not isinstance(messages, list):
        return ""
    for m in messages:
        if isinstance(m, list) and len(m) >= 2:
            event, mdata = m[0], m[1]
            if event == "execution_error" and isinstance(mdata, dict) and mdata.get("exception_message"):
                return f": {mdata['exception_message']}"
            if event == "execution_interrupted":
                return " (interrupted)"
    return ""


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


def watch_prompt(
    client_id,
    prompt_id,
    on_event=None,
    timeout=None,
    poll_ms=None,
    unreachable_tries=None,
):
    """Return when ComfyUI finishes OUR prompt successfully; raise ComfyWatchError the
    moment it is cancelled / errors / crashes.

    The WebSocket alone is not enough: a manual cancel from the ComfyUI UI, a job cleared
    while still PENDING, or a ComfyUI crash/restart can leave the socket with no terminal
    event, so the old watcher just blocked until its 30-minute timeout (the render_jobs
    row stuck 'running', the client polling forever). So we ALSO actively poll /history +
    /queue: the WS gives fast progress + fast terminal signals, and the poll is the robust
    backstop that notices the job is gone (or ComfyUI is unreachable) within poll_ms.
    """
    timeout = _WATCH_TIMEOUT_S if timeout is None else timeout
    poll_ms = _POLL_MS if poll_ms is None else poll_ms
    unreachable_tries = _UNREACHABLE_TRIES if unreachable_tries is None else unreachable_tries
    poll_interval = max(0.5, poll_ms / 1000.0)
    deadline = time.time() + timeout

    state = {"unreachable": 0, "vanished": 0}

    def check_once():
        """True = succeeded; raise ComfyWatchError = terminal failure; None = keep waiting.
        This is what catches a cancel/crash the WS never told us about."""
        try:
            hist = get_history(prompt_id)
            state["unreachable"] = 0  # a response means ComfyUI is reachable
            status = (hist or {}).get("status")
            if status:
                status_str = status.get("status_str")
                completed = status.get("completed")
                if completed and status_str == "success":
                    return True
                if status_str == "error" or completed is False:
                    # A manual interrupt and a node error BOTH land as status_str "error" /
                    # completed false — only the recorded event tells them apart. An
                    # interrupt was deliberate, so it must not be re-rendered.
                    msgs = status.get("messages") or []
                    interrupted = any(isinstance(m, list) and m and m[0] == "execution_interrupted" for m in msgs)
                    note = _summarize_status_messages(msgs)
                    raise ComfyWatchError(
                        f"ComfyUI reported {status_str}{note}",
                        kind="interrupted" if interrupted else "node_error",
                        retriable=not interrupted,
                    )
                return None  # has a status object but not terminal-shaped yet
            # No history entry yet: still queued/running, or already gone?
            q = get_queue()
            items = (q.get("queue_running") or []) + (q.get("queue_pending") or [])
            present = any(isinstance(it, list) and len(it) > 1 and it[1] == prompt_id for it in items)
            if present:
                state["vanished"] = 0
                return None
            # Not in history AND not in the queue: cleared/cancelled while pending, or
            # ComfyUI restarted and lost it. Two consecutive misses to avoid a race right
            # after submit (before it registers in the queue).
            state["vanished"] += 1
            if state["vanished"] >= 2:
                raise ComfyWatchError(
                    "prompt vanished from ComfyUI (queue cleared, cancelled, or server restarted)",
                    kind="vanished",
                    retriable=False,
                )
            return None
        except ComfyWatchError:
            raise
        except Exception as e:  # noqa: BLE001 - the poll itself failed -> unreachable
            state["unreachable"] += 1
            if state["unreachable"] >= unreachable_tries:
                raise ComfyWatchError(
                    f"ComfyUI unreachable after {state['unreachable']} checks: {e}",
                    kind="unreachable",
                    retriable=True,
                )
            return None

    ws = None
    try:
        try:
            ws = create_connection(f"{WS_BASE}?clientId={client_id}", timeout=poll_interval)
        except Exception as e:  # noqa: BLE001 - fall back to polling only
            print(f"[comfy] watch {prompt_id}: WS open failed ({e}); polling only", flush=True)
            ws = None

        last_poll = time.time()
        while True:
            if time.time() >= deadline:
                # A wedged ComfyUI can't be polled out of — give up and interrupt the
                # abandoned render so the GPU is free for the next one.
                interrupt(prompt_id)
                raise ComfyWatchError(
                    f"ComfyUI watch timed out after {int(timeout)}s", kind="timeout", retriable=True
                )

            # --- WS read, bounded to one poll interval (fast terminal signals) -------
            if ws is not None:
                remaining = deadline - time.time()
                ws.settimeout(max(0.1, min(poll_interval, remaining)))
                try:
                    raw = ws.recv()
                except WebSocketTimeoutException:
                    raw = None  # quiet interval — fall through to the active poll
                except Exception as e:  # noqa: BLE001 - WS died; rely on the poll
                    print(f"[comfy] watch {prompt_id}: WS error ({e}); relying on queue poll", flush=True)
                    try:
                        ws.close()
                    except Exception:  # noqa: BLE001
                        pass
                    ws = None
                    raw = None

                if raw == "" or raw is None:
                    pass  # closed frame / timeout — poll below decides the fate
                elif isinstance(raw, (bytes, bytearray)):
                    continue  # preview frame, not a status event
                else:
                    try:
                        msg = json.loads(raw)
                    except Exception:  # noqa: BLE001
                        continue
                    data = msg.get("data") or {}
                    pid = data.get("prompt_id")
                    if pid and pid != prompt_id:
                        continue  # an event for another client's prompt
                    if on_event:
                        on_event(msg)
                    mtype = msg.get("type")
                    if mtype == "executing" and data.get("node") is None and pid == prompt_id:
                        return
                    if mtype == "execution_success" and pid == prompt_id:
                        return
                    if mtype == "execution_interrupted" and pid == prompt_id:
                        raise ComfyWatchError(
                            "ComfyUI execution interrupted (manually cancelled)",
                            kind="interrupted",
                            retriable=False,
                        )
                    if mtype == "execution_error" and pid == prompt_id:
                        detail = data.get("exception_message") or data.get("node_type") or "node error"
                        raise ComfyWatchError(f"ComfyUI execution error: {detail}", kind="node_error", retriable=True)
            else:
                time.sleep(min(poll_interval, max(0.0, deadline - time.time())))

            # --- Active poll on a fixed cadence, regardless of WS chatter ------------
            now = time.time()
            if now - last_poll >= poll_interval:
                last_poll = now
                if check_once():
                    return
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
