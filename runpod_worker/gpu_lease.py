"""Single-GPU co-tenancy: a cross-service Redis mutex + Ollama eviction.

Python port of render-worker/src/gpuLease.js and freeOllama.js (which are themselves
identical copies of chat-api's) — so this image can take over that service's role on a
box where ComfyUI shares ONE GPU with ai-chat's llm-worker (Ollama). Neither may sit on
the card at the same time.

GPU_LEASE_KEY + REDIS_URL must match the llm-worker's, like INTERNAL_TOKEN does.
Disabled by default (GPU_LEASE unset/0 -> straight passthrough): on RunPod, or any
GPU-dedicated box, there is nothing to coordinate with and the lease is pure overhead.
"""
import contextlib
import threading
import time
import uuid

import requests

from config import (
    GPU_LEASE,
    GPU_LEASE_KEY,
    GPU_LEASE_POLL_MS,
    GPU_LEASE_TTL_MS,
    OLLAMA_HOST,
    REDIS_URL,
)

# Owner-checked so we can only ever renew/release a lease we still hold — a lease that
# expired mid-render (and was taken by someone else) must not be deleted by us.
_RENEW = "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('pexpire',KEYS[1],ARGV[2]) else return 0 end"
_RELEASE = "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end"

_client = None
_client_lock = threading.Lock()

enabled = GPU_LEASE


def _redis():
    global _client
    with _client_lock:
        if _client is None:
            import redis as redis_lib

            _client = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True)
        return _client


@contextlib.contextmanager
def gpu_lease(label=""):
    """Hold the GPU lease for the duration of the block (blocking until acquired).

    A heartbeat thread extends the TTL so a multi-minute render survives; the release is
    owner-checked; a crashed holder's lease simply expires after the TTL so the GPU can
    never be deadlocked by a dead worker.
    """
    if not enabled:
        yield
        return

    client = _redis()
    token = uuid.uuid4().hex
    t0 = time.time()
    poll_s = max(0.05, GPU_LEASE_POLL_MS / 1000.0)
    while not client.set(GPU_LEASE_KEY, token, nx=True, px=GPU_LEASE_TTL_MS):
        time.sleep(poll_s)

    waited_ms = int((time.time() - t0) * 1000)
    suffix = f" for {label}" if label else ""
    print(
        f"[gpu-lease] acquired{suffix}" + (f" (waited {waited_ms}ms)" if waited_ms > GPU_LEASE_POLL_MS else ""),
        flush=True,
    )

    stop = threading.Event()

    def heartbeat():
        interval = max(1.0, GPU_LEASE_TTL_MS / 3000.0)
        while not stop.wait(interval):
            try:
                client.eval(_RENEW, 1, GPU_LEASE_KEY, token, str(GPU_LEASE_TTL_MS))
            except Exception:  # noqa: BLE001 - a missed heartbeat is covered by the next
                pass

    hb = threading.Thread(target=heartbeat, name="gpu-lease-hb", daemon=True)
    hb.start()
    try:
        yield
    finally:
        stop.set()
        try:
            client.eval(_RELEASE, 1, GPU_LEASE_KEY, token)
        except Exception:  # noqa: BLE001 - the TTL frees it anyway
            pass
        print(f"[gpu-lease] released{suffix} (held {int((time.time() - t0) * 1000)}ms)", flush=True)


# --------------------------------------------------------------------- ollama ---
# The lease guarantees we don't run CONCURRENTLY with a text turn, but a just-finished
# one can leave its model resident in VRAM — so the render, having taken the lease, asks
# Ollama to unload first or ComfyUI fails with "VRAM grow failed". No model names needed:
# evict whatever /api/ps reports loaded. No-op unless OLLAMA_HOST is set.


def _loaded():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
        if not r.ok:
            return []
        return [m.get("name") or m.get("model") for m in (r.json().get("models") or []) if m.get("name") or m.get("model")]
    except Exception:  # noqa: BLE001
        return []


def _unload(model):
    try:
        # keep_alive:0 tells Ollama to evict the model from (V)RAM immediately.
        requests.post(f"{OLLAMA_HOST}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except Exception:  # noqa: BLE001 - freeing VRAM must never break the render
        pass


def free_ollama():
    if not OLLAMA_HOST:
        return
    models = _loaded()
    if not models:
        return
    print(f"[gpu] evicting Ollama before ComfyUI: {', '.join(models)}", flush=True)
    for model in models:
        _unload(model)
    deadline = time.time() + 15
    while time.time() < deadline:
        if not _loaded():
            return
        time.sleep(0.25)
    print(f"[gpu] Ollama still resident after wait: {', '.join(_loaded())}", flush=True)
