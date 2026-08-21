"""Persist a finished render and return the value chat-api stores in
render_jobs.output_url. Python port of render-worker/src/storage.js — same two
delivery paths, but GCS is tried FIRST and local disk is a true fallback, not a
redundant write on every render:

  1. GCS    — when a response bucket + credentials are configured, upload and return
     the DURABLE `gs://bucket/object` ref. chat-api signs a fresh short-lived read url
     from it on every read (dto.signedMediaUrl), so what we store never expires. This
     is the only path that works for a REMOTE worker sharing no volume with chat-api —
     which is the whole point of "run this image anywhere" — so it's the default.
  2. LOCAL  — when MEDIA_DIR is set (a volume shared with chat-api), write the MP4
     there and build `PUBLIC_BASE/media/<file>.mp4`. chat-api re-serves that path.
     Used when GCS isn't configured at all, or as the fallback if a configured GCS
     upload errors — never in addition to a GCS upload that succeeded.

A worker with NEITHER configured cannot deliver its output at all, so we raise rather
than report a `completed` job whose url nobody can fetch.
"""
import os

import gcs
from config import MEDIA_DIR, PUBLIC_BASE


def describe():
    """Human-readable delivery target for the boot log."""
    bucket = gcs.response_bucket()
    if gcs.enabled(bucket):
        target = f"gcs gs://{bucket}"
        if MEDIA_DIR:
            target += f" (falls back to local {MEDIA_DIR} -> {PUBLIC_BASE}/media on upload failure)"
        return target
    if MEDIA_DIR:
        return f"local {MEDIA_DIR} -> {PUBLIC_BASE}/media"
    return "(none configured — renders cannot be delivered)"


def _save_local(data, filename):
    os.makedirs(MEDIA_DIR, exist_ok=True)
    with open(os.path.join(MEDIA_DIR, filename), "wb") as f:
        f.write(data)
    return f"{PUBLIC_BASE}/media/{filename}"


def save_video(data, filename):
    bucket = gcs.response_bucket()
    if gcs.enabled(bucket):
        try:
            gs_uri = gcs.upload_video(data, filename, bucket=bucket)
            print(f"⬆ {filename} -> {gs_uri}", flush=True)
            return gs_uri
        except Exception as e:  # noqa: BLE001 - a storage hiccup must not fail a good render
            if not MEDIA_DIR:
                raise
            print(f"⚠ GCS upload of {filename} failed, falling back to local /media: {e}", flush=True)
            return _save_local(data, filename)

    if MEDIA_DIR:
        return _save_local(data, filename)

    raise RuntimeError(
        "no delivery configured: set MEDIA_DIR (a volume shared with chat-api) "
        "or a GCS response bucket + credentials"
    )
