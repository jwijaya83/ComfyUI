"""Persist a finished render and return the value chat-api stores in
render_jobs.output_url. Python port of render-worker/src/storage.js — same two
delivery paths, same precedence, so swapping the workers changes nothing downstream.

  1. LOCAL  — when MEDIA_DIR is set (a volume shared with chat-api) write the MP4 there
     and build `PUBLIC_BASE/media/<file>.mp4`. chat-api re-serves that path. This is
     also the safe fallback if a GCS upload errors.
  2. GCS    — when a response bucket + credentials are configured, ALSO upload and
     return the DURABLE `gs://bucket/object` ref. chat-api signs a fresh short-lived
     read url from it on every read (dto.signedMediaUrl), so what we store never
     expires. This is the only path that works for a REMOTE worker sharing no volume
     with chat-api — which is the whole point of "run this image anywhere".

A worker with NEITHER configured cannot deliver its output at all, so we raise rather
than report a `completed` job whose url nobody can fetch.
"""
import os

import gcs
from config import MEDIA_DIR, PUBLIC_BASE


def describe():
    """Human-readable delivery target for the boot log."""
    parts = []
    if MEDIA_DIR:
        parts.append(f"local {MEDIA_DIR} -> {PUBLIC_BASE}/media")
    if gcs.enabled(gcs.response_bucket()):
        parts.append(f"gcs gs://{gcs.response_bucket()}")
    return " + ".join(parts) or "(none configured — renders cannot be delivered)"


def save_video(data, filename):
    local_url = None
    if MEDIA_DIR:
        os.makedirs(MEDIA_DIR, exist_ok=True)
        with open(os.path.join(MEDIA_DIR, filename), "wb") as f:
            f.write(data)
        local_url = f"{PUBLIC_BASE}/media/{filename}"

    bucket = gcs.response_bucket()
    if not gcs.enabled(bucket):
        if local_url:
            return local_url
        raise RuntimeError(
            "no delivery configured: set MEDIA_DIR (a volume shared with chat-api) "
            "or a GCS response bucket + credentials"
        )

    try:
        gs_uri = gcs.upload_video(data, filename, bucket=bucket)
        print(f"⬆ {filename} -> {gs_uri}", flush=True)
        return gs_uri
    except Exception as e:  # noqa: BLE001 - a storage hiccup must not fail a good render
        if local_url:
            print(f"⚠ GCS upload of {filename} failed, serving local /media instead: {e}", flush=True)
            return local_url
        raise
