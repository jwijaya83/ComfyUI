"""Upload the finished MP4 to GCS and return a URL chat-api can hand to clients.

Replaces render-worker/storage.js (local disk). On RunPod the worker is ephemeral
and not co-located with chat-api, so the bytes go to object storage instead.

Auth: set GOOGLE_APPLICATION_CREDENTIALS to a service-account key file, OR pass the
key JSON inline via GCS_SA_KEY_JSON (RunPod env/secret) and we write it out at boot.
A V4 signed URL needs the SA private key, so default-compute creds alone won't sign.
"""
import os
from datetime import timedelta

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "renders").strip("/")
GCS_SIGN = os.environ.get("GCS_SIGN", "1") != "0"
GCS_SIGNED_URL_TTL = int(os.environ.get("GCS_SIGNED_URL_TTL", str(7 * 24 * 3600)))

_client = None


def _ensure_creds():
    key = os.environ.get("GCS_SA_KEY_JSON")
    if key and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        path = "/tmp/gcs-sa.json"
        with open(path, "w") as f:
            f.write(key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


def _bucket():
    global _client
    if _client is None:
        _ensure_creds()
        from google.cloud import storage

        _client = storage.Client()
    return _client.bucket(GCS_BUCKET)


def upload_video(data, filename, content_type="video/mp4"):
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET not set — cannot upload render output")
    path = f"{GCS_PREFIX}/{filename}" if GCS_PREFIX else filename
    blob = _bucket().blob(path)
    blob.upload_from_string(data, content_type=content_type)
    if GCS_SIGN:
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=GCS_SIGNED_URL_TTL),
            method="GET",
        )
    # Public-bucket fallback (object must be world-readable).
    return f"https://storage.googleapis.com/{GCS_BUCKET}/{path}"
