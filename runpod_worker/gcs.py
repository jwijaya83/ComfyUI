"""Upload finished MP4s to GCS and return the DURABLE gs:// reference to persist.

Replaces render-worker/storage.js (local disk). On RunPod the worker is ephemeral
and not co-located with chat-api, so the bytes go to object storage instead. We hand
back the durable `gs://bucket/object` uri (it NEVER expires); chat-api stores it and
mints a FRESH short-lived V4 signed READ url from it on EVERY read (dto.signedMediaUrl)
— so nothing signed is ever persisted and there is no url to "refresh". The Node worker
(render-worker/storage.js) returns this same gs:// shape; signing lives on the chat-api
READ side now, NOT here. (Returning a pre-signed https url here would rot after its TTL
with nothing for chat-api to re-sign from — the exact bug this avoids.)

Two buckets (both project aichat-500601, SA gcs-api-user@aichat-500601.iam.
gserviceaccount.com, role roles/storage.objectAdmin):
  - GCS_BUCKET       chat-response renders (per-turn video)  ->  "video-response"
  - GCS_SEED_BUCKET  admin seed-video container clips        ->  "video-seed"

Auth — first source that resolves wins (uploads need objectAdmin; the local `selftest`
also signs a probe url, which needs the SA private key default-compute creds lack):
  1. GOOGLE_APPLICATION_CREDENTIALS  path to an SA key file (respected as-is)
  2. GCS_KEY_FILE                    path to an SA key file (local convenience)
  3. inline SA-key JSON in one of GCS_SA_KEY_JSON / RUNPOD_SECRET_gcs_api_key
     — RunPod injects the `gcs_api_key` secret into the worker as the env var
     RUNPOD_SECRET_gcs_api_key; we write it to a temp file and point (1) at it.

Run this module directly to wire/verify the upload locally, e.g.:
  export GCS_KEY_FILE=~/Downloads/aichat-500601-XXXX.json
  python gcs.py selftest --bucket video-response
  python gcs.py upload path/to/clip.mp4 --bucket video-seed
"""
import json
import os
import sys
import tempfile
from datetime import timedelta

# Canonical buckets for this deployment (project aichat-500601) are baked as defaults
# so the RunPod endpoint only needs creds set, not bucket names. Override per env.
GCS_BUCKET = os.environ.get("GCS_BUCKET", "video-response")
GCS_SEED_BUCKET = os.environ.get("GCS_SEED_BUCKET", "video-seed")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "renders").strip("/")
# GCS_SIGN gates ONLY the local `selftest` signing round-trip — delivery always returns a
# durable gs:// ref that chat-api signs on read (the read-side TTL lives on chat-api's
# GCS_SIGNED_URL_TTL_MS, not here).
GCS_SIGN = os.environ.get("GCS_SIGN", "1") != "0"

# Env vars that may carry the SA-key JSON inline, checked in order. RunPod exposes a
# secret named `gcs_api_key` to the container as RUNPOD_SECRET_gcs_api_key.
_KEY_JSON_ENVS = ("GCS_SA_KEY_JSON", "RUNPOD_SECRET_gcs_api_key")

_client = None
_creds_ready = False


def _ensure_creds():
    """Make sure GOOGLE_APPLICATION_CREDENTIALS points at a usable SA key file."""
    global _creds_ready
    if _creds_ready:
        return

    # 1/2. A key-file path was given explicitly — use it as-is.
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCS_KEY_FILE")
    if path:
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise RuntimeError(f"GCS key file not found: {path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
        _creds_ready = True
        return

    # 3. Inline JSON from env/secret -> materialize a temp file GAC points at. The
    #    RunPod secret is the whole key JSON pasted as a string; json.loads restores
    #    the escaped "\n"s inside private_key, so writing the raw string is correct.
    for name in _KEY_JSON_ENVS:
        raw = os.environ.get(name)
        if not raw or not raw.strip():
            continue
        raw = raw.strip()
        try:
            info = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{name} is set but is not valid JSON: {e}") from e
        if not info.get("private_key"):
            raise RuntimeError(f"{name} JSON has no 'private_key' — cannot sign URLs")
        fd, tmp = tempfile.mkstemp(prefix="gcs-sa-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(raw)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp
        _creds_ready = True
        return

    # Nothing found: fall through to Application Default Credentials. Works only
    # where ADC exists, and signed URLs will fail without a private key.
    _creds_ready = True


def _get_client():
    global _client
    if _client is None:
        _ensure_creds()
        from google.cloud import storage

        _client = storage.Client()
    return _client


def upload_video(data, filename, content_type="video/mp4", bucket=None, prefix=None):
    """Upload bytes and return the DURABLE gs://bucket/object reference to persist.

    chat-api signs a fresh short-lived read url from this on every read, so we hand back
    the never-expiring gs:// uri (NOT a pre-signed https url that would rot after its TTL
    with nothing to refresh it from). Mirrors render-worker/storage.js.

    bucket: override the destination (defaults to GCS_BUCKET, the response bucket).
    prefix: override the object prefix (defaults to GCS_PREFIX; "" for none).
    """
    bucket_name = bucket or GCS_BUCKET
    if not bucket_name:
        raise RuntimeError("GCS bucket not set — pass bucket=… or set GCS_BUCKET")
    prefix = GCS_PREFIX if prefix is None else prefix.strip("/")
    path = f"{prefix}/{filename}" if prefix else filename
    blob = _get_client().bucket(bucket_name).blob(path)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{path}"


def log_config():
    """Print (once, at boot) where uploads go and validate that creds resolve, so a
    misconfig fails the worker fast — before it burns a GPU render only to fail at
    upload. Raises on hard misconfig (no bucket / unusable creds); the handler lets
    that kill the process so RunPod recycles the worker (like a failed model fetch).

    Creds are validated OFFLINE only (file/JSON present + has a private key). We do NOT
    probe the bucket over the network: roles/storage.objectAdmin grants object perms
    but not storage.buckets.get, so a live bucket.exists() check would false-negative.
    """
    if not GCS_BUCKET and not GCS_SEED_BUCKET:
        raise RuntimeError("GCS not configured: set GCS_BUCKET (response bucket)")
    try:
        _ensure_creds()
    except Exception as e:  # bad path / unparseable secret JSON / missing private_key
        raise RuntimeError(f"GCS credentials not usable: {e}") from e

    source = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "<application-default>"
    print(
        f"[gcs] response={GCS_BUCKET or '(unset)'} "
        f"seed={GCS_SEED_BUCKET or GCS_BUCKET or '(unset)'} "
        f"prefix={GCS_PREFIX or '(none)'} deliver=gs://ref (chat-api signs on read)",
        flush=True,
    )
    print(f"[gcs] creds={source} sa={_sa_email()}", flush=True)
    # Delivery returns a durable gs:// ref (chat-api signs it), so upload — not signing —
    # is what must work here; both need a resolvable SA key (objectAdmin). If we got here
    # with none, every upload would fail after a GPU render — so fail fast now.
    if _sa_email() == "?":
        raise RuntimeError(
            "no SA key resolved — set RUNPOD_SECRET_gcs_api_key (or GCS_KEY_FILE). "
            "The worker needs objectAdmin creds to upload the render to GCS"
        )


# --------------------------------------------------------------------------- CLI
# `python gcs.py …` — wire and verify the upload locally before deploying.


def _sa_email():
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        with open(path) as f:
            return json.load(f).get("client_email", "?")
    except Exception:  # noqa: BLE001 - best-effort label only
        return "?"


def _selftest(bucket=None):
    import time
    import urllib.request

    bucket_name = bucket or GCS_BUCKET
    if not bucket_name:
        raise SystemExit("no bucket: pass --bucket or set GCS_BUCKET")
    client = _get_client()  # forces _ensure_creds()
    b = client.bucket(bucket_name)
    key = f"_selftest/wiring-{int(time.time())}.txt"
    payload = b"gcs wiring ok\n"

    print(f"[selftest] project={client.project} sa={_sa_email()} bucket={bucket_name}")
    print(f"[selftest] PUT gs://{bucket_name}/{key} ...")
    blob = b.blob(key)
    blob.upload_from_string(payload, content_type="text/plain")
    print("[selftest] upload OK")

    if GCS_SIGN:
        url = blob.generate_signed_url(
            version="v4", expiration=timedelta(minutes=10), method="GET"
        )
        print(f"[selftest] signed URL (10m): {url[:90]}...")
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - our own signed URL
            body = r.read()
        if body != payload:
            raise SystemExit(f"[selftest] round-trip MISMATCH: {body!r}")
        print("[selftest] signed-URL GET OK (round-trip verified)")

    blob.delete()
    print(f"[selftest] cleaned up gs://{bucket_name}/{key}")
    print("[selftest] ✅ wiring is good")


def _upload_file(path, bucket=None, name=None, prefix=None, content_type=None):
    if not os.path.isfile(path):
        raise SystemExit(f"no such file: {path}")
    name = name or os.path.basename(path)
    if content_type is None:
        content_type = "video/mp4" if name.lower().endswith(".mp4") else "application/octet-stream"
    with open(path, "rb") as f:
        data = f.read()
    url = upload_video(data, name, content_type=content_type, bucket=bucket, prefix=prefix)
    print(url)


def _cli(argv):
    import argparse

    ap = argparse.ArgumentParser(prog="gcs.py", description="GCS upload wiring for the render worker")
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="upload+sign+GET+delete a tiny object to prove wiring")
    st.add_argument("--bucket", default=None, help="destination bucket (default $GCS_BUCKET)")

    up = sub.add_parser("upload", help="upload a local file, print the URL")
    up.add_argument("file")
    up.add_argument("--bucket", default=None, help="destination bucket (default $GCS_BUCKET)")
    up.add_argument("--name", default=None, help="object filename (default: file basename)")
    up.add_argument("--prefix", default=None, help="object prefix (default $GCS_PREFIX)")
    up.add_argument("--content-type", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "selftest":
        _selftest(bucket=args.bucket)
    elif args.cmd == "upload":
        _upload_file(
            args.file, bucket=args.bucket, name=args.name,
            prefix=args.prefix, content_type=args.content_type,
        )


if __name__ == "__main__":
    _cli(sys.argv[1:])
