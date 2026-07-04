# runpod_worker — ComfyUI + render handler, one RunPod Serverless image

Bundles this ComfyUI install (prebuilt venv w/ compiled SageAttention, models, custom
nodes) **and** a Python RunPod handler into a single container. The handler is the
serverless port of `ai-chat/services/render-worker` (render.js / comfyui.js /
workflowLoader.js / storage.js / callback.js).

## How it runs

`entrypoint.sh` boots ComfyUI in the background via `source runComfy` (its own venv),
then starts `handler.py` on a **separate** `/opt/handler-venv` (no torch). The handler
waits for ComfyUI on `127.0.0.1:8188`, then calls `runpod.serverless.start`. RunPod
pushes jobs to it; per job it renders, uploads the MP4 to GCS, and reports status to
chat-api's `/internal/render-events` webhook (the same one the current worker uses).

## Build

Build context is THIS directory (so the venv lands at its original absolute path):

```bash
cd /media/justin-wijaya/7d3e3892-cb10-43b8-83b4-a35e3cdf9ab0/justin/Workspace/ComfyUI
DOCKER_BUILDKIT=1 docker build -t <registry>/comfy-runpod:latest .
docker push <registry>/comfy-runpod:latest
```

Then create a RunPod **Serverless** endpoint from that image (GPU filter = Ada, see
caveats), and POST jobs to `https://api.runpod.ai/v2/<endpoint>/run`.

## Job input contract (`event["input"]`)

What chat-api's `runpod` queue driver sends. The asset fields (`referenceImageUrl`,
`sourceVideoUrl`, `personaReferenceUrl`) are **public download URLs** the worker GETs
directly: chat-api serves them from `PUBLIC_BASE/internal/…` with `INTERNAL_TOKEN` in
the `?t=` query string, so the worker needs no chat-api credentials or headers (it
never calls back for files, and no GCS is required on the chat-api side).

| field | required | notes |
|---|---|---|
| `jobId` | ✓ | render_jobs row id; used in status reports + output filename |
| `workflow` | ✓ | `basic_workflow` \| `latent_injection` \| `seed_workflow` |
| `positive` | ✓* | the ComfyUI prompt (falls back to `dialogue`) |
| `dialogue` | | spoken line; fallback prompt for non-seed turns |
| `durationSeconds` | | snapped to LTX 8n+1 frame count |
| `loraName` | | persona LoRA |
| `chatId` | | output filename prefix `ltx23/chat_<id>` |
| `personalitySlug` | | naming / logging |
| `useReferenceImage` | | default true |
| `referenceImageUrl` | | i2v reference (chat-api picks from the pool) |
| `sourceVideoUrl` (+`sourceSeedId`) | | latent injection (reuse) |
| `personaReferenceUrl` | | seed-gen IC-LoRA sheet |

Returns `{ jobId, status, outputUrl }`; also reports the same to the webhook. `outputUrl`
is the **durable `gs://bucket/object` ref** (never expires); chat-api stores it and mints a
fresh short-lived signed read url from it on every read.

## Required env (set on the RunPod endpoint)

| var | purpose |
|---|---|
| `CHAT_API_INTERNAL_URL` | chat-api base (publicly reachable) for status webhooks |
| `INTERNAL_TOKEN` | must match chat-api's `INTERNAL_TOKEN` |
| `RUNPOD_SECRET_gcs_api_key` *or* `GCS_SA_KEY_JSON` | **the only GCS var you must set.** SA key JSON pasted inline (V4 signing needs the SA key). RunPod exposes the `gcs_api_key` secret to the worker under `RUNPOD_SECRET_gcs_api_key` automatically. |
| `GCS_KEY_FILE` *or* `GOOGLE_APPLICATION_CREDENTIALS` | alternative to the above: path to an SA key file (used for local testing) |
| `GCS_BUCKET` | response bucket — per-turn renders (**default `video-response`**) |
| `GCS_SEED_BUCKET` | seed-video bucket (**default `video-seed`**); unset → falls back to `GCS_BUCKET` |
| `GCS_PREFIX` | object prefix (default `renders`) |
| `GCS_SIGN` | `1` (default) — signs the local `selftest` probe url; does **not** affect delivery (renders always return a durable `gs://` ref chat-api signs on read) |

The two bucket names are **baked as defaults** (this deployment's canonical buckets in
project `aichat-500601`), so the endpoint normally only needs the creds secret set.
Creds resolve in order: `GOOGLE_APPLICATION_CREDENTIALS` → `GCS_KEY_FILE` (both paths)
→ inline JSON in `GCS_SA_KEY_JSON` → `RUNPOD_SECRET_gcs_api_key` (written to a temp file
we sign from). The SA `gcs-api-user@aichat-500601.iam.gserviceaccount.com` needs
`roles/storage.objectAdmin`.

**Boot fail-fast:** the handler calls `gcs.log_config()` at startup and refuses to serve
(process exits → RunPod recycles) if no bucket is configured or, with signing on, no SA
private key resolves — so a creds misconfig is caught immediately instead of after a GPU
render fails at upload. Creds are checked offline only (no `bucket.exists()` probe, since
`objectAdmin` lacks `storage.buckets.get`).
| `WORKER_ID`, `JOB_MAX_ATTEMPTS`, `COMFY_READY_TIMEOUT` | optional |
| `HF_TOKEN` | optional — the manifest repo (`jwijaya17/aichat`) is PUBLIC, so unset is fine; set only for a gated repo |
| `MODELS_MANIFEST` | override the model manifest path (e.g. point at one on the volume) |

## Model caching (foundation models are NOT baked in)

Only the **persona** LoRAs (`models/loras/*.safetensors`) are baked into the image. The
big base checkpoint, text encoder, and the dynamic distilled LoRA come from **RunPod's
Model Caching** at boot, keeping the image ~16 GB instead of ~65 GB. (The image runs the
**per-turn** render path — `basic_workflow` / `latent_injection`. The admin seed-video
workflow runs on the host, so its 384 LoRA + IC-LoRA are intentionally not fetched here.)

**1. Enable Model Caching on the endpoint** and add the **one** PUBLIC repo
`jwijaya17/aichat` (all three files live under it, so a single cache entry covers them;
RunPod pre-downloads it to `/runpod-volume/huggingface-cache/hub` before the worker starts):

| HF repo | file (flat in repo) | → linked into (`target`) |
|---|---|---|
| `jwijaya17/aichat` | `ltx-2.3-22b-dev-fp8.safetensors` | `models/checkpoints/` |
| `jwijaya17/aichat` | `gemma_3_12B_it_fp8_scaled.safetensors` | `models/text_encoders/` |
| `jwijaya17/aichat` | `ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors` | `models/loras/` |

**2. At boot**, `fetch_models.py` reads `models_manifest.json`, resolves each file from
the HF cache (`HF_HOME=/runpod-volume/huggingface-cache`, a cache HIT when RunPod
pre-cached it), and symlinks it to the exact path the workflows reference. To add/change
a model, edit `models_manifest.json` (`repo_id` + `filename` + `target`).

Notes: `jwijaya17/aichat` is PUBLIC, so **no `HF_TOKEN` is needed**. The VAE is bundled
in the LTX checkpoint (no separate fetch). Without managed
caching, attach a network volume at `/runpod-volume` and the first boot downloads there
and persists (`fetch_models.py`'s `hf_hub_download` pulls only the listed files, not whole
repos). Increase the endpoint **Container Disk to ≥ 20 GB**.

## Wire / verify GCS upload locally (before deploying)

The handler deps (incl. `google-cloud-storage`) install into a throwaway `.venv` here
(gitignored, excluded from the image), mirroring the container's `/opt/handler-venv`:

```bash
cd runpod_worker
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
export GCS_KEY_FILE=~/Downloads/aichat-500601-XXXX.json   # your local SA key file

# Prove creds + bucket + V4 signing round-trip (uploads a tiny object, GETs it, deletes it):
./.venv/bin/python gcs.py selftest --bucket video-response
./.venv/bin/python gcs.py selftest --bucket video-seed

# Upload a real file and print its signed URL:
./.venv/bin/python gcs.py upload path/to/seed.mp4 --bucket video-seed
```

To rehearse the RunPod path (secret injected as an env var, no key file), set
`RUNPOD_SECRET_gcs_api_key="$(cat …key.json)"` and unset `GCS_KEY_FILE` /
`GOOGLE_APPLICATION_CREDENTIALS` before running `selftest`.

## Local smoke test (handler logic, against a running ComfyUI)

```bash
/opt/handler-venv/bin/python handler.py --test_input '{"input":{"jobId":"t1","workflow":"basic_workflow","positive":"a calm portrait, soft light","durationSeconds":4,"useReferenceImage":false}}'
```

## Caveats (READ before deploying)

1. **GPU architecture.** SageAttention 2.2.0 is compiled for the build host (RTX 4080 =
   Ada, sm_89). Run the endpoint on **Ada** GPUs (RTX 4090/4080, L4, L40/L40S) or the
   compiled kernels may not load. Other archs (A100 sm_80, H100 sm_90) need SageAttention
   rebuilt for them.
2. **Driver / CUDA 13.** torch is `cu130`; the RunPod host driver must support CUDA 13.
3. **Image size (~16 GB).** The base checkpoint + text encoder + the dynamic distilled
   LoRA (~45 GB) are fetched from the HF cache at boot (see Model caching); only code +
   venv + the persona LoRAs are baked in.
4. **Cold start.** First job after scale-to-zero pays ComfyUI boot + loading the models
   from the cache volume (network-volume read latency for ~50 GB). Use RunPod FlashBoot
   and/or 1 active worker for latency-sensitive traffic.
5. **chat-api side is wired (push).** Set `QUEUE_DRIVER=runpod` + `RUNPOD_ENDPOINT_ID` +
   `RUNPOD_API_KEY` on chat-api and it POSTs each job to this endpoint's `/run`. Asset
   URLs are PUBLIC chat-api URLs (`PUBLIC_BASE/internal/…?t=INTERNAL_TOKEN`), so
   `PUBLIC_BASE` must be reachable from RunPod and `INTERNAL_TOKEN` must match.
6. **Two workflow-referenced LoRAs are missing on disk** (so they aren't baked, same as
   on the host): `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors` and
   `kristin_ohwx_woman_no_audio.safetensors`. If a workflow path actually needs them,
   either drop the files into `models/loras/` before building or add the official LTX
   IC-LoRA to `models_manifest.json`.
