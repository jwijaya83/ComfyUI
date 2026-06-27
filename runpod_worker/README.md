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

What chat-api's new `runpod` queue driver must send. Assets are **signed GCS URLs** —
the worker never calls back into chat-api for files.

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

Returns `{ jobId, status, outputUrl }`; also reports the same to the webhook.

## Required env (set on the RunPod endpoint)

| var | purpose |
|---|---|
| `CHAT_API_INTERNAL_URL` | chat-api base (publicly reachable) for status webhooks |
| `INTERNAL_TOKEN` | must match chat-api's `INTERNAL_TOKEN` |
| `GCS_BUCKET` | output bucket |
| `GCS_SA_KEY_JSON` *or* `GOOGLE_APPLICATION_CREDENTIALS` | SA creds (V4 signing needs the SA key) |
| `GCS_PREFIX` | object prefix (default `renders`) |
| `GCS_SIGN` | `1` signed URL (default) \| `0` public-bucket URL |
| `GCS_SIGNED_URL_TTL` | seconds (default 7d) |
| `WORKER_ID`, `JOB_MAX_ATTEMPTS`, `COMFY_READY_TIMEOUT` | optional |
| `HF_TOKEN` | only if a manifest repo is gated and NOT pre-cached by RunPod |
| `MODELS_MANIFEST` | override the model manifest path (e.g. point at one on the volume) |

## Model caching (foundation models are NOT baked in)

Only the **persona** LoRAs (`models/loras/*.safetensors`) are baked into the image. The
big base checkpoint, text encoder, and the dynamic distilled LoRA come from **RunPod's
Model Caching** at boot, keeping the image ~16 GB instead of ~65 GB. (The image runs the
**per-turn** render path — `basic_workflow` / `latent_injection`. The admin seed-video
workflow runs on the host, so its 384 LoRA + IC-LoRA are intentionally not fetched here.)

**1. Enable Model Caching on the endpoint** and add these **3** HF models (RunPod
pre-downloads them to `/runpod-volume/huggingface-cache/hub` before the worker starts):

| HF model path | file used | → linked into |
|---|---|---|
| `Lightricks/LTX-2.3-fp8` | `ltx-2.3-22b-dev-fp8.safetensors` | `models/checkpoints/ltx23/` |
| `Comfy-Org/ltx-2` | `split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors` | `models/text_encoders/ltx23/` |
| `Kijai/LTX2.3_comfy` | `loras/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors` | `models/loras/ltx23/` |

**2. At boot**, `fetch_models.py` reads `models_manifest.json`, resolves each file from
the HF cache (`HF_HOME=/runpod-volume/huggingface-cache`, a cache HIT when RunPod
pre-cached it), and symlinks it to the exact path the workflows reference. To add/change
a model, edit `models_manifest.json` (`repo_id` + `filename` + `target`).

Notes: caching `Lightricks/LTX-2.3-fp8` pulls the whole repo (~59 GB) to the volume, but
only the one file is linked. The VAE is bundled in the LTX checkpoint (no separate fetch).
If a repo is gated (Lightricks/Gemma) and not pre-cached, set `HF_TOKEN`. Without managed
caching, attach a network volume at `/runpod-volume` and the first boot downloads there
and persists (`fetch_models.py`'s `hf_hub_download` pulls only the listed files, not whole
repos). Increase the endpoint **Container Disk to ≥ 20 GB**.

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
5. **chat-api side is not wired yet.** This image expects the signed-URL contract above;
   the `runpod` queue driver + signed-URL asset resolution in chat-api is the next pass.
6. **Two workflow-referenced LoRAs are missing on disk** (so they aren't baked, same as
   on the host): `ltx23/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors` and
   `kristin_ohwx_woman_no_audio.safetensors`. If a workflow path actually needs them,
   either drop the files into `models/loras/` before building or add the official LTX
   IC-LoRA to `models_manifest.json`.
