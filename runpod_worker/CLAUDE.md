# CLAUDE.md — runpod_worker

One **RunPod Serverless** image = this ComfyUI install + a Python render handler. It is
the serverless port of `ai-chat/services/render-worker` (the same render → upload →
report flow, reimplemented in Python). Human-facing deploy docs live in
[README.md](README.md); this file is the orientation for working on the code.

## What it is

- A single container bundling the ComfyUI install (code, custom nodes, and a venv **built
  from source in the image** by `../install.sh`, including a compiled SageAttention)
  **plus** the handler on a separate, torch-free `/opt/handler-venv`.
- **Serverless, not a Pod.** `handler.py` calls `runpod.serverless.start` and polls
  RunPod's queue **outbound**. ComfyUI binds `127.0.0.1:8188` (loopback only) — **no
  inbound HTTP port is exposed.** Jobs arrive via `POST
  https://api.runpod.ai/v2/<endpoint>/run`, not a port on the container.
- Per job: `build_workflow` patches the workflow template → submit to ComfyUI
  (`comfy_client.py`) → upload the MP4 to GCS (`gcs.py`) → report status to chat-api's
  `/internal/render-events` webhook (`reporter.py`).

## Boot sequence (`entrypoint.sh`)

1. `fetch_models.py` resolves the foundation models from the HF cache and symlinks them
   into `models/` (**fatal** on failure — no models, no renders → let RunPod recycle).
2. ComfyUI boots in the background (`source runComfy`, its own venv).
3. `handler.py` (handler-venv) logs config and runs `gcs.log_config()` — **fatal** if GCS
   is misconfigured (no bucket, or signing on with no SA key) so a broken worker recycles
   instead of failing at upload after a GPU render — then waits for `127.0.0.1:8188` and
   calls `runpod.serverless.start`.

## Module map

- `handler.py` — `runpod.serverless.start`; per-job entry. Reads `event["input"]` (job
  descriptor: `jobId`, `workflow`, `positive`, `loraName`, reference/seed URLs, …).
- `workflow_builder.py` — `build_workflow(name, …)`: load `workflows/<name>.json`, patch
  the nodes its `<name>.meta.json` sidecar names (prompt, lora, frames/fps, reference,
  save). Deliberately does NOT touch seed nodes.
- `comfy_client.py` — submit + poll ComfyUI over `127.0.0.1:8188`. `watch_prompt()`
  **actively monitors** the render (mirrors render-worker's `comfyui.js`): the WS gives
  fast progress + fast terminal signals (`executing`→null / `execution_success` = done;
  `execution_interrupted` = manual cancel; `execution_error` = node error), interleaved
  with a `/history` + `/queue` poll every `COMFY_POLL_MS` that is the backstop for what
  the WS can't tell us — a job **cancelled/cleared while pending** (gone from both →
  non-retriable), a **ComfyUI crash/restart** (poll fails `COMFY_UNREACHABLE_TRIES` times
  → retriable), or a **wedge** (hard `COMFY_WATCH_TIMEOUT_MS` cap, then `/interrupt`). It
  raises a typed `ComfyWatchError(kind, retriable)`, so a stuck ComfyUI is reported
  `failed` in seconds instead of hanging; `handler.py` skips the retry loop for a
  non-retriable (deliberate) cancel. Env mirrors render-worker: `COMFY_POLL_MS` (5000),
  `COMFY_UNREACHABLE_TRIES` (3), `COMFY_WATCH_TIMEOUT_MS` (30m), `COMFY_POLL_TIMEOUT_MS` (8s).
- `gcs.py` — upload the MP4; returns the **durable `gs://` ref** (never expires — chat-api signs a fresh short-lived read url from it on every read, mirroring render-worker/storage.js; `GCS_SIGN` now gates only the local `selftest` round-trip). Two buckets:
  `GCS_BUCKET` (`video-response`, per-turn) and `GCS_SEED_BUCKET` (`video-seed`, seed
  clips). Creds resolve `GOOGLE_APPLICATION_CREDENTIALS`/`GCS_KEY_FILE` (paths) →
  inline JSON in `GCS_SA_KEY_JSON`/`RUNPOD_SECRET_gcs_api_key` (RunPod injects the
  `gcs_api_key` secret under the latter). Run it directly to wire/verify locally:
  `python gcs.py selftest --bucket <b>` / `python gcs.py upload <file> --bucket <b>`.
- `reporter.py` — `POST {CHAT_API_INTERNAL_URL}/internal/render-events` (x-internal-token).
- `fetch_models.py` + `models_manifest.json` — the boot-time model resolver (below).

## Model strategy (baked vs fetched)

- **Baked into the image:** code, venv, custom nodes. **Nothing under `models/`** —
  `.dockerignore` excludes the whole tree.
- **Fetched at boot** from HuggingFace via RunPod **Model Caching** (or a network volume),
  listed in `models_manifest.json`, symlinked by `fetch_models.py` (~42 GB total): the
  int8-convrot diffusion model, the Gemma text encoder + LTX text projection, the video +
  audio VAEs, the Crisp-Enhance LoRA, and the persona LoRAs.
- **The templates load a transformer-only diffusion model, not a checkpoint.** Both
  per-turn workflows moved from `CheckpointLoaderSimple(ltx-2.3-22b-dev-fp8)` to
  `UNETLoader(ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot)`. Two consequences
  the manifest encodes: (a) transformer-only bundles **no VAE and no text-encoder
  projection**, so `VAELoaderKJ` ×2 (`vae/LTX23_{video,audio}_vae_bf16`) and
  `DualCLIPLoader` clip_name2 (`text_encoders/ltx-2.3_text_projection_bf16`) are now
  separate fetches — mirrored from `Kijai/LTX2.3_comfy` into `jwijaya17/aichat` so the
  endpoint still needs only **one** Model-Caching repo; (b) the weights are **already
  distilled 1.1**, so the dynamic distilled LoRA (old node 4922) is gone from both
  templates and from the manifest.
- **Scope = the per-turn render path only** (`basic_workflow` / `latent_injection`). The
  admin **seed-video** workflow (`seed_workflow.json`, still on the dev-fp8 checkpoint,
  plus the 384 distilled LoRA and the `ic-lora-ingredients-0.9` IC-LoRA) runs **on the
  host**, so none of those are fetched here — it is not runnable in this image.
- `HF_HOME=/runpod-volume/huggingface-cache` (set in the Dockerfile). `fetch_models.py`
  calls `hf_hub_download` with **no** `force_download`, so a Model-Caching HIT is reused
  and only a miss downloads. It is idempotent (skips any `target` that already resolves)
  and fatal on failure.

### INVARIANT: manifest `target` ↔ workflow reference ↔ `.dockerignore`

A manifest entry's `target` is the path **under `models/`** where the file is symlinked,
and it **MUST equal the path the workflows load it by** — i.e. the loader node's folder
(`UNETLoader`→`diffusion_models/`, `DualCLIPLoader`→`text_encoders/`,
`VAELoaderKJ`→`vae/`, `LoraLoader*`→`loras/`) joined with its `unet_name` / `clip_name*` /
`vae_name` / `lora_name` value. ComfyUI resolves a subfolder literally: a workflow value
like `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors` is found ONLY
at `models/diffusion_models/<that name>`. Note the manifest's `filename` (the path inside
the HF repo) is independent of `target` and need not match it — `jwijaya17/aichat` is
**flat**, so e.g. `filename: "LTX23_video_vae_bf16.safetensors"` →
`target: "vae/LTX23_video_vae_bf16.safetensors"`. The `.dockerignore`
exclude pattern for a fetched file must likewise match its real on-disk path, or the file
isn't actually excluded and gets baked. When you change any one of these three, change all
three together.

## Build

Two stages: a `nvidia/cuda:*-devel` builder that runs `install.sh` to compile the venv
(SageAttention needs `nvcc`), and an `ubuntu:24.04` runtime that only receives
`/opt/venv` — torch 2.13+cu130 vendors its own CUDA libs, so no system CUDA ships.

```bash
cd /media/justin-wijaya/7d3e3892-cb10-43b8-83b4-a35e3cdf9ab0/justin/Workspace/ComfyUI
./dockerBuild.sh                    # TORCH_CUDA_ARCH_LIST=8.9;9.0 by default
# then: docker tag … <registry>/comfy-runpod:latest && docker push …
# create a RunPod Serverless endpoint from the pushed image (GPU filter = Ada).
```

**`TORCH_CUDA_ARCH_LIST` must cover every GPU the endpoint can schedule onto.** There is
no GPU during `docker build`, so SageAttention cannot autodetect archs — and it loads each
arch's kernel under a bare `except:`, so a missing arch degrades *silently* at runtime
rather than erroring. The Dockerfile's final `RUN` asserts at least one arch loaded.

The venv is built at `/opt/venv` and symlinked to `$COMFY_DIR/venv`, so `COMFY_DIR` is now
free to change (it defaults to `/opt/ComfyUI`). The old constraint — that the tree had to
land at the host's absolute path because the copied venv hardcoded it — is gone.

For local testing use `docker-compose.test.yml`, which bind-mounts `models/` instead of
baking it; see the ComfyUI-only `comfy` service.

## Gotchas / open items

- **Per-turn default LoRA.** `basic_workflow` / `latent_injection` now default node 4990 to
  `alinasverre_alxnxsvez_woman_000003500.safetensors`, which **is** in the manifest, so a
  job that omits `loraName` no longer fails at LoRA load (it just renders that persona).
  The handler still overrides the node whenever `loraName` is sent (`workflow_builder.py`).
  Keep the default pointing at a manifest entry when editing the templates.
- **chat-api side is wired (push).** chat-api's `runpod` queue driver
  (`renderQueue.js` `submitRunpod`) POSTs each render job to this endpoint's `/run`
  when `QUEUE_DRIVER=runpod` + `RUNPOD_ENDPOINT_ID` + `RUNPOD_API_KEY` are set. Assets
  arrive as download URLs the handler GETs (`_download` in `handler.py`), split by type:
  - **Seed clips (latent injection) ride as GCS SIGNED URLs** when GCS is on — chat-api
    resolves `sourceSeedId` → the seed's `gs://` (video-seed for approved, video-response
    for a candidate) → a signed read url (`renderQueue.js` `seedGcsUrl`), so the clip
    downloads straight from the bucket. If GCS is off / the seed isn't mirrored, it falls
    back to the chat-api url (`PUBLIC_BASE/internal/seed-media/…?t=INTERNAL_TOKEN`).
  - **Persona reference / img2video images still ride as chat-api URLs**
    (`PUBLIC_BASE/internal/persona-reference/…?t=INTERNAL_TOKEN`) — so `PUBLIC_BASE` must
    stay internet-reachable and `INTERNAL_TOKEN` must match chat-api's.
- **GPU arch.** SageAttention is compiled for Ada (sm_89) → run on Ada GPUs
  (RTX 4090/4080, L4, L40/L40S); other archs need it rebuilt.
