# CLAUDE.md — runpod_worker (THE render worker)

One image = this ComfyUI install + a Python render handler. It **is** ai-chat's render
pool: it replaced `ai-chat/services/render-worker` (now deprecated), of which it is a
faithful Python port — same job descriptor, same status webhook, same queue semantics.
Human-facing deploy docs live in [README.md](README.md); this file is the orientation for
working on the code.

*(The directory name is historical — RunPod is now just one of three intakes. Renaming it
would break the Dockerfile/entrypoint paths and any deployed endpoint, so it stays.)*

## What it is

- A single container bundling the ComfyUI install (code, custom nodes, and a venv **built
  from source in the image** by `../install.sh`, including a compiled SageAttention)
  **plus** the handler on a separate, torch-free `/opt/handler-venv`.
- **THE POINT: ComfyUI and the worker ship together, and the intake is OUTBOUND.** The
  old split (a Node worker pointed at a `COMFYUI_HTTP` someone else deployed) meant new
  render capacity was a two-part manual setup. Here, capacity is "run this container
  somewhere with a GPU, with `REDIS_URL` + `CHAT_API_INTERNAL_URL` + `INTERNAL_TOKEN`
  pointed at production" — it joins the shared consumer group and starts claiming jobs.
  Nothing to register; **no inbound port required**; stop it and its work goes back to
  the queue. Local box, RunPod, any GPU cloud — same image, same behaviour.
- Per job: `build_workflow` patches the workflow template → submit to ComfyUI
  (`comfy_client.py`) → deliver the MP4 (`storage.py`: GCS and/or a shared media dir) →
  report status to chat-api's `/internal/render-events` webhook (`reporter.py`).

## The three intakes (`QUEUE_DRIVER`, see `config.py`)

| | what it does | used by |
|---|---|---|
| `redis` | Drain ai-chat's **main render queue** — the per-lane Redis Streams (`render:high`, `render:low`) as a member of the shared `renderers` consumer group. Priority high→low, at-least-once with `XACK`, `XAUTOCLAIM` crash recovery, dead-letter. | the in-compose `comfy-worker`, and any extra GPU box you point at the same Redis |
| `runpod` | `runpod.serverless.start` polls RunPod's own queue outbound; jobs arrive as `{"input": <job>}`. | a RunPod Serverless endpoint |
| `http` | chat-api POSTs one job to `:RENDER_PORT/render` (token-guarded). | the no-broker local profile |

All three call the **same `process_job()`** and report to the same webhook, so behaviour
is identical whichever way work arrives. `MOCK_COMFY=1` swaps the GPU render for an
ffmpeg one (`mock.py`) and the entrypoint then skips booting ComfyUI entirely — that's
how the queue path is smoke-tested on a box with no GPU.

## Boot sequence (`entrypoint.sh`)

1. Unless `MOCK_COMFY=1`: `fetch_models.py` resolves the foundation models from the HF
   cache and symlinks them into `models/` (**fatal** on failure — no models, no renders).
   `FETCH_MODELS=0` skips it when `models/` is bind-mounted from a tree that has them.
2. ComfyUI boots in the background (`source runComfy`, its own venv).
3. `handler.py` (handler-venv) logs config, validates that a **delivery path exists**
   (`_check_delivery`: GCS credentials hard-validated when a bucket is set, else a shared
   `MEDIA_DIR`; neither → exit) — better to recycle now than to burn a GPU render and
   discover it at upload time — then waits for `127.0.0.1:8188` and starts the intake
   its `QUEUE_DRIVER` selects.

## Module map

- `handler.py` — the shared render path: `_resolve_assets` → `build_workflow` → submit →
  `storage.save_video` → report, with `JOB_MAX_ATTEMPTS` retries. **Never raises**: on
  exhaustion it reports `failed` and returns, so an intake can ACK unconditionally.
  `main()` dispatches on `QUEUE_DRIVER`; `handler(event)` is the RunPod entry point.
- `config.py` — every knob, with **the same env names render-worker used**, so this image
  is a drop-in replacement for that service's docker-compose environment block.
- `queue_consumer.py` — the Redis Streams consumer (port of `consumer.js`). One group
  across all instances; `RENDER_CONCURRENCY` jobs at a time via a thread pool; reclaims a
  dead worker's jobs after `RENDER_VISIBILITY_MS`; dead-letters a job redelivered past
  `RENDER_MAX_DELIVERIES` to `render:dead` **and reports it `failed`** so the UI doesn't
  hang on it. Consumer name is `${WORKER_ID}-${hostname}-${pid}` — must be unique per
  instance (every container is PID 1, so pid alone collides).
- `http_server.py` — `GET /health` (the compose healthcheck, always up on every intake)
  and the `QUEUE_DRIVER=http` `POST /render` intake.
- `storage.py` — `save_video()` (port of `storage.js`): writes to `MEDIA_DIR` when set
  (a volume shared with chat-api, which re-serves it at `/media`) and/or uploads to the
  GCS response bucket, returning the **durable `gs://` ref** chat-api signs on read. A
  remote worker shares no volume, so GCS is its only real delivery path.
- `mock.py` — `MOCK_COMFY=1`: a real playable MP4 from ffmpeg, no GPU (port of `mock.js`).
- `gpu_lease.py` — the cross-service Redis mutex + Ollama eviction (port of `gpuLease.js`
  + `freeOllama.js`), for a single-GPU box where ComfyUI shares the card with ai-chat's
  llm-worker. Off by default; on RunPod the GPU is ours alone. **Keep in sync with the
  two JS copies.**
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
  clips) — each also accepts ai-chat's spelling (`GCS_BUCKET_RESPONSE` /
  `GCS_BUCKET_SEED`) so one env vocabulary drives every service. Creds resolve
  `GOOGLE_APPLICATION_CREDENTIALS`/`GCS_KEY_FILE` (paths) →
  inline JSON in `GCS_SA_KEY_JSON`/`RUNPOD_SECRET_gcs_api_key` (RunPod injects the
  `gcs_api_key` secret under the latter). Run it directly to wire/verify locally:
  `python gcs.py selftest --bucket <b>` / `python gcs.py upload <file> --bucket <b>`.
  Set **`GCS_PREFIX=""`** when replacing render-worker: that service wrote at the bucket
  root, and the default `renders/` prefix would split one bucket across two layouts.
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

**Normally you don't build it by hand:** ai-chat's `docker-compose.yml` has a
`comfy-worker` service whose build context IS this checkout (`COMFY_DIR`, default
`../ComfyUI`), so `docker compose up --build` from the ai-chat repo builds and runs it
wired to Postgres/Redis/chat-api. `dockerBuild.sh` is for producing a **pushable** image
(multi-arch SageAttention) that a remote GPU box or a RunPod endpoint pulls.

## Running it against ai-chat's queue

```bash
# In the ai-chat repo (the normal path — builds + wires everything):
docker compose up -d --build comfy-worker
docker compose up --scale comfy-worker=2       # more GPUs = more workers

# Anywhere else with a GPU (a second box, a RunPod Pod), against production:
docker run --gpus all \
  -e QUEUE_DRIVER=redis -e REDIS_URL=redis://<prod-redis>:6379 \
  -e CHAT_API_INTERNAL_URL=https://api.example.com -e INTERNAL_TOKEN=<same as chat-api> \
  -e GCS_BUCKET_RESPONSE=video-response -e GCS_KEY_FILE=/secrets/gcs-key.json \
  -e WORKER_ID=gpu-box-2 \
  -v /path/to/models:/opt/ComfyUI/models -v /path/to/key.json:/secrets/gcs-key.json:ro \
  <registry>/comfy-runpod:latest
```

The second form needs **no inbound access and no chat-api change** — it dials out to
Redis, claims jobs from the shared group, and reports over the webhook. A remote worker
shares no volume with chat-api, so GCS (not `MEDIA_DIR`) is what makes its output
reachable; the boot check refuses to start if neither is configured.

Smoke-test the whole path with no GPU by adding `-e MOCK_COMFY=1` (ffmpeg renders a real
MP4 and ComfyUI never boots).

## Gotchas / open items

- **A blocking `XREADGROUP` needs `socket_timeout` > `BLOCK`.** redis-py applies its own
  default (5s in 8.x), which exactly races the default `RENDER_BLOCK_MS=5000`: every idle
  cycle raised `TimeoutError`, hit the loop's error branch and slept a second.
  `queue_consumer.py` sets the timeout explicitly (BLOCK + 15s) rather than inheriting
  whatever the installed version defaults to. Don't remove it.
- **`RENDER_VISIBILITY_MS` must exceed the worst-case `process_job`** (≈ `JOB_MAX_ATTEMPTS`
  × a full render), or a still-rendering job is reclaimed by another worker and rendered
  twice.
- **Delivery must be configured or the worker exits at boot** (`_check_delivery`): a GCS
  response bucket + credentials, or a `MEDIA_DIR` shared with chat-api. This is
  deliberate — the alternative is discovering it after a GPU render.

- **Per-turn default LoRA.** `basic_workflow` / `latent_injection` now default node 4990 to
  `alinasverre_alxnxsvez_woman_000003500.safetensors`, which **is** in the manifest, so a
  job that omits `loraName` no longer fails at LoRA load (it just renders that persona).
  The handler still overrides the node whenever `loraName` is sent (`workflow_builder.py`).
  Keep the default pointing at a manifest entry when editing the templates.
- **Assets arrive two ways, and the worker handles both.** With the `runpod` driver
  chat-api pre-resolves every asset to a URL; with the `redis`/`http` drivers the payload
  may carry only a `sourceSeedId` / `personaReference` / `personalitySlug`, and the worker
  fetches the bytes itself from chat-api's token-guarded `/internal/seed-media` +
  `/internal/persona-reference` (`_chat_api_asset` in `handler.py`). Precedence is
  render.js's: **seed reuse > persona reference > img2video reference**.
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
