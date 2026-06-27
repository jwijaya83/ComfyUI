# CLAUDE.md — runpod_worker

One **RunPod Serverless** image = this ComfyUI install + a Python render handler. It is
the serverless port of `ai-chat/services/render-worker` (the same render → upload →
report flow, reimplemented in Python). Human-facing deploy docs live in
[README.md](README.md); this file is the orientation for working on the code.

## What it is

- A single container bundling the working ComfyUI install (prebuilt venv with compiled
  SageAttention, custom nodes, **persona** LoRAs) **plus** the handler on a separate,
  torch-free `/opt/handler-venv`.
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
3. `handler.py` (handler-venv) waits for `127.0.0.1:8188`, then `runpod.serverless.start`.

## Module map

- `handler.py` — `runpod.serverless.start`; per-job entry. Reads `event["input"]` (job
  descriptor: `jobId`, `workflow`, `positive`, `loraName`, reference/seed URLs, …).
- `workflow_builder.py` — `build_workflow(name, …)`: load `workflows/<name>.json`, patch
  the nodes its `<name>.meta.json` sidecar names (prompt, lora, frames/fps, reference,
  save). Deliberately does NOT touch seed nodes.
- `comfy_client.py` — submit + poll ComfyUI over `127.0.0.1:8188`.
- `gcs.py` — upload the MP4; returns a signed (or public) URL.
- `reporter.py` — `POST {CHAT_API_INTERNAL_URL}/internal/render-events` (x-internal-token).
- `fetch_models.py` + `models_manifest.json` — the boot-time model resolver (below).

## Model strategy (baked vs fetched)

- **Baked into the image:** code, venv, custom nodes, and the **persona** LoRAs
  (`models/loras/*.safetensors`).
- **Fetched at boot** from HuggingFace via RunPod **Model Caching** (or a network volume),
  listed in `models_manifest.json`, symlinked by `fetch_models.py`: the LTX 2.3 base
  checkpoint, the Gemma text encoder, and the **dynamic** distilled LoRA.
- **Scope = the per-turn render path only** (`basic_workflow` / `latent_injection`). The
  admin **seed-video** workflow (`seed_workflow.json`, which adds the 384 distilled LoRA
  and the `ic-lora-ingredients-0.9` IC-LoRA) runs **on the host**, so those two are
  intentionally NOT fetched or baked here.
- `HF_HOME=/runpod-volume/huggingface-cache` (set in the Dockerfile). `fetch_models.py`
  calls `hf_hub_download` with **no** `force_download`, so a Model-Caching HIT is reused
  and only a miss downloads. It is idempotent (skips any `target` that already resolves)
  and fatal on failure.

### INVARIANT: manifest `target` ↔ workflow reference ↔ `.dockerignore`

A manifest entry's `target` is the path **under `models/`** where the file is symlinked,
and it **MUST equal the path the workflows load it by** (`ckpt_name` / `text_encoder` /
`lora_name`). ComfyUI resolves a subfolder literally: a workflow value like
`ltx23/ltx-2.3-22b-dev-fp8.safetensors` is found ONLY at
`models/checkpoints/ltx23/ltx-2.3-22b-dev-fp8.safetensors` — so the target must carry the
same `ltx23/` segment (or the workflows must be flattened to match). The `.dockerignore`
exclude pattern for a fetched file must likewise match its real on-disk path, or the file
isn't actually excluded and gets baked. When you change any one of these three, change all
three together.

## Build

```bash
cd /media/justin-wijaya/7d3e3892-cb10-43b8-83b4-a35e3cdf9ab0/justin/Workspace/ComfyUI
DOCKER_BUILDKIT=1 docker build -t comfy-runpod:latest .   # context = whole tree; .dockerignore trims it
# then: docker tag … <registry>/comfy-runpod:latest && docker push …
# create a RunPod Serverless endpoint from the pushed image (GPU filter = Ada).
```
**Do NOT override `COMFY_DIR`** — the prebuilt venv hardcodes its absolute path, so the
tree must land back at exactly that path. Run the build from the ComfyUI root, not here.

## Gotchas / open items

- **Per-turn default LoRA placeholder.** `basic_workflow` / `latent_injection` carry a
  default `lora_name: "kristin_ohwx_woman_no_audio.safetensors"` that is NOT on disk → not
  baked. Harmless **only** while every job sends `loraName` (the handler overrides the lora
  node, `workflow_builder.py`). A job that omits it would fail at LoRA load — either always
  send `loraName` or change the templates' default to a baked LoRA.
- **chat-api side not wired yet.** No `runpod` queue driver / signed-URL job contract in
  chat-api, so nothing sends this endpoint jobs until that pass (see README contract).
- **GPU arch.** SageAttention is compiled for Ada (sm_89) → run on Ada GPUs
  (RTX 4090/4080, L4, L40/L40S); other archs need it rebuilt.
