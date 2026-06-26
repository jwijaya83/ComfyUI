#!/usr/bin/env python3
"""Resolve the big foundation models from the HuggingFace cache and link them into
ComfyUI's models/ dirs. Run at container startup BEFORE ComfyUI boots.

Why: the persona LoRAs (models/loras/) are baked into the image, but the multi-GB
base checkpoint + text encoder are NOT — they come from RunPod's Model Caching
feature, which pre-downloads HF models to /runpod-volume/huggingface-cache/hub before
the worker starts (HF cache layout). We point HF_HOME there (see Dockerfile), resolve
each file via hf_hub_download (a cache HIT when RunPod pre-cached it; a one-time
download otherwise — e.g. when only a network volume is attached), then symlink it to
the exact path the workflows reference.

Idempotent: skips a target that already resolves. Fatal on failure (no models = no
renders) so the entrypoint can abort and let RunPod recycle the worker.
"""
import json
import os
import sys

COMFY_DIR = os.environ.get("COMFY_DIR")
MANIFEST = os.environ.get(
    "MODELS_MANIFEST",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_manifest.json"),
)


def main():
    if not COMFY_DIR:
        print("[fetch-models] FATAL: COMFY_DIR not set", file=sys.stderr)
        sys.exit(1)
    models_dir = os.path.join(COMFY_DIR, "models")

    with open(MANIFEST) as f:
        entries = json.load(f)

    # Gated repos (Lightricks / Gemma) may need a token when NOT pre-cached by RunPod.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    from huggingface_hub import hf_hub_download

    for e in entries:
        target = os.path.join(models_dir, e["target"])
        # A valid existing file/symlink (target resolves) → nothing to do.
        if os.path.exists(target):
            print(f"[fetch-models] present: {e['target']}", flush=True)
            continue

        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"[fetch-models] resolving {e['repo_id']} :: {e['filename']}", flush=True)
        src = hf_hub_download(
            repo_id=e["repo_id"],
            filename=e["filename"],
            revision=e.get("revision"),
            token=token,
        )
        # Clear a stale/broken symlink before re-linking.
        if os.path.islink(target) or os.path.exists(target):
            os.remove(target)
        os.symlink(src, target)
        print(f"[fetch-models] linked {e['target']} -> {src}", flush=True)

    print("[fetch-models] all foundation models resolved", flush=True)


if __name__ == "__main__":
    main()
