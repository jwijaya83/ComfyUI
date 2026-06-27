# syntax=docker/dockerfile:1
#
# Bundles the WORKING ComfyUI install (its prebuilt venv with compiled SageAttention,
# all models, custom nodes, code) + the RunPod serverless handler into ONE image.
#
# Why ubuntu:24.04 and not an nvidia/cuda base: torch in the venv is 2.12.1+cu130 and
# ships its own CUDA runtime wheels (cublas/cudnn/cudart). RunPod injects the NVIDIA
# driver at runtime, so the base only needs Python 3.12 (the venv's base interpreter).
# An `nvidia/cuda:13.0.x-runtime-ubuntu24.04` base also works if you prefer system CUDA
# libs — override with --build-arg BASE_IMAGE=... — but ubuntu keeps the image smaller
# and avoids cuda-13-on-LD_LIBRARY_PATH conflicts with torch's bundled libs.
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

# MUST equal the host path the venv was built at: the prebuilt venv hardcodes this
# absolute path in activate/shebangs/pyvenv.cfg, so we copy the install back to it
# verbatim rather than relocating (zero relocation risk).
ARG COMFY_DIR=/media/justin-wijaya/7d3e3892-cb10-43b8-83b4-a35e3cdf9ab0/justin/Workspace/ComfyUI
ENV COMFY_DIR=${COMFY_DIR} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    # RunPod Model Caching pre-downloads HF models here (HF cache layout). fetch_models.py
    # resolves the base checkpoint + text encoder from it at boot and links them into
    # models/. If only a network volume is attached (no managed caching), the first boot
    # downloads to the same path and it persists across cold starts.
    HF_HOME=/runpod-volume/huggingface-cache

# python3.12 = the venv's base interpreter (pyvenv.cfg -> /usr/bin/python3.12);
# ffmpeg for VideoHelperSuite; git for comfyui-manager; libgl/glib for cv2-style nodes.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip \
      ffmpeg git curl ca-certificates \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Handler runs on its OWN venv (no torch), installed BEFORE the giant COPY so it
# caches independently of the model/venv layers.
COPY runpod_worker/requirements.txt /opt/runpod_worker/requirements.txt
RUN python3 -m venv /opt/handler-venv \
    && /opt/handler-venv/bin/pip install --no-cache-dir -r /opt/runpod_worker/requirements.txt

# The working ComfyUI tree (venv + persona LoRAs + custom nodes + code + the handler
# under runpod_worker/) copied to its original absolute path. The base checkpoint, text
# encoder, and the two LTX distilled LoRAs (~50 GB) are excluded via .dockerignore and
# fetched from the HF cache at boot, so this layer is ~16 GB instead of ~65 GB.
COPY . ${COMFY_DIR}

RUN cp "${COMFY_DIR}/runpod_worker/entrypoint.sh" /entrypoint.sh && chmod +x /entrypoint.sh

# ComfyUI binds 127.0.0.1:8188 internally; the handler reaches it over loopback and
# polls RunPod's queue outbound — no inbound port is exposed.
ENTRYPOINT ["/entrypoint.sh"]
