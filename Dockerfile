# syntax=docker/dockerfile:1
#
# Two-stage build for the ComfyUI RunPod serverless worker.
#
#   builder  nvidia/cuda *devel* — the only stage that needs nvcc, which
#            SageAttention compiles its kernels with. Runs install.sh to build
#            /opt/venv from source, then is discarded (~9 GB never shipped).
#   runtime  plain ubuntu:24.04. torch 2.13+cu130 vendors its own CUDA runtime
#            (nvidia-cublas / cudnn / cudart wheels sit inside the venv) and the
#            host injects the driver, so no system CUDA is needed here. Saves
#            ~2 GB over a cuda-runtime base.
#
# Unlike the previous image this does NOT copy the host venv. The environment is
# built inside the image at a fixed path, so nothing depends on the host's
# absolute ComfyUI path — the host venv/bin/activate had already drifted from it
# (it hardcodes a stale `...a35e3cdf9ab02/` path that no longer exists).
#
# Both stages must share an Ubuntu release: the venv's compiled extensions are
# cp312 ABI, matching the python3.12 that 24.04 ships.

ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG RUNTIME_IMAGE=ubuntu:24.04

# ----------------------------------------------------------------- builder ---
FROM ${CUDA_DEVEL_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Everything downstream resolves python3/pip3 to the venv via PATH; install.sh
# never creates or activates one itself.
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    VIRTUAL_ENV=/opt/venv

# Compute capabilities to compile SageAttention for — no GPU is visible during
# `docker build`, so this cannot be autodetected. See install.sh for the mapping.
ARG TORCH_CUDA_ARCH_LIST=8.9
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Copy only what install.sh reads, so editing workflows/handler code does not
# invalidate the (very expensive) SageAttention compile layer.
WORKDIR /src
COPY requirements.txt install.sh ./
COPY custom_nodes/ ./custom_nodes/
COPY SageAttention/ ./SageAttention/

# The pip cache lives in a BuildKit cache mount, not in the layer, so repeat
# builds skip re-downloading ~5 GB of wheels without inflating the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    COMFY_DIR=/src bash install.sh

# ----------------------------------------------------------------- runtime ---
FROM ${RUNTIME_IMAGE} AS runtime

ARG COMFY_DIR=/opt/ComfyUI
ENV COMFY_DIR=${COMFY_DIR} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    # RunPod Model Caching pre-downloads HF models here. fetch_models.py resolves
    # the base checkpoint + text encoder from it at boot and links them into
    # models/. Harmless when running locally with models/ bind-mounted instead.
    HF_HOME=/runpod-volume/huggingface-cache

# python3.12 = the interpreter /opt/venv was built against. python3.12-venv is
# needed for ensurepip when building the handler venv below (the copied /opt/venv
# does not provide it). ffmpeg for video encode/decode; libgl/glib for opencv.
# git is a RUNTIME dep, not just a build one: comfyui-manager imports GitPython,
# which raises "Bad git executable" at import and takes the whole node down.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv \
      ffmpeg curl ca-certificates git \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# The handler runs on its OWN venv (no torch). Installed before the app copy so
# it caches independently of code changes.
COPY runpod_worker/requirements.txt /opt/runpod_worker/requirements.txt
RUN python3.12 -m venv /opt/handler-venv \
    && /opt/handler-venv/bin/pip install --no-cache-dir -r /opt/runpod_worker/requirements.txt

# App code, custom nodes, workflows and the handler. models/ and venv/ are
# excluded via .dockerignore — models are bind-mounted locally, and supplied by
# the network volume / HF cache in production.
WORKDIR ${COMFY_DIR}
COPY . ${COMFY_DIR}

# entrypoint.sh does `source venv/bin/activate` exactly as on the host. The venv
# now lives at /opt/venv (outside the bind-mountable tree), so point the expected
# path at it — activate then exports the correct VIRTUAL_ENV=/opt/venv.
RUN ln -s /opt/venv "${COMFY_DIR}/venv" \
    && cp "${COMFY_DIR}/runpod_worker/entrypoint.sh" /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Sanity-check the expensive parts at build time rather than on a cold RunPod
# worker. sageattention/core.py loads each arch's compiled extension under a
# bare `except:`, so a plain `import sageattention` still succeeds when the
# kernels failed to load — assert at least one arch is live, otherwise the
# worker would silently fall back and --use-sage-attention would do nothing.
RUN python3 -c "\
import torch, sageattention; from sageattention import core; \
archs = {n[:4]: getattr(core, n) for n in dir(core) if n.endswith('_ENABLED')}; \
print('torch', torch.__version__, '| sageattention archs', archs); \
assert any(archs.values()), 'no SageAttention CUDA kernels loaded: ABI or arch mismatch'"

# ComfyUI binds 8188; the handler reaches it over loopback and polls RunPod's
# queue outbound, so no inbound port is exposed in production. Local testing
# publishes it explicitly (see docker-compose.test.yml).
ENTRYPOINT ["/entrypoint.sh"]
