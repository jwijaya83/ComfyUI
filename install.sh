#!/usr/bin/env bash
#
# Builds the ComfyUI Python environment from source: torch (cu130) -> ComfyUI
# deps -> custom-node deps -> SageAttention compiled against that torch.
#
# Only ever calls `pip`, never creates a venv, so it installs into whatever
# environment is active. Used two ways:
#
#   host:   source venv/bin/activate && ./install.sh
#   docker: builder stage runs it with /opt/venv on PATH (see Dockerfile)
#
# Assumes custom_nodes/ and SageAttention/ are present and at the revisions you
# want — nothing here clones or updates them.
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$COMFY_DIR"

# requirements.txt pins a bare `torch`, which off default PyPI resolves to a
# cu12 build. SageAttention would then compile against the wrong torch and fail
# to load at runtime. Pin the trio explicitly from the cu130 index.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.28.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"

# `docker build` runs without a GPU, so SageAttention cannot autodetect compute
# capabilities — it reads TORCH_CUDA_ARCH_LIST instead. 8.9 = Ada (RTX 4080/4090,
# L40S). Add 9.0 (H100) or 12.0 (Blackwell) if you deploy there; each extra arch
# costs build time and .so size.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
# nvcc jobs are memory-hungry (~2 GB each); cap so a many-core box doesn't OOM.
nproc_count="$(nproc)"
export MAX_JOBS="${MAX_JOBS:-$(( nproc_count < 8 ? nproc_count : 8 ))}"

echo "==> env: python=$(python3 -V 2>&1) arch_list=$TORCH_CUDA_ARCH_LIST jobs=$MAX_JOBS"

# Python 3.12 venvs no longer ship setuptools, and SageAttention builds with
# --no-build-isolation (so it uses *this* env's build tools, not a fresh one).
echo "==> build tooling"
pip3 install --upgrade pip setuptools wheel packaging ninja

echo "==> torch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
pip3 install --index-url "$TORCH_INDEX_URL" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

# torch hard-pins triton==3.7.1, so triton is intentionally NOT installed
# separately here — an unpinned `pip install triton` would break that pin.
echo "==> ComfyUI requirements"
pip3 install -r requirements.txt

echo "==> custom node requirements"
for req in custom_nodes/*/requirements.txt; do
    [ -e "$req" ] || continue
    echo "    -> $req"
    pip3 install -r "$req"
done

# Runs last so these versions win over anything the node requirements pulled:
# ComfyUI-LTXVideo asks for an unpinned kornia and 0.8.x breaks its nodes.
# NOTE: comfyui-kjnodes requests opencv-python-headless while videohelpersuite
# requests opencv-python — both provide cv2/. Reinstalling opencv-python here
# matches the known-good host env; if cv2 imports break, drop the headless one.
echo "==> pinned extras"
pip3 install einops opencv-python "kornia<0.8.0"

echo "==> SageAttention (archs: ${TORCH_CUDA_ARCH_LIST})"
pip3 install --no-build-isolation ./SageAttention

echo "==> installed:"
pip3 list 2>/dev/null | grep -iE "^(torch|torchvision|torchaudio|triton|sageattention|kornia|opencv) " || true
echo "==> done"
