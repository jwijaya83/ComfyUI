#!/usr/bin/env bash
# Production image build (pushable, multi-arch SageAttention). For local viability
# testing use ai-chat's docker-compose.yml `comfy-worker` service instead — it builds
# from this checkout and bind-mounts models/ so you are not rebuilding around 35 GB:
#   cd ../ai-chat && docker compose up -d --build comfy-worker
set -euo pipefail

# Compute capabilities to compile SageAttention kernels for. Must cover every GPU
# the RunPod endpoint can schedule onto, since this cannot be detected at build
# time — a missing arch means SageAttention silently disables itself at runtime.
#   8.9   Ada        RTX 4080/4090, L40S
#   9.0   Hopper     H100, H200
#   12.0  Blackwell  B200, RTX 50xx
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;9.0}"
TAG="${TAG:-comfy-runpod:latest}"

DOCKER_BUILDKIT=1 docker build \
  --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" \
  -t "$TAG" .
