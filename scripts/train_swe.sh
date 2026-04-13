#!/bin/bash
# LeanRL: Train GRPO on SWE-bench with Docker sandboxes
# Requires: 2+ GPUs, Ray started, Docker running, SWE-bench images pulled
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$PROJECT_DIR/configs/swe_grpo_1.5b.yaml}"

echo "=== LeanRL SWE-bench Training ==="
echo "Config: $CONFIG"

# Detect container runtime: prefer docker, fall back to podman
if docker info &>/dev/null; then
    CONTAINER_RT=docker
elif podman info &>/dev/null; then
    CONTAINER_RT=podman
    if [ -z "${DOCKER_HOST:-}" ]; then
        podman system service --time=0 "unix:///tmp/podman.sock" &
        export DOCKER_HOST="unix:///tmp/podman.sock"
        sleep 1
    fi
else
    echo "ERROR: Neither Docker nor Podman is available."
    exit 1
fi
echo "Container runtime: $CONTAINER_RT"

# Check for SWE-bench images
IMAGE_COUNT=$($CONTAINER_RT images --filter "reference=sweb.eval*" --format '{{.Repository}}' 2>/dev/null | wc -l || echo 0)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "WARNING: No SWE-bench container images found."
    echo "Run 'bash scripts/setup_swe_docker.sh' first to build the required images."
    echo "Continuing anyway (sandbox creation may fail)..."
fi

# Stop Ray
if ray status &>/dev/null; then
    echo "Stopping Ray..."
    ray stop
fi

echo "Starting Ray head node on GPU 1 for vLLM..."
CUDA_VISIBLE_DEVICES=1 ray start --head --num-gpus=1

echo "Starting GRPO math training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m leanrl.trainer --config "$CONFIG"

echo "=== Training complete ==="
