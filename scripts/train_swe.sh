#!/bin/bash
# LeanRL: Train GRPO on SWE-bench with Docker sandboxes
# Requires: 2+ GPUs, Ray started, Docker running, SWE-bench images pulled
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$PROJECT_DIR/configs/swe_grpo_1.5b.yaml}"

echo "=== LeanRL SWE-bench Training ==="
echo "Config: $CONFIG"

# Verify Docker is running
if ! docker info &>/dev/null; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Check for SWE-bench images
IMAGE_COUNT=$(docker images --filter "reference=swebench/*" --format '{{.Repository}}' 2>/dev/null | wc -l || echo 0)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "WARNING: No SWE-bench Docker images found."
    echo "Run 'bash scripts/setup_swe_docker.sh' first to pull the required images."
    echo "Continuing anyway (sandbox creation may fail)..."
fi

# Start Ray head if not already running
if ! ray status &>/dev/null; then
    echo "Starting Ray head node..."
    ray start --head --num-gpus "$(nvidia-smi -L 2>/dev/null | wc -l || echo 2)"
    STARTED_RAY=1
else
    echo "Ray already running."
    STARTED_RAY=0
fi

# Run training
python -m leanrl.trainer --config "$CONFIG"

# Optionally stop Ray if we started it
if [ "$STARTED_RAY" -eq 1 ]; then
    echo "Stopping Ray..."
    ray stop
fi

echo "=== Training complete ==="
