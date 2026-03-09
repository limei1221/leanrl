#!/bin/bash
# LeanRL: Train GRPO on GSM8K math
# Recommended: 2 GPUs
#   GPU 0: Policy + Reference (DeepSpeed)
#   GPU 1: vLLM rollout engine (Ray)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$PROJECT_DIR/configs/math_grpo_1.5b.yaml}"

echo "=== LeanRL Math Training ==="
echo "Config: $CONFIG"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A')"

# Start Ray head on GPU 1 (vLLM engine) if not already running
if ! ray status &>/dev/null; then
    echo "Starting Ray head node on GPU 1 for vLLM..."
    CUDA_VISIBLE_DEVICES=1 ray start --head --num-gpus=1
    STARTED_RAY=1
else
    echo "Ray already running."
    STARTED_RAY=0
fi

echo "Starting GRPO math training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m leanrl.trainer --config "$CONFIG"

# Optionally stop Ray if we started it
if [ "$STARTED_RAY" -eq 1 ]; then
    echo "Stopping Ray..."
    ray stop
fi

echo "=== Training complete ==="
