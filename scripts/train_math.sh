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
