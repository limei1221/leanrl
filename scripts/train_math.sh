#!/bin/bash
# LeanRL: Train GRPO on GSM8K math
#
# Supports two GPU layouts:
#   Dedicated (default, TP=1):
#     GPU 0: Policy + Reference (DeepSpeed)
#     GPU 1: vLLM rollout engine (Ray)
#   Colocated (TP=2):
#     Both GPUs shared between vLLM (rollout) and DeepSpeed (training)
#     via sleep/wake memory time-sharing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$PROJECT_DIR/configs/math_grpo_1.5b.yaml}"

echo "=== LeanRL Math Training ==="
echo "Config: $CONFIG"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A')"

# Detect TP size from config to choose GPU layout
TP_SIZE=$(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('infra', {}).get('vllm_tensor_parallel_size', 1))
" 2>/dev/null || echo 1)

# Stop Ray
if ray status &>/dev/null; then
    echo "Stopping Ray..."
    ray stop
fi

if [ "$TP_SIZE" -ge 2 ]; then
    echo "Colocated mode (TP=$TP_SIZE): Ray uses all GPUs, shared with trainer"
    ray start --head --num-gpus="$TP_SIZE"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m leanrl.trainer --config "$CONFIG"
else
    echo "Dedicated mode (TP=1): Ray on GPU 1, trainer on GPU 0"
    CUDA_VISIBLE_DEVICES=1 ray start --head --num-gpus=1

    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m leanrl.trainer --config "$CONFIG"
fi

echo "=== Training complete ==="
