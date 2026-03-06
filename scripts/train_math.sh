#!/bin/bash
# LeanRL: Train GRPO on GSM8K math
# Requires: 2+ GPUs, Ray started
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-$PROJECT_DIR/configs/math_grpo_1.5b.yaml}"

echo "=== LeanRL Math Training ==="
echo "Config: $CONFIG"
echo "GPUs available: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'N/A')"

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
