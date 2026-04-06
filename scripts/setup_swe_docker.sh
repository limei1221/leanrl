#!/bin/bash
# Build SWE-bench instance Docker images locally for LeanRL training.
#
# Patches known swebench build failures (pylint, scikit-learn, stale branches),
# then builds sweb.eval.x86_64.{instance_id}:latest images used by the sandbox.
#
# Usage:
#   bash scripts/setup_swe_docker.sh                    # all 300 Lite instances
#   MAX_SAMPLES=50 bash scripts/setup_swe_docker.sh     # first 50 instances only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== SWE-bench Docker Image Setup ==="

if ! docker info &>/dev/null; then
    echo "ERROR: Docker is not running."
    exit 1
fi

pip install -q swebench 2>/dev/null || true

MAX_WORKERS="${SWE_MAX_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RETRY_COUNT="${RETRY_COUNT:-2}"

echo "Max workers:  $MAX_WORKERS"
echo "Max samples:  $MAX_SAMPLES (0=all)"
echo "Retry count:  $RETRY_COUNT"
echo ""

# Step 1: Patch swebench source files (separate process so the build
# step imports the already-patched modules).
echo "Patching swebench for known build failures..."
python3 "$SCRIPT_DIR/setup_swe_docker.py" --patch-only

# Step 2: Build images (fresh import picks up patched files).
echo ""
python3 "$SCRIPT_DIR/setup_swe_docker.py" \
    --max-workers "$MAX_WORKERS" \
    --max-samples "$MAX_SAMPLES" \
    --retries "$RETRY_COUNT"

echo ""
echo "=== Setup complete ==="
total=$(docker images --filter "reference=sweb.eval.x86_64.*" --format "{{.Repository}}" | wc -l)
echo "Instance images built: $total"
