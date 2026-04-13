#!/bin/bash
# Build SWE-bench instance Docker images locally for LeanRL training.
#
# Patches known swebench build failures (pylint, scikit-learn, stale branches),
# then builds sweb.eval.x86_64.{instance_id}:latest images used by the sandbox.
#
# Usage:
#   bash scripts/setup_swe_docker.sh                    # all test instances
#   bash scripts/setup_swe_docker.sh dev                # all dev instances
#   MAX_SAMPLES=50 bash scripts/setup_swe_docker.sh     # first 50 test instances
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect container runtime: prefer docker, fall back to podman
if docker info &>/dev/null; then
    CONTAINER_RT=docker
elif podman info &>/dev/null; then
    CONTAINER_RT=podman
    # Expose Podman via Docker-compatible socket so the Python docker SDK works
    if [ -z "${DOCKER_HOST:-}" ]; then
        podman system service --time=0 "unix:///tmp/podman.sock" &
        export DOCKER_HOST="unix:///tmp/podman.sock"
        sleep 1
    fi
else
    echo "ERROR: Neither Docker nor Podman is available."
    exit 1
fi

echo "=== SWE-bench Docker Image Setup (runtime: $CONTAINER_RT) ==="

pip install -q swebench 2>/dev/null || true

SPLIT="${1:-test}"
MAX_WORKERS="${SWE_MAX_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RETRY_COUNT="${RETRY_COUNT:-2}"

echo "Split:        $SPLIT"
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
    --split "$SPLIT" \
    --max-workers "$MAX_WORKERS" \
    --max-samples "$MAX_SAMPLES" \
    --retries "$RETRY_COUNT"

echo ""
echo "=== Setup complete ==="
total=$($CONTAINER_RT images --filter "reference=sweb.eval.x86_64.*" --format "{{.Repository}}" | wc -l)
echo "Instance images built: $total"
