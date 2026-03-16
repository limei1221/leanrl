#!/bin/bash
# Build SWE-bench instance Docker images locally for LeanRL training.
#
# Builds sweb.eval.x86_64.{instance_id}:latest images used by the sandbox.
# Each instance image has the repo checked out at base_commit with test patch applied.
#
# Usage:
#   bash scripts/setup_swe_docker.sh                    # all 300 Lite instances
#   MAX_SAMPLES=50 bash scripts/setup_swe_docker.sh     # first 50 instances only
set -euo pipefail

echo "=== SWE-bench Docker Image Setup ==="

# Verify Docker
if ! docker info &>/dev/null; then
    echo "ERROR: Docker is not running."
    exit 1
fi

# Install swebench Python package for harness
pip install swebench 2>/dev/null || echo "swebench package already installed or pip failed"

MAX_WORKERS="${SWE_MAX_WORKERS:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"   # 0 = all instances

echo "Max workers:  $MAX_WORKERS"
echo "Max samples:  ${MAX_SAMPLES:-all}"
echo ""

python - <<EOF
import sys
import docker
from datasets import load_dataset
from swebench.harness.docker_build import build_env_images, build_instance_images

print("Loading SWE-bench Lite dataset (princeton-nlp/SWE-bench_Lite)...")
dataset = list(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"))

max_samples = int("${MAX_SAMPLES}")
if max_samples > 0:
    dataset = dataset[:max_samples]

print(f"Building instance images for {len(dataset)} instances...")
print("Step 1/2: env images (environment + dependencies)...\n")

client = docker.from_env()
successful, failed = build_env_images(
    client=client,
    dataset=dataset,
    force_rebuild=False,
    max_workers=int("${MAX_WORKERS}"),
    instance_image_tag="latest",
    env_image_tag="latest",
)
if failed:
    print(f"WARNING: {len(failed)} env image(s) failed to build:")
    for img in failed:
        print(f"  {img}")

print(f"\nStep 2/2: instance images (repo at base_commit + test patch)...\n")
successful, failed = build_instance_images(
    client=client,
    dataset=dataset,
    force_rebuild=False,
    max_workers=int("${MAX_WORKERS}"),
    env_image_tag="latest",
    tag="latest",
)

print(f"\nSuccessful: {len(successful)}")
print(f"Failed:     {len(failed)}")
if failed:
    print("Failed images:")
    for img in failed:
        print(f"  {img}")
    sys.exit(1)
EOF

echo ""
echo "=== Setup complete ==="
echo "Built instance images:"
docker images --filter "reference=sweb.eval.x86_64.*" --format "  {{.Repository}}:{{.Tag}} ({{.Size}})" | head -20
