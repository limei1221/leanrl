#!/bin/bash
# Pull pre-built SWE-bench Docker images for LeanRL training.
#
# The official SWE-bench harness provides Docker images for each repository
# environment. This script pulls the images needed for SWE-bench Lite (300 tasks).
#
# For the full registry of optimized images, see:
#   https://github.com/swe-bench/SWE-bench
#   https://bayes.net/swebench-docker/
set -euo pipefail

echo "=== SWE-bench Docker Image Setup ==="

# Verify Docker
if ! docker info &>/dev/null; then
    echo "ERROR: Docker is not running."
    exit 1
fi

# Install swebench Python package for harness
pip install swebench 2>/dev/null || echo "swebench package already installed or pip failed"

REGISTRY="${SWE_DOCKER_REGISTRY:-ghcr.io/swe-bench}"

# Core repos in SWE-bench Lite
REPOS=(
    "astropy__astropy"
    "django__django"
    "matplotlib__matplotlib"
    "mwaskom__seaborn"
    "pallets__flask"
    "psf__requests"
    "pydata__xarray"
    "pylint-dev__pylint"
    "pytest-dev__pytest"
    "scikit-learn__scikit-learn"
    "sympy__sympy"
)

echo "Pulling base environment images for SWE-bench Lite repos..."
echo "Registry: $REGISTRY"

for repo in "${REPOS[@]}"; do
    image="${REGISTRY}/${repo}:latest"
    echo ""
    echo "--- Pulling: $image ---"
    docker pull "$image" || echo "WARNING: Failed to pull $image (may not exist yet)"
done

echo ""
echo "=== Setup complete ==="
echo "Pulled images:"
docker images --filter "reference=${REGISTRY}/*" --format "  {{.Repository}}:{{.Tag}} ({{.Size}})"

echo ""
echo "Disk usage for SWE-bench images:"
docker images --filter "reference=${REGISTRY}/*" --format '{{.Size}}' | paste -sd+ - | bc 2>/dev/null || echo "(install bc to see total)"
