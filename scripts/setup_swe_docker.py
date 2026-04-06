"""Build SWE-bench Docker images with automatic patching of known failures.

Called by scripts/setup_swe_docker.sh.  Can also be run directly:
    python scripts/setup_swe_docker.py --max-samples 50
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


# ── Patch swebench constants before any build ────────────────────────────────

def patch_swebench_constants() -> None:
    """Fix known swebench 4.x build failures by patching source files.

    1. scikit-learn: remove --no-use-pep517 (dropped in pip 23.1+).
    2. pylint: add --no-build-isolation to avoid the build_editable PEP 660
       hook that pylint's setuptools backend doesn't support.
    3. sympy: remove stale branch hints from REPO_BASE_COMMIT_BRANCH so
       clones don't fail when upstream deletes old branches (e.g. "1.7").

    All patches are idempotent — safe to run multiple times.
    """
    import swebench.harness.constants.python as constants
    import swebench.harness.constants as swe_constants

    # ── Fix 1 & 2: patch constants/python.py ──────────────────────────────
    py_path = os.path.abspath(constants.__file__)
    with open(py_path) as f:
        src = f.read()

    original = src

    # Fix 1: sklearn — remove the obsolete --no-use-pep517 flag.
    src = src.replace("--no-use-pep517 ", "")

    # Fix 2: pylint — add --no-build-isolation to `pip install -e .`
    # only inside the SPECS_PYLINT block.
    pylint_start = src.find("SPECS_PYLINT")
    if pylint_start != -1:
        m = re.search(r"\nSPECS_(?!PYLINT)", src[pylint_start + 1:])
        pylint_end = pylint_start + 1 + m.start() if m else len(src)
        block = src[pylint_start:pylint_end]
        if "--no-build-isolation" not in block:
            block = block.replace(
                '"install": "python -m pip install -e .",',
                '"install": "python -m pip install --no-build-isolation -e .",',
            )
            src = src[:pylint_start] + block + src[pylint_end:]

    if src != original:
        with open(py_path, "w") as f:
            f.write(src)
        print(f"  Patched: {py_path}")
    else:
        print("  constants/python.py: already patched")

    # ── Fix 3: fix --single-branch without --branch in test_spec/python.py ─
    # When REPO_BASE_COMMIT_BRANCH has no entry for a commit, the generated
    # git clone command becomes `git clone ... --single-branch URL` which
    # only fetches the default branch's history.  If the target commit is on
    # a different branch, `git reset --hard <commit>` fails.
    # Fix: only include --single-branch when --branch is also present.
    import swebench.harness.test_spec.python as ts_python
    ts_path = os.path.abspath(ts_python.__file__)
    with open(ts_path) as f:
        ts_src = f.read()

    old_clone = '{branch} --single-branch https://github.com/{repo}'
    new_clone = '{branch} {"--single-branch" if branch else ""} https://github.com/{repo}'
    if old_clone in ts_src:
        ts_src = ts_src.replace(old_clone, new_clone)
        with open(ts_path, "w") as f:
            f.write(ts_src)
        print(f"  Patched --single-branch: {ts_path}")
    else:
        print("  test_spec/python.py: already patched")

    # ── Fix 4: remove stale branch hints ──────────────────────────────────
    # REPO_BASE_COMMIT_BRANCH maps (repo, commit) → branch for faster
    # git clone --branch X --single-branch.  When upstream deletes a
    # branch, the clone fails.  Remove stale entries so swebench falls
    # back to a full clone + git reset --hard.
    init_path = os.path.abspath(swe_constants.__file__)
    with open(init_path) as f:
        init_src = f.read()

    if "REPO_BASE_COMMIT_BRANCH" not in init_src:
        print("  No REPO_BASE_COMMIT_BRANCH found (skipping)")
        return

    from swebench.harness.constants import REPO_BASE_COMMIT_BRANCH

    stale = []
    for repo, commits in REPO_BASE_COMMIT_BRANCH.items():
        for commit, branch in list(commits.items()):
            try:
                ret = subprocess.run(
                    ["git", "ls-remote", "--heads",
                     f"https://github.com/{repo}", branch],
                    capture_output=True, timeout=15,
                )
                if ret.returncode != 0 or not ret.stdout.strip():
                    stale.append((repo, commit, branch))
            except subprocess.TimeoutExpired:
                pass  # keep the entry if we can't verify

    if not stale:
        print("  Branch hints: all valid")
        return

    init_original = init_src
    for repo, commit, branch in stale:
        pat = rf'\s*"{re.escape(commit)}":\s*"{re.escape(branch)}",?\n'
        init_src = re.sub(pat, "\n", init_src)
        print(f"  Removed stale branch: {repo} {commit[:12]}→{branch}")

    # Clean up empty repo dicts left behind: "repo/name": {},
    init_src = re.sub(r'\s*"[^"]+/[^"]+":\s*\{\s*\},?\n', "\n", init_src)

    if init_src != init_original:
        with open(init_path, "w") as f:
            f.write(init_src)
        print(f"  Patched: {init_path}")


# ── Build images ─────────────────────────────────────────────────────────────

def build_all(max_samples: int, max_workers: int, retries: int) -> bool:
    """Build all SWE-bench Lite instance images. Returns True if all succeed."""
    import docker
    from datasets import load_dataset
    from swebench.harness.docker_build import build_instance_images

    dataset = list(load_dataset("princeton-nlp/SWE-bench_Lite", split="test"))
    if max_samples > 0:
        dataset = dataset[:max_samples]

    print(f"\nBuilding images for {len(dataset)} instances "
          f"(workers={max_workers}) ...\n")

    client = docker.from_env()

    # build_instance_images handles base → env → instance in one call.
    successful, failed = build_instance_images(
        client=client,
        dataset=dataset,
        force_rebuild=False,
        max_workers=max_workers,
        tag="latest",
        env_image_tag="latest",
    )

    # Retry failed builds — transient failures (git clone timeouts, network
    # blips) are common when building 300 images in parallel.
    for attempt in range(1, retries + 1):
        if not failed:
            break
        failed_ids = set()
        for item in failed:
            if hasattr(item[0], "instance_id"):
                failed_ids.add(item[0].instance_id)
        if not failed_ids:
            break

        failed_dataset = [r for r in dataset if r["instance_id"] in failed_ids]
        print(f"\n--- Retry {attempt}/{retries}: "
              f"rebuilding {len(failed_dataset)} failed instances ---\n")

        retry_ok, retry_fail = build_instance_images(
            client=client,
            dataset=failed_dataset,
            force_rebuild=True,
            max_workers=max(1, max_workers // 2),
            tag="latest",
            env_image_tag="latest",
        )
        successful.extend(retry_ok)
        failed = retry_fail

    print(f"\n{'=' * 50}")
    print(f"Successful: {len(successful)}")
    print(f"Failed:     {len(failed)}")
    if failed:
        print("\nFailed instances:")
        for item in failed:
            if hasattr(item[0], "instance_id"):
                print(f"  {item[0].instance_id}")
    return len(failed) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--patch-only", action="store_true",
                        help="Only patch swebench constants, don't build")
    args = parser.parse_args()

    patch_swebench_constants()

    if args.patch_only:
        return

    ok = build_all(args.max_samples, args.max_workers, args.retries)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
