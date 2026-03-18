"""SWE-bench reward function: runs tests inside Docker sandbox."""

from __future__ import annotations

import re

import torch
from torch import Tensor

from leanrl.agent.sandbox import DockerSandbox, TaskInstance
from leanrl.utils.logging import logger


def parse_pytest_results(output: str) -> dict[str, str]:
    """Parse pytest output to determine pass/fail status of individual tests.

    Returns:
        dict mapping test name -> "passed" | "failed" | "error".
    """
    # Strip ANSI escape codes (e.g. \x1b[32mPASSED\x1b[0m) before parsing
    ansi_escape = re.compile(r"\x1b\[[0-9;]*[mK]")
    output = ansi_escape.sub("", output)

    results = {}

    # sympy bin/test synthetic results: "SYMPY_RESULT: test_name PASSED"
    sympy_pat = re.compile(r"^SYMPY_RESULT: (\S+)\s+(PASSED|FAILED)", re.MULTILINE)
    for m in sympy_pat.finditer(output):
        results[m.group(1)] = m.group(2).lower()

    # pytest verbose: "path/to/test.py::Class::test_foo PASSED"
    pytest_pat = re.compile(r"(\S+::\S+)\s+(PASSED|FAILED|ERROR)", re.IGNORECASE)
    for m in pytest_pat.finditer(output):
        results[m.group(1)] = m.group(2).lower()

    # unittest verbose with module.Class: "test_foo (module.Class) ... ok"
    # Description may contain spaces (docstring-based test names).
    unittest_pat = re.compile(
        r"^(.+?)\s+\(([a-zA-Z0-9_.]+)\)\s+\.+\s+(ok|FAIL|ERROR)",
        re.MULTILINE | re.IGNORECASE,
    )
    for m in unittest_pat.finditer(output):
        method, module_class = m.group(1), m.group(2)
        test_name = f"{method} ({module_class})"
        raw = m.group(3).lower()
        results[test_name] = "passed" if raw == "ok" else "failed" if raw == "fail" else "error"

    # unittest bare-docstring: "Some description ... ok" (no module.Class context)
    # Use .+? (allows parens in test names like setUp() or (#1234)); the
    # `if test_name not in results` guard below prevents overwriting module.Class matches.
    bare_pat = re.compile(
        r"^(.+?)\s+\.{3}\s+(ok|FAIL|ERROR)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    for m in bare_pat.finditer(output):
        test_name = m.group(1).strip()
        raw = m.group(2).lower()
        if test_name not in results:  # don't overwrite module.Class matches
            results[test_name] = "passed" if raw == "ok" else "failed" if raw == "fail" else "error"

    return results


_ROMAN_PY = '''\
"""Minimal roman-numeral helpers (toRoman / fromRoman) for sphinx containers."""
def toRoman(n):
    result = ""
    for val, sym in [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
                     (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]:
        while n >= val:
            result += sym; n -= val
    return result
def fromRoman(s):
    vals = {"M":1000,"D":500,"C":100,"L":50,"X":10,"V":5,"I":1}
    result, prev = 0, 0
    for ch in reversed(s.upper()):
        v = vals[ch]
        result += v if v >= prev else -v
        prev = v
    return result
'''


def _ensure_roman_package(sandbox: "DockerSandbox") -> None:  # type: ignore[name-defined]
    """Inject a minimal roman.py if the package is missing (some sphinx images)."""
    r = sandbox.execute(
        "python -c 'import roman' 2>/dev/null && echo OK || echo MISSING"
    )
    if "MISSING" in r.stdout:
        site = sandbox.execute(
            "python -c 'import site; print(site.getsitepackages()[0])'"
        ).stdout.strip()
        if site:
            sandbox.write_file(f"{site}/roman.py", _ROMAN_PY)


def compute_swe_reward(
    sandbox: DockerSandbox,
    task: TaskInstance,
) -> tuple[float, dict]:
    """Evaluate whether the current sandbox state passes the SWE-bench criteria.

    Runs the FAIL_TO_PASS and PASS_TO_PASS tests and checks:
    1. All FAIL_TO_PASS tests now pass (the fix works)
    2. All PASS_TO_PASS tests still pass (no regressions)

    Args:
        sandbox: active Docker sandbox with the current code state.
        task: TaskInstance with test specifications.

    Returns:
        reward: 1.0 if all criteria met, 0.0 otherwise.
        info: dict with detailed test results for logging.
    """
    info = {
        "fail_to_pass_total": len(task.fail_to_pass),
        "pass_to_pass_total": len(task.pass_to_pass),
        "fail_to_pass_passed": 0,
        "pass_to_pass_passed": 0,
        "error": None,
    }

    # Ensure the 'roman' package exists (missing from some sphinx containers)
    _ensure_roman_package(sandbox)

    # Apply test patch so FAIL_TO_PASS tests exist in the container.
    # Use --3way so the patch applies even when the agent has modified
    # overlapping files, and fall back to a hard-reset of test files.
    if task.test_patch:
        result = sandbox.apply_patch(task.test_patch)
        if result.exit_code != 0:
            # Reset only test files touched by the patch, then retry
            sandbox.execute("cd /testbed && git checkout HEAD -- $(grep '^diff --git' /tmp/patch.diff | sed 's|.*b/||') 2>/dev/null")
            result = sandbox.apply_patch(task.test_patch)
            if result.exit_code != 0:
                info["error"] = f"test_patch failed to apply: {result.stderr[:200]}"
                return 0.0, info

    # Run FAIL_TO_PASS tests
    if task.fail_to_pass:
        result = sandbox.run_tests(task.fail_to_pass)
        if result.timed_out:
            info["error"] = "fail_to_pass tests timed out"
            return 0.0, info

        test_results = parse_pytest_results(result.stdout + result.stderr)
        passed = sum(1 for t in task.fail_to_pass if test_results.get(t) == "passed")
        info["fail_to_pass_passed"] = passed

        for test in task.fail_to_pass:
            if test_results.get(test) != "passed":
                return 0.0, info

    # Run PASS_TO_PASS tests (regression check)
    if task.pass_to_pass:
        result = sandbox.run_tests(task.pass_to_pass)
        if result.timed_out:
            info["error"] = "pass_to_pass tests timed out"
            return 0.0, info

        test_results = parse_pytest_results(result.stdout + result.stderr)
        passed = sum(1 for t in task.pass_to_pass if test_results.get(t) == "passed")
        info["pass_to_pass_passed"] = passed

        for test in task.pass_to_pass:
            status = test_results.get(test)
            # "not found" (None) means the test doesn't exist at this commit —
            # treat as skipped rather than failed to avoid false negatives caused
            # by dataset tests that were added after the base commit.
            if status == "failed" or status == "error":
                return 0.0, info

    return 1.0, info


def compute_swe_rewards_batch(
    sandboxes: list[DockerSandbox],
    tasks: list[TaskInstance],
) -> tuple[Tensor, list[dict]]:
    """Compute rewards for a batch of SWE-bench tasks.

    Args:
        sandboxes: list of active sandboxes, one per task.
        tasks: list of TaskInstance objects.

    Returns:
        rewards: (B,) tensor of 0.0 or 1.0.
        infos: list of dicts with per-task test details.
    """
    rewards = []
    infos = []

    for sandbox, task in zip(sandboxes, tasks):
        reward, info = compute_swe_reward(sandbox, task)
        rewards.append(reward)
        infos.append(info)
        logger.info(
            f"SWE reward [{task.instance_id}]: {reward} "
            f"(f2p={info['fail_to_pass_passed']}/{info['fail_to_pass_total']}, "
            f"p2p={info['pass_to_pass_passed']}/{info['pass_to_pass_total']})"
        )

    return torch.tensor(rewards, dtype=torch.float32), infos
