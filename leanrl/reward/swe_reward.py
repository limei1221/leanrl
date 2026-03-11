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
    results = {}

    # Match lines like: "test_foo.py::test_bar PASSED" or "FAILED"
    pattern = re.compile(r"(\S+::\S+)\s+(PASSED|FAILED|ERROR)", re.IGNORECASE)
    for match in pattern.finditer(output):
        test_name = match.group(1)
        status = match.group(2).lower()
        results[test_name] = status

    return results


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
            if test_results.get(test) != "passed":
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
