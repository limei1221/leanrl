"""Oracle evaluation: apply gold patches and measure resolve rate.

For each SWE-bench instance, applies the gold patch (row["patch"]) to the
Docker sandbox and then runs compute_swe_reward.  A perfect pipeline should
score ~100%.

Usage:
    python eval_oracle.py [--num_samples N] [--max_workers W] [--output_json out.json]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from tqdm import tqdm

from leanrl.agent.sandbox import DockerSandbox, TaskInstance
from leanrl.reward.swe_reward import compute_swe_reward


def build_task(row: dict) -> tuple[TaskInstance, str]:
    fail_to_pass = row.get("FAIL_TO_PASS", [])
    pass_to_pass = row.get("PASS_TO_PASS", [])
    if isinstance(fail_to_pass, str):
        fail_to_pass = json.loads(fail_to_pass)
    if isinstance(pass_to_pass, str):
        pass_to_pass = json.loads(pass_to_pass)

    task = TaskInstance(
        instance_id=row["instance_id"],
        repo=row.get("repo", ""),
        base_commit=row.get("base_commit", ""),
        test_patch=row.get("test_patch", ""),
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        problem_statement=row["problem_statement"],
    )
    gold_patch = row.get("patch", "")
    return task, gold_patch


def run_oracle(task: TaskInstance, gold_patch: str, test_timeout: int,
               memory_limit: str, cpu_limit: float) -> tuple[float, dict]:
    with DockerSandbox(task=task, timeout=test_timeout,
                       memory_limit=memory_limit, cpu_limit=cpu_limit) as sandbox:
        if gold_patch:
            sandbox.apply_patch(gold_patch)
        reward, info = compute_swe_reward(sandbox, task)
    return reward, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for sampling instances (default: None = no shuffle)")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--test_timeout", type=int, default=300)
    parser.add_argument("--memory_limit", type=str, default="4g")
    parser.add_argument("--cpu_limit", type=float, default=2.0)
    parser.add_argument("--output_json", type=str, default="oracle_results.json")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.split}) ...")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.num_samples is not None:
        if args.random_seed is not None:
            dataset = dataset.shuffle(seed=args.random_seed).select(
                range(min(args.num_samples, len(dataset)))
            )
        else:
            dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    items = [build_task(row) for row in dataset]
    print(f"Running oracle on {len(items)} instances with {args.max_workers} workers ...")

    results = []
    resolved = 0
    ordered = {task.instance_id: None for task, _ in items}

    with tqdm(total=len(items)) as pbar:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_oracle, task, gold, args.test_timeout,
                                args.memory_limit, args.cpu_limit): task
                for task, gold in items
            }
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    reward, info = fut.result()
                except Exception as e:
                    reward, info = 0.0, {"error": str(e)}
                info["instance_id"] = task.instance_id
                info["resolved"] = reward == 1.0
                ordered[task.instance_id] = info
                if reward == 1.0:
                    resolved += 1
                pbar.update(1)
                pbar.set_postfix({"resolved": f"{resolved}/{len(futures)}"})

    results = list(ordered.values())
    resolve_rate = resolved / len(items) if items else 0.0

    print(f"\nOracle resolve rate: {resolve_rate:.1%}  ({resolved}/{len(items)})")

    failures = [r for r in results if not r.get("resolved")]
    if failures:
        print(f"\nFailed instances ({len(failures)}):")
        for r in failures:
            err = r.get("error", "")
            f2p = f"{r.get('fail_to_pass_passed',0)}/{r.get('fail_to_pass_total',0)}"
            p2p = f"{r.get('pass_to_pass_passed',0)}/{r.get('pass_to_pass_total',0)}"
            print(f"  {r['instance_id']:50s}  f2p={f2p}  p2p={p2p}  {err}")

    with open(args.output_json, "w") as f:
        json.dump({"resolve_rate": resolve_rate, "resolved": resolved,
                   "total": len(items), "results": results}, f, indent=2)
    print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
