"""Evaluate a model's resolve rate on SWE-bench Lite.

Runs the same multi-turn agent loop used during training: for each problem the
model iteratively issues <bash>/<edit>/<done> actions inside a Docker sandbox,
then the FAIL_TO_PASS / PASS_TO_PASS tests are executed to determine whether
the issue is resolved.

Usage:
    python eval_swe.py --model_name_or_path <path> [options]

Requirements:
    - Docker running with SWE-bench images pulled (run scripts/setup_swe_docker.sh)
    - pip install docker datasets transformers tqdm
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from leanrl.agent.multi_turn import SYSTEM_PROMPT, parse_action, format_observation, ACTION_DONE, ACTION_BASH, ACTION_EDIT
from leanrl.agent.sandbox import DockerSandbox, TaskInstance, SandboxResult
from leanrl.reward.swe_reward import compute_swe_reward


def build_task(row: dict) -> TaskInstance:
    """Convert a SWE-bench dataset row into a TaskInstance."""
    fail_to_pass = row.get("FAIL_TO_PASS", [])
    pass_to_pass = row.get("PASS_TO_PASS", [])
    # Dataset stores these as JSON strings in some splits
    if isinstance(fail_to_pass, str):
        fail_to_pass = json.loads(fail_to_pass)
    if isinstance(pass_to_pass, str):
        pass_to_pass = json.loads(pass_to_pass)

    return TaskInstance(
        instance_id=row["instance_id"],
        repo=row.get("repo", ""),
        base_commit=row.get("base_commit", ""),
        test_patch=row.get("test_patch", ""),
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        problem_statement=row["problem_statement"],
    )


def run_trajectory(
    task: TaskInstance,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_turns: int,
    max_new_tokens: int,
    device: str,
    sandbox_timeout: int,
    test_timeout: int,
    memory_limit: str,
    cpu_limit: float,
    model_lock: threading.Lock,
    verbose: bool = False,
) -> tuple[float, dict]:
    """Run a single multi-turn trajectory for one SWE-bench task.

    Returns:
        reward: 1.0 if all tests pass, 0.0 otherwise.
        info: dict with per-task test results.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Issue\n{task.problem_statement}"},
    ]

    with DockerSandbox(
        task=task,
        timeout=sandbox_timeout,
        memory_limit=memory_limit,
        cpu_limit=cpu_limit,
    ) as sandbox:
        last_action = None
        repeat_count = 0

        for turn in range(max_turns):
            conversation = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Serialize GPU inference: only one thread may call generate() at a time.
            with model_lock:
                inputs = tokenizer(
                    conversation,
                    return_tensors="pt",
                    truncation=True,
                    max_length=16384,
                ).to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

            input_len = inputs["input_ids"].shape[1]
            response_text = tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            )

            action_type, action_content = parse_action(response_text)

            if verbose:
                print(f"\n[{task.instance_id}] Turn {turn+1}")
                print(f"  Response ({len(response_text)} chars): {response_text[:200]!r}")
                print(f"  Action: {action_type}, content ({len(action_content)} chars): {action_content[:150]!r}")

            # Detect repetitive loops — if the model repeats the same action
            # 2+ times, nudge it to try something different.
            current_action = (action_type, action_content.strip())
            if current_action == last_action:
                repeat_count += 1
                if repeat_count >= 2:
                    if verbose:
                        print(f"  → loop detected ({repeat_count} repeats), nudging model")
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content":
                        "You are repeating the same action. Try a different approach."
                    })
                    last_action = None
                    repeat_count = 0
                    continue
            else:
                last_action = current_action
                repeat_count = 0

            # Add model response to message history
            messages.append({"role": "assistant", "content": response_text})

            if action_type == ACTION_DONE:
                # Genuine <done/> tag — model signalled completion
                if re.search(r"<done\s*/?>", response_text, re.IGNORECASE):
                    break
                # No parseable action — prompt the model to give one
                if verbose:
                    print(f"  → no action found, prompting for explicit action")
                messages.append({"role": "user", "content":
                    "Please provide a bash command wrapped in <bash>...</bash> tags, "
                    "or <done/> if you are finished."
                })
                continue

            if not action_content.strip():
                if verbose:
                    print(f"  → empty action_content, stopping")
                break

            if action_type == ACTION_BASH:
                result = sandbox.execute(action_content)
            elif action_type == ACTION_EDIT:
                lines = action_content.split("\n", 1)
                if len(lines) == 2:
                    file_path = lines[0]
                    result = sandbox.write_file(file_path, lines[1])
                else:
                    file_path = ""
                    result = SandboxResult("", "Invalid edit format", 1)
            else:
                # Unknown action type — skip rather than running raw text as bash
                break

            if verbose:
                obs_preview = (result.stdout + result.stderr)[:300]
                print(f"  Sandbox exit={result.exit_code}: {obs_preview!r}")

            observation = format_observation(result)

            # Provide explicit feedback for edits so the model knows it worked
            if action_type == ACTION_EDIT and result.exit_code == 0:
                observation = f"File {file_path} written successfully."

            # Add observation as a new user turn to maintain proper chat format
            messages.append({"role": "user", "content": f"[Observation]\n{observation}\n\nContinue fixing the issue."})

        # Use a longer timeout for the final test scoring run.
        sandbox.timeout = test_timeout
        reward, info = compute_swe_reward(sandbox, task)

    return reward, info


def evaluate(
    model_path: str,
    tasks: list[TaskInstance],
    max_turns: int,
    max_new_tokens: int,
    max_workers: int,
    sandbox_timeout: int,
    test_timeout: int,
    memory_limit: str,
    cpu_limit: float,
    device: str,
    verbose: bool = False,
) -> tuple[float, list[dict]]:
    print(f"\nLoading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    results = []
    resolved = 0
    # Lock ensures only one thread calls model.generate() at a time.
    # Parallel workers overlap sandbox I/O (docker exec, file writes) with inference.
    model_lock = threading.Lock()

    def _run(task):
        return run_trajectory(
            task, model, tokenizer,
            max_turns, max_new_tokens, device,
            sandbox_timeout, test_timeout,
            memory_limit, cpu_limit, model_lock,
            verbose=verbose,
        )

    with tqdm(total=len(tasks), desc=model_path.split("/")[-1]) as pbar:
        if max_workers == 1:
            for task in tasks:
                try:
                    reward, info = _run(task)
                except Exception as e:
                    reward, info = 0.0, {"error": str(e)}

                info["instance_id"] = task.instance_id
                info["resolved"] = reward == 1.0
                results.append(info)
                if reward == 1.0:
                    resolved += 1
                pbar.update(1)
                pbar.set_postfix({"resolved": f"{resolved}/{len(results)}"})
        else:
            # Parallel: sandbox I/O runs concurrently; GPU inference is serialized
            # by model_lock so CUDA is never accessed from two threads at once.
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for task in tasks:
                    futures[executor.submit(_run, task)] = task

                ordered = {task.instance_id: None for task in tasks}
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
                    pbar.set_postfix({"resolved": f"{resolved}/{len(tasks)}"})

            results = list(ordered.values())

    resolve_rate = resolved / len(tasks) if tasks else 0.0
    return resolve_rate, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on SWE-bench Lite.")
    parser.add_argument("--model_name_or_path", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                        help="SWE-bench dataset name (default: princeton-nlp/SWE-bench_Lite)")
    parser.add_argument("--split", default="test",
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of instances to evaluate (default: all)")
    parser.add_argument("--max_turns", type=int, default=5,
                        help="Max agent turns per instance (default: 5)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens to generate per turn (default: 2048)")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Parallel sandbox workers (default: 8)")
    parser.add_argument("--sandbox_timeout", type=int, default=120,
                        help="Per-command sandbox timeout during trajectory in seconds (default: 120)")
    parser.add_argument("--test_timeout", type=int, default=300,
                        help="Timeout for the final pytest scoring run in seconds (default: 300)")
    parser.add_argument("--memory_limit", type=str, default="4g",
                        help="Docker memory limit per sandbox (default: 4g)")
    parser.add_argument("--cpu_limit", type=float, default=2.0,
                        help="Docker CPU limit per sandbox (default: 2.0)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default="eval_results.json",
                        help="Optional path to write per-instance results as JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each turn's model output and action for debugging")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ({args.split}) ...")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    tasks = [build_task(row) for row in dataset]
    print(f"Evaluating on {len(tasks)} instances.")

    resolve_rate, results = evaluate(
        model_path=args.model_name_or_path,
        tasks=tasks,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        max_workers=args.max_workers,
        sandbox_timeout=args.sandbox_timeout,
        test_timeout=args.test_timeout,
        memory_limit=args.memory_limit,
        cpu_limit=args.cpu_limit,
        device=args.device,
        verbose=args.verbose,
    )

    resolved = sum(1 for r in results if r.get("resolved"))
    print("\n" + "=" * 50)
    print(f"Model:        {args.model_name_or_path}")
    print(f"Dataset:      {args.dataset} ({args.split})")
    print(f"Resolve rate: {resolve_rate:.1%}  ({resolved}/{len(tasks)})")
    print("=" * 50)

    if args.output_json:
        import json as _json
        resolved = int(resolved)
        total = int(len(tasks))
        with open(args.output_json, "w") as f:
            _json.dump(
                {
                    "model": args.model_name_or_path,
                    "dataset": args.dataset,
                    "split": args.split,
                    "resolve_rate": resolve_rate,
                    "resolved": resolved,
                    "total": total,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Per-instance results written to {args.output_json}")


if __name__ == "__main__":
    main()
