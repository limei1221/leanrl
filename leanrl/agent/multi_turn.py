"""Multi-turn agent executor for SWE-bench tasks.

Implements the agent loop: prompt -> generate action -> execute in sandbox ->
observe feedback -> generate next action -> ... -> final reward from test suite.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch import Tensor

from leanrl.agent.sandbox import DockerSandbox, SandboxPool, TaskInstance, SandboxResult
from leanrl.experience import RolloutResult, Experience, pad_sequences
from leanrl.grpo import compute_grpo_advantages
from leanrl.reward.swe_reward import compute_swe_reward
from leanrl.utils.config import TrainConfig
from leanrl.utils.logging import logger


# Action types the agent can emit
ACTION_BASH = "bash"
ACTION_DONE = "done"


def parse_action(text: str) -> tuple[str, str]:
    """Parse the model's output into an action type and content.

    Supports two formats:
      1. Structured tags:
           <bash>command</bash>
           <done/>
      2. Markdown code fences:
           ```bash / ```sh
           command
           ```

    Returns:
        (action_type, action_content) tuple.
    """
    # <bash>...</bash>
    bash_match = re.search(r"<bash>(.*?)</bash>", text, re.DOTALL)
    if bash_match:
        return ACTION_BASH, bash_match.group(1).strip()

    # ```bash / ```sh fenced blocks
    bash_fence = re.search(r"```(?:bash|sh)\s*\n(.*?)```", text, re.DOTALL)
    if bash_fence:
        content = bash_fence.group(1).strip()
        # Model sometimes puts <done/> inside a bash block — treat as done signal
        if re.match(r"<done\s*/?>$", content, re.IGNORECASE):
            return ACTION_DONE, ""
        return ACTION_BASH, content

    # <done/>
    if re.search(r"<done\s*/?>", text, re.IGNORECASE):
        return ACTION_DONE, ""

    # No recognizable action — treat as done
    return ACTION_DONE, ""


def format_observation(result: SandboxResult, max_chars: int = 4096) -> str:
    """Format sandbox execution result as an observation for the model."""
    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    if result.timed_out:
        output += "\n[TIMEOUT: command exceeded time limit]"
    if result.exit_code != 0:
        output += f"\n[exit code: {result.exit_code}]"

    # Truncate long outputs
    if len(output) > max_chars:
        output = output[:max_chars // 2] + "\n...[truncated]...\n" + output[-max_chars // 2:]
    return output


def _append_prompt_delta(
    response_ids: list[int],
    old_log_probs: list[float],
    response_mask: list[float],
    new_prompt_ids: list[int],
    initial_prompt_ids: list[int],
) -> bool:
    """Append chat-template boilerplate between turns as masked context.

    Multi-turn generation re-renders the full chat template every turn. The
    new turn's prompt should extend the accumulated trajectory
    (``initial_prompt_ids + response_ids``) as a prefix; the suffix — chat
    wrapping plus the new user/observation turn — is appended with mask 0 so
    it does not contribute to the loss.

    Returns False when the new prompt does not extend the accumulated
    trajectory, which almost always means a previously-generated assistant
    response did not round-trip cleanly through detokenize→tokenize. Caller
    should end the trajectory early rather than corrupt the token stream.
    """
    accumulated = initial_prompt_ids + response_ids
    if (
        len(new_prompt_ids) < len(accumulated)
        or new_prompt_ids[: len(accumulated)] != accumulated
    ):
        return False

    delta = new_prompt_ids[len(accumulated):]
    if delta:
        response_ids.extend(delta)
        old_log_probs.extend([0.0] * len(delta))
        response_mask.extend([0.0] * len(delta))
    return True


SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer shell to solve programming tasks.
You will be given a bug report for a repository checked out at /testbed. Fix the bug by modifying the source code.

For each response, include your reasoning, then a bash command wrapped in <bash>...</bash> tags.
When you are finished, write <done/>.

Important rules:
- Only bash actions are allowed. Use shell tools (sed, awk, cat, grep, python3 -c, etc.) to inspect and edit files.
- Do not use interactive editors (vi, nano, emacs).
- Each action runs in a fresh shell, so always use: cd /testbed && ...
- Pipe long outputs through head or tail to keep them short.
- Only modify source code files, not tests or configuration.
- Run relevant tests before finishing to verify your fix.
- Emit <done/> when finished."""


def build_initial_prompt(problem_statement: str) -> list[dict[str, str]]:
    """Build the initial chat messages for a SWE-bench task."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Issue\n{problem_statement}"},
    ]


class MultiTurnExecutor:
    """Executes multi-turn agent interactions for SWE-bench tasks.

    For each task instance:
    1. Set up Docker sandbox
    2. Run the agent loop (generate action -> execute -> observe -> repeat)
    3. Score final state with test suite
    4. Collect trajectory for training
    """

    def __init__(
        self,
        rollout_engine,
        ref_model,
        config: TrainConfig,
        policy_model=None,
    ):
        self.rollout_engine = rollout_engine
        self.ref_model = ref_model
        self.policy_model = policy_model
        self.config = config
        self.sandbox_pool = SandboxPool(
            max_concurrent=config.swe.max_concurrent_sandboxes,
            image_prefix=config.swe.docker_image_prefix,
            timeout=config.swe.sandbox_timeout,
            memory_limit=config.swe.sandbox_memory_limit,
            cpu_limit=config.swe.sandbox_cpu_limit,
        )

    def execute(
        self,
        prompts: list[str],
        labels: list[str],
        tokenizer=None,
    ) -> Experience:
        """Run multi-turn rollouts for a batch of SWE-bench tasks.

        Args:
            prompts: list of B problem statements.
            labels: list of B task metadata (JSON strings or TaskInstance dicts).
            tokenizer: tokenizer for chat template formatting.

        Returns:
            Experience batch for training.
        """
        cfg = self.config
        G = cfg.grpo.n_samples_per_prompt

        tasks = self._parse_tasks(prompts, labels)

        all_rollouts: list[RolloutResult] = []
        all_rewards: list[float] = []

        max_workers = self.config.swe.max_concurrent_sandboxes
        with ThreadPoolExecutor(max_workers=min(len(tasks), max_workers)) as executor:
            futures = {
                executor.submit(self._run_task_rollouts, task, G, tokenizer): i
                for i, task in enumerate(tasks)
            }
            results: list[tuple] = [None] * len(tasks)
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        for task_rollouts, task_rewards in results:
            all_rollouts.extend(task_rollouts)
            all_rewards.extend(task_rewards)

        rewards = torch.tensor(all_rewards, dtype=torch.float32)
        advantages = compute_grpo_advantages(rewards, G, eps=cfg.grpo.advantage_eps)

        # Truncate before ref log-probs to avoid OOM on long multi-turn sequences
        max_seq_len = cfg.training.max_seq_len
        if max_seq_len > 0:
            from leanrl.experience import truncate_rollout
            all_rollouts = [truncate_rollout(r, max_seq_len) for r in all_rollouts]

        self._refresh_old_logprobs(all_rollouts, tokenizer)
        ref_log_probs_list = self._compute_ref_logprobs(all_rollouts, tokenizer)

        experience = self._build_experience(
            all_rollouts, rewards, advantages, ref_log_probs_list, tokenizer
        )

        # pass_rate reflects test success (reward > 0.3 shaping cap), not shaping
        logger.info(
            f"Multi-turn: {len(tasks)} tasks, {len(all_rollouts)} trajectories, "
            f"reward_mean={rewards.mean():.3f}, "
            f"pass_rate={(rewards > 0.3).float().mean():.3f}"
        )
        return experience

    def _parse_tasks(self, prompts: list[str], labels: list[str]) -> list[TaskInstance]:
        """Convert prompts/labels into TaskInstance objects."""
        import json

        tasks = []
        for prompt, label in zip(prompts, labels):
            if isinstance(label, str):
                try:
                    meta = json.loads(label)
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            elif isinstance(label, dict):
                meta = label
            else:
                meta = {}

            tasks.append(TaskInstance(
                instance_id=meta.get("instance_id", f"task_{len(tasks)}"),
                repo=meta.get("repo", ""),
                base_commit=meta.get("base_commit", ""),
                test_patch=meta.get("test_patch", ""),
                fail_to_pass=meta.get("fail_to_pass", []),
                pass_to_pass=meta.get("pass_to_pass", []),
                problem_statement=prompt,
                docker_image=meta.get("docker_image", None),
            ))
        return tasks

    def _run_task_rollouts(
        self,
        task: TaskInstance,
        n_samples: int,
        tokenizer,
    ) -> tuple[list[RolloutResult], list[float]]:
        """Run G independent rollouts for a single task.

        Each rollout gets its own sandbox (or resets the sandbox between runs).
        """
        rollouts = []
        rewards = []
        max_turns = self.config.swe.max_turns

        for sample_idx in range(n_samples):
            try:
                sandbox = self.sandbox_pool.get_sandbox(task)
            except Exception as e:
                logger.warning(f"Sandbox creation failed for {task.instance_id}: {e}")
                rollouts.append(self._make_empty_rollout(task, tokenizer))
                rewards.append(0.0)
                continue

            try:
                trajectory, traj_stats = self._run_single_trajectory(
                    task, sandbox, max_turns, tokenizer
                )
                reward, info = compute_swe_reward(sandbox, task, traj_stats)
                rollouts.append(trajectory)
                rewards.append(reward)
            except Exception as e:
                logger.warning(f"Trajectory failed for {task.instance_id} sample {sample_idx}: {e}")
                rollouts.append(self._make_empty_rollout(task, tokenizer))
                rewards.append(0.0)
            finally:
                self.sandbox_pool.release_sandbox(task.instance_id)

        return rollouts, rewards

    def _run_single_trajectory(
        self,
        task: TaskInstance,
        sandbox: DockerSandbox,
        max_turns: int,
        tokenizer,
    ) -> tuple[RolloutResult, dict]:
        """Run a single multi-turn trajectory.

        Accumulates the exact rendered chat-token stream into a single
        RolloutResult, with a response_mask that marks only model-generated
        tokens.

        Returns:
            (rollout, trajectory_stats) where trajectory_stats contains
            signals for dense reward shaping.
        """
        # Build initial prompt
        messages = build_initial_prompt(task.problem_statement)

        import ray

        all_response_ids: list[int] = []
        all_log_probs: list[float] = []
        all_response_mask: list[float] = []
        all_response_texts: list[str] = []
        prompt_ids_tensor = None

        # Track trajectory-level signals for dense reward
        traj_stats = {
            "total_turns": 0,
            "valid_actions": 0,
            "successful_actions": 0,
            "files_modified": False,
            "used_done": False,
            "tokenizer_drift": False,
        }

        for turn in range(max_turns):
            traj_stats["total_turns"] = turn + 1

            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                conversation = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            else:
                conversation = "\n".join(m["content"] for m in messages)

            # Generate one turn
            generation_results = ray.get(
                self.rollout_engine.generate.remote(
                    [conversation],
                    n_samples=1,
                    max_new_tokens=self.config.rollout.max_new_tokens,
                    temperature=self.config.rollout.temperature,
                )
            )

            if not generation_results:
                break

            result = generation_results[0]
            if prompt_ids_tensor is None:
                prompt_ids_tensor = result.prompt_ids
            else:
                appended = _append_prompt_delta(
                    response_ids=all_response_ids,
                    old_log_probs=all_log_probs,
                    response_mask=all_response_mask,
                    new_prompt_ids=result.prompt_ids.tolist(),
                    initial_prompt_ids=prompt_ids_tensor.tolist(),
                )
                if not appended:
                    logger.warning(
                        f"Chat-template re-tokenization drift for {task.instance_id} "
                        f"at turn {turn + 1}; ending trajectory early."
                    )
                    traj_stats["tokenizer_drift"] = True
                    break

            # Accumulate response tokens and log probs
            generated_ids = result.response_ids.tolist()
            all_response_ids.extend(generated_ids)
            all_log_probs.extend(result.old_log_probs.tolist())
            all_response_mask.extend([1.0] * len(generated_ids))

            # Add model response to message history
            all_response_texts.append(result.response_text)
            messages.append({"role": "assistant", "content": result.response_text})

            # Parse action and execute
            action_type, action_content = parse_action(result.response_text)

            if action_type == ACTION_DONE:
                traj_stats["used_done"] = True
                break

            if not action_content.strip():
                # Model emitted no executable content; stop the trajectory.
                break

            traj_stats["valid_actions"] += 1
            sandbox_result = sandbox.execute(action_content)
            if sandbox_result.exit_code == 0:
                traj_stats["successful_actions"] += 1

            observation = format_observation(sandbox_result)
            obs_content = f"[Observation]\n{observation}\n\nContinue fixing the issue."

            # Add observation as a new user turn to maintain proper chat format
            messages.append({"role": "user", "content": obs_content})

        # Check if the model modified any source files
        diff_result = sandbox.execute("cd /testbed && git diff --name-only")
        if diff_result.exit_code == 0 and diff_result.stdout.strip():
            traj_stats["files_modified"] = True

        # Assemble final RolloutResult
        if prompt_ids_tensor is None:
            return self._make_empty_rollout(task, tokenizer), traj_stats

        response_ids = torch.tensor(all_response_ids, dtype=torch.long)
        old_log_probs = torch.tensor(all_log_probs, dtype=torch.float32)
        response_mask = torch.tensor(all_response_mask, dtype=torch.float32)
        full_ids = torch.cat([prompt_ids_tensor, response_ids])

        return RolloutResult(
            prompt_ids=prompt_ids_tensor,
            response_ids=response_ids,
            full_ids=full_ids,
            old_log_probs=old_log_probs,
            response_text="\n".join(all_response_texts),
            prompt_text=task.problem_statement,
            prompt_len=len(prompt_ids_tensor),
            response_len=len(response_ids),
            response_mask=response_mask,
        ), traj_stats

    def _make_empty_rollout(self, task: TaskInstance, tokenizer) -> RolloutResult:
        """Create an empty rollout for failed trajectories."""
        prompt_text = task.problem_statement
        if tokenizer:
            prompt_ids = torch.tensor(
                tokenizer.encode(prompt_text, add_special_tokens=True),
                dtype=torch.long,
            )
        else:
            prompt_ids = torch.tensor([0], dtype=torch.long)

        eos_id = tokenizer.eos_token_id if tokenizer else 0
        response_ids = torch.tensor([eos_id], dtype=torch.long)

        return RolloutResult(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            full_ids=torch.cat([prompt_ids, response_ids]),
            old_log_probs=torch.zeros(1),
            response_text="",
            prompt_text=prompt_text,
            prompt_len=len(prompt_ids),
            response_len=1,
            response_mask=torch.ones(1, dtype=torch.float32),
        )

    def _refresh_old_logprobs(self, rollouts: list[RolloutResult], tokenizer) -> None:
        """Recompute old log-probs with the policy over the training context."""
        if self.policy_model is None:
            return

        pad_id = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
        mini_bs = self.config.training.micro_batch_size

        for start in range(0, len(rollouts), mini_bs):
            chunk = rollouts[start : start + mini_bs]

            input_ids = pad_sequences(
                [r.full_ids for r in chunk],
                pad_value=pad_id,
            )

            response_lengths = torch.tensor(
                [r.response_len for r in chunk],
                dtype=torch.long,
            )
            seq_lengths = torch.tensor(
                [r.prompt_len + r.response_len for r in chunk],
                dtype=torch.long,
            )
            attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
            for i, sl in enumerate(seq_lengths):
                attention_mask[i, :sl] = 1

            max_resp_len = max(r.response_len for r in chunk)

            new_lp = self.policy_model.forward_logprobs_no_grad(
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_lengths=response_lengths,
                max_resp_len=max_resp_len,
            )

            for j, r in enumerate(chunk):
                r.old_log_probs = new_lp[j, : r.response_len].detach().cpu()

    def _compute_ref_logprobs(self, rollouts: list[RolloutResult], tokenizer) -> list[Tensor]:
        """Compute reference model log-probs for each rollout, in mini-batches
        to avoid OOM from large logits tensors (batch * seq_len * vocab_size)."""
        if self.ref_model is None:
            return [torch.zeros_like(r.old_log_probs) for r in rollouts]

        pad_id = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
        mini_bs = self.config.training.micro_batch_size

        result = [None] * len(rollouts)
        for start in range(0, len(rollouts), mini_bs):
            chunk = rollouts[start : start + mini_bs]

            input_ids = pad_sequences(
                [r.full_ids for r in chunk],
                pad_value=pad_id,
            ).to(self.ref_model.device)

            # Build attention_mask from actual lengths so EOS tokens are not
            # masked out when pad_token_id == eos_token_id.
            response_lengths = torch.tensor(
                [r.response_len for r in chunk], dtype=torch.long,
            )
            seq_lengths = torch.tensor(
                [r.prompt_len + r.response_len for r in chunk], dtype=torch.long,
            )
            attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
            for i, sl in enumerate(seq_lengths):
                attention_mask[i, :sl] = 1
            attention_mask = attention_mask.to(self.ref_model.device)

            max_resp_len = max(r.response_len for r in chunk)

            ref_lp = self.ref_model.forward_logprobs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_lengths=response_lengths,
                max_resp_len=max_resp_len,
            )

            for j, r in enumerate(chunk):
                result[start + j] = ref_lp[j, : r.response_len].cpu()

        return result

    def _build_experience(
        self,
        rollouts: list[RolloutResult],
        rewards: Tensor,
        advantages: Tensor,
        ref_log_probs_list: list[Tensor],
        tokenizer,
    ) -> Experience:
        """Build Experience from multi-turn rollouts with proper response masking."""
        from leanrl.experience import build_experience_from_rollouts

        pad_id = tokenizer.pad_token_id if tokenizer else 0
        return build_experience_from_rollouts(
            rollouts=rollouts,
            rewards=rewards,
            advantages=advantages,
            ref_log_probs_list=ref_log_probs_list,
            pad_token_id=pad_id,
            max_seq_len=self.config.training.max_seq_len,
        )

    def cleanup(self):
        """Release all sandboxes."""
        self.sandbox_pool.release_all()
