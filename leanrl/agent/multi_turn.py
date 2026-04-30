"""Multi-turn agent executor for SWE-bench tasks.

Implements the agent loop: prompt -> generate action -> execute in sandbox ->
observe feedback -> generate next action -> ... -> final reward from test suite.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

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


SYSTEM_PROMPT = """Fix the bug in the repo at /testbed.

Format every turn as: brief reasoning in plain text, then a bash command in <bash>...</bash>. The <bash> block must contain shell commands only — no comments, no prose. Emit <done/> when the fix is complete and tests pass.

Rules:
- Each action runs in a fresh shell. Start with `cd /testbed && ...`.
- Read files in bounded ranges with `sed -n '<a>,<b>p'`, `head`, `tail`, or `grep -n`. Never `cat` a source file.
- No interactive editors (vi/nano/emacs).
- Modify source files only — do not change tests or configuration.
- Run the relevant tests before emitting <done/>."""


def build_initial_prompt(problem_statement: str) -> list[dict[str, str]]:
    """Build the initial chat messages for a SWE-bench task."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Issue\n{problem_statement}"},
    ]


# `cat <file>` invocations waste context (whole-file dump). Match `cat`
# preceded by start/separator and NOT followed by `<<` (heredoc writes are
# legitimate). `cat < file` (single-redirect read) still counts.
_CAT_PAT = re.compile(r"(?:^|[\s;&|`(])cat\b(?!\s*<<)")


def _new_stats() -> dict:
    return {
        "total_turns": 0,
        "valid_actions": 0,
        "successful_actions": 0,
        "files_modified": False,
        "used_done": False,
        "tokenizer_drift": False,
        "cat_invocations": 0,
    }


def _pool_task(task: TaskInstance, pool_key: str) -> TaskInstance:
    """Clone a TaskInstance with a unique pool key but the original Docker image."""
    image = task.docker_image or f"sweb.eval.x86_64.{task.instance_id}:latest"
    return TaskInstance(
        instance_id=pool_key,
        repo=task.repo,
        base_commit=task.base_commit,
        test_patch=task.test_patch,
        fail_to_pass=task.fail_to_pass,
        pass_to_pass=task.pass_to_pass,
        problem_statement=task.problem_statement,
        docker_image=image,
    )


@dataclass
class _Traj:
    """Per-trajectory mutable state for a batched multi-turn rollout."""

    task: TaskInstance
    pool_key: str
    messages: list[dict[str, str]]
    sandbox: DockerSandbox | None = None
    prompt_ids: Tensor | None = None
    response_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    response_mask: list[float] = field(default_factory=list)
    response_texts: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=_new_stats)
    active: bool = True


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
        all_rollouts, all_rewards = self._run_batched_rollouts(tasks, G, tokenizer)

        rewards = torch.tensor(all_rewards, dtype=torch.float32)
        advantages = compute_grpo_advantages(rewards, G, eps=cfg.grpo.advantage_eps)

        # Truncate before ref log-probs to avoid OOM on long multi-turn sequences
        max_seq_len = cfg.training.max_seq_len
        if max_seq_len > 0:
            from leanrl.experience import truncate_rollout
            all_rollouts = [truncate_rollout(r, max_seq_len) for r in all_rollouts]

        # Free GPU 0 for the policy forward in _refresh_old_logprobs over long
        # multi-turn sequences; bring the ref model back for its own pass.
        if self.ref_model is not None:
            self.ref_model.offload_to_cpu()
        self._refresh_old_logprobs(all_rollouts, tokenizer)
        if self.ref_model is not None:
            self.ref_model.reload_to_gpu()
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

    def _run_batched_rollouts(
        self,
        tasks: list[TaskInstance],
        n_samples: int,
        tokenizer,
    ) -> tuple[list[RolloutResult], list[float]]:
        """Drive all task/sample trajectories with one batched vLLM call per turn.

        Sandboxes stream in/out so live containers stay capped at
        ``max_concurrent_sandboxes`` while every in-flight conversation is
        sent to vLLM as a single batch.
        """
        import ray

        cfg = self.config
        max_concurrent = max(1, cfg.swe.max_concurrent_sandboxes)
        max_turns = cfg.swe.max_turns

        queue: list[_Traj] = []
        for task_idx, task in enumerate(tasks):
            for s in range(n_samples):
                queue.append(_Traj(
                    task=task,
                    pool_key=f"{task.instance_id}__sample_{s}_{task_idx * n_samples + s}",
                    messages=build_initial_prompt(task.problem_statement),
                ))
        if not queue:
            return [], []

        waiting = list(queue)
        active: list[_Traj] = []
        pool = ThreadPoolExecutor(max_workers=max_concurrent)
        try:
            while waiting or active:
                slots = max_concurrent - len(active)
                if slots > 0 and waiting:
                    admitted, waiting = waiting[:slots], waiting[slots:]
                    sb_futs = {
                        pool.submit(self.sandbox_pool.get_sandbox,
                                    _pool_task(t.task, t.pool_key)): t
                        for t in admitted
                    }
                    for fut in as_completed(sb_futs):
                        t = sb_futs[fut]
                        try:
                            t.sandbox = fut.result()
                            active.append(t)
                        except Exception as e:
                            logger.warning(
                                f"Sandbox creation failed for {t.task.instance_id}: {e}"
                            )

                if not active:
                    continue

                for t in active:
                    t.stats["total_turns"] += 1
                conversations = [self._render(t.messages, tokenizer) for t in active]
                try:
                    results = ray.get(self.rollout_engine.generate.remote(
                        conversations,
                        n_samples=1,
                        max_new_tokens=cfg.rollout.max_new_tokens,
                        temperature=cfg.rollout.temperature,
                    ))
                except Exception as e:
                    logger.warning(f"Batched SWE generation failed: {e}")
                    for t in active:
                        t.active = False
                    results = []

                action_jobs: list[tuple[_Traj, str]] = []
                for t, r in zip(active, results):
                    action = self._consume_result(t, r)
                    if action is not None:
                        action_jobs.append((t, action))
                for t in active[len(results):]:
                    t.active = False

                if action_jobs:
                    cmd_futs = {
                        pool.submit(t.sandbox.execute, action): t
                        for t, action in action_jobs
                    }
                    for fut in as_completed(cmd_futs):
                        t = cmd_futs[fut]
                        try:
                            sr = fut.result()
                        except Exception as e:
                            sr = SandboxResult(
                                stdout="", stderr=f"Execution error: {e}", exit_code=-1,
                            )
                        if sr.exit_code == 0:
                            t.stats["successful_actions"] += 1
                        t.messages.append({
                            "role": "user",
                            "content": (
                                f"[Observation]\n{format_observation(sr)}\n\n"
                                "Continue fixing the issue."
                            ),
                        })

                active = [
                    t for t in active
                    if t.active and t.stats["total_turns"] < max_turns
                ]
        finally:
            pool.shutdown(wait=True)

        rollouts: list[RolloutResult | None] = [None] * len(queue)
        rewards: list[float] = [0.0] * len(queue)
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(self._finalize_traj, t, tokenizer): i
                for i, t in enumerate(queue)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    rollouts[i], rewards[i] = fut.result()
                except Exception as e:
                    logger.warning(
                        f"Trajectory failed for {queue[i].task.instance_id}: {e}"
                    )
                    rollouts[i] = self._make_empty_rollout(queue[i].task, tokenizer)
                    rewards[i] = 0.0

        return rollouts, rewards

    def _render(self, messages: list[dict[str, str]], tokenizer) -> str:
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        return "\n".join(m["content"] for m in messages)

    def _consume_result(self, t: _Traj, result: RolloutResult) -> str | None:
        """Apply one generation result to ``t``; return next bash action or None."""
        if t.prompt_ids is None:
            t.prompt_ids = result.prompt_ids
        else:
            ok = _append_prompt_delta(
                response_ids=t.response_ids,
                old_log_probs=t.log_probs,
                response_mask=t.response_mask,
                new_prompt_ids=result.prompt_ids.tolist(),
                initial_prompt_ids=t.prompt_ids.tolist(),
            )
            if not ok:
                logger.warning(
                    f"Chat-template re-tokenization drift for {t.task.instance_id}; "
                    "ending trajectory early."
                )
                t.stats["tokenizer_drift"] = True
                t.active = False
                return None

        generated = result.response_ids.tolist()
        t.response_ids.extend(generated)
        t.log_probs.extend(result.old_log_probs.tolist())
        t.response_mask.extend([1.0] * len(generated))
        t.response_texts.append(result.response_text)
        t.messages.append({"role": "assistant", "content": result.response_text})

        action_type, action_content = parse_action(result.response_text)
        if action_type == ACTION_DONE:
            t.stats["used_done"] = True
            t.active = False
            return None
        if not action_content.strip():
            t.active = False
            return None
        t.stats["valid_actions"] += 1
        t.stats["cat_invocations"] += len(_CAT_PAT.findall(action_content))
        return action_content

    def _finalize_traj(self, t: _Traj, tokenizer) -> tuple[RolloutResult, float]:
        reward = 0.0
        try:
            if t.sandbox is not None:
                diff = t.sandbox.execute("cd /testbed && git diff --name-only")
                if diff.exit_code == 0 and diff.stdout.strip():
                    t.stats["files_modified"] = True
                reward, _ = compute_swe_reward(t.sandbox, t.task, t.stats)
        finally:
            self.sandbox_pool.release_sandbox(t.pool_key)

        if t.prompt_ids is None:
            return self._make_empty_rollout(t.task, tokenizer), reward

        response_ids = torch.tensor(t.response_ids, dtype=torch.long)
        return RolloutResult(
            prompt_ids=t.prompt_ids,
            response_ids=response_ids,
            full_ids=torch.cat([t.prompt_ids, response_ids]),
            old_log_probs=torch.tensor(t.log_probs, dtype=torch.float32),
            response_text="\n".join(t.response_texts),
            prompt_text=t.task.problem_statement,
            prompt_len=len(t.prompt_ids),
            response_len=len(response_ids),
            response_mask=torch.tensor(t.response_mask, dtype=torch.float32),
        ), reward

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
