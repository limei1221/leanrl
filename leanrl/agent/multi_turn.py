"""Multi-turn agent executor for SWE-bench tasks.

Implements the agent loop: prompt -> generate action -> execute in sandbox ->
observe feedback -> generate next action -> ... -> final reward from test suite.
"""

from __future__ import annotations

import re

import ray
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
ACTION_EDIT = "edit"
ACTION_DONE = "done"


def parse_action(text: str) -> tuple[str, str]:
    """Parse the model's output into an action type and content.

    Expected formats:
        <bash>command here</bash>
        <edit path="file.py">new content</edit>
        <done/>

    Returns:
        (action_type, action_content) tuple.
    """
    # <bash>...</bash>
    bash_match = re.search(r"<bash>(.*?)</bash>", text, re.DOTALL)
    if bash_match:
        return ACTION_BASH, bash_match.group(1).strip()

    # <edit path="...">...</edit>
    edit_match = re.search(r'<edit\s+path="([^"]+)">(.*?)</edit>', text, re.DOTALL)
    if edit_match:
        path = edit_match.group(1)
        content = edit_match.group(2)
        return ACTION_EDIT, f"{path}\n{content}"

    # <done/>
    if "<done" in text.lower():
        return ACTION_DONE, ""

    # Fall back: treat entire output as bash
    return ACTION_BASH, text.strip()


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


SYSTEM_PROMPT = """You are a software engineer fixing a bug in a repository. You have access to a Docker container with the repository checked out.

Available actions:
- <bash>command</bash> : Run a shell command
- <edit path="filepath">new file content</edit> : Write/overwrite a file
- <done/> : Signal that you've completed the fix

After each action, you'll see the output. Use it to debug and iterate.
Fix the issue described below, then run the tests to verify your fix works."""


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
    ):
        self.rollout_engine = rollout_engine
        self.ref_model = ref_model
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

        for task in tasks:
            task_rollouts, task_rewards = self._run_task_rollouts(task, G, tokenizer)
            all_rollouts.extend(task_rollouts)
            all_rewards.extend(task_rewards)

        rewards = torch.tensor(all_rewards, dtype=torch.float32)
        advantages = compute_grpo_advantages(rewards, G, eps=cfg.grpo.advantage_eps)

        ref_log_probs_list = self._compute_ref_logprobs(all_rollouts)

        experience = self._build_experience(
            all_rollouts, rewards, advantages, ref_log_probs_list, tokenizer
        )

        logger.info(
            f"Multi-turn: {len(tasks)} tasks, {len(all_rollouts)} trajectories, "
            f"reward_mean={rewards.mean():.3f}, pass_rate={(rewards > 0).float().mean():.3f}"
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
                trajectory = self._run_single_trajectory(task, sandbox, max_turns, tokenizer)
                reward, info = compute_swe_reward(sandbox, task)
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
    ) -> RolloutResult:
        """Run a single multi-turn trajectory.

        Accumulates all generated tokens into a single RolloutResult,
        with a response_mask that marks only model-generated tokens.
        """
        # Build initial prompt
        initial_prompt = f"{SYSTEM_PROMPT}\n\n## Issue\n{task.problem_statement}"
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": initial_prompt}]
            conversation = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            conversation = initial_prompt

        all_response_ids: list[int] = []
        all_log_probs: list[float] = []
        prompt_ids_tensor = None

        for turn in range(max_turns):
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

            # Accumulate response tokens and log probs
            all_response_ids.extend(result.response_ids.tolist())
            all_log_probs.extend(result.old_log_probs.tolist())

            # Parse action and execute
            action_type, action_content = parse_action(result.response_text)

            if action_type == ACTION_DONE:
                break

            if action_type == ACTION_BASH:
                sandbox_result = sandbox.execute(action_content)
            elif action_type == ACTION_EDIT:
                lines = action_content.split("\n", 1)
                if len(lines) == 2:
                    sandbox_result = sandbox.write_file(lines[0], lines[1])
                else:
                    sandbox_result = SandboxResult("", "Invalid edit format", 1)
            else:
                sandbox_result = sandbox.execute(action_content)

            observation = format_observation(sandbox_result)

            # Append observation to conversation for next turn
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                conversation = result.prompt_text + result.response_text
                obs_turn = f"\n\n[Observation]\n{observation}\n\nContinue fixing the issue."
                conversation += obs_turn
            else:
                conversation = result.prompt_text + result.response_text
                conversation += f"\n\n[Observation]\n{observation}\n\n"

            # Tokenize observation tokens (these are NOT model-generated)
            if tokenizer:
                obs_tokens = tokenizer.encode(
                    f"\n\n[Observation]\n{observation}\n\n",
                    add_special_tokens=False,
                )
                # Pad log_probs with zeros for observation tokens (masked out during training)
                all_response_ids.extend(obs_tokens)
                all_log_probs.extend([0.0] * len(obs_tokens))

        # Assemble final RolloutResult
        if prompt_ids_tensor is None:
            return self._make_empty_rollout(task, tokenizer)

        response_ids = torch.tensor(all_response_ids, dtype=torch.long)
        old_log_probs = torch.tensor(all_log_probs, dtype=torch.float32)
        full_ids = torch.cat([prompt_ids_tensor, response_ids])

        return RolloutResult(
            prompt_ids=prompt_ids_tensor,
            response_ids=response_ids,
            full_ids=full_ids,
            old_log_probs=old_log_probs,
            response_text=conversation,
            prompt_text=initial_prompt,
            prompt_len=len(prompt_ids_tensor),
            response_len=len(response_ids),
        )

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
        )

    def _compute_ref_logprobs(self, rollouts: list[RolloutResult]) -> list[Tensor]:
        """Compute reference model log-probs for each rollout."""
        if self.ref_model is None:
            return [torch.zeros_like(r.old_log_probs) for r in rollouts]

        input_ids = pad_sequences([r.full_ids for r in rollouts], pad_value=0)
        input_ids = input_ids.to(self.ref_model.device)
        attention_mask = (input_ids != 0).long().to(self.ref_model.device)
        response_ids = pad_sequences([r.response_ids for r in rollouts], pad_value=0)
        response_ids = response_ids.to(self.ref_model.device)

        ref_lp = self.ref_model.forward_logprobs(input_ids, attention_mask, response_ids)

        result = []
        for i, r in enumerate(rollouts):
            result.append(ref_lp[i, : r.response_len].cpu())
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
        )

    def cleanup(self):
        """Release all sandboxes."""
        self.sandbox_pool.release_all()
