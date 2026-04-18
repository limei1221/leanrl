"""Single-turn agent executor for math-style one-shot generation tasks."""

from __future__ import annotations

from typing import Callable

import ray
import torch
from torch import Tensor

from leanrl.experience import RolloutResult, Experience, build_experience_from_rollouts
from leanrl.grpo import compute_grpo_advantages
from leanrl.utils.config import TrainConfig
from leanrl.utils.logging import logger


MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and give your final numeric answer after ####."
)


def build_math_messages(question: str) -> list[dict[str, str]]:
    """Build the chat messages for GSM8K-style math tasks."""
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


class SingleTurnExecutor:
    """Generates G completions per prompt, scores them, and builds Experience.

    This is the simplest executor: prompt -> G completions -> reward -> done.
    Used for math verification and similar single-turn tasks.
    """

    def __init__(
        self,
        rollout_engine,
        reward_fn: Callable[[list[str], list[str]], Tensor],
        ref_model,
        config: TrainConfig,
    ):
        self.rollout_engine = rollout_engine
        self.reward_fn = reward_fn
        self.ref_model = ref_model
        self.config = config

    def start_rollout(
        self,
        prompts: list[str],
        labels: list[str],
        tokenizer=None,
    ) -> tuple:
        """Start async vLLM generation and return a Ray future + metadata.

        Returns:
            (ray_future, expanded_labels, G) tuple. Pass these to
            ``finish_experience`` to build the Experience batch.
        """
        G = self.config.grpo.n_samples_per_prompt
        cfg = self.config

        formatted_prompts = []
        for p in prompts:
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                messages = build_math_messages(p)
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
            else:
                formatted_prompts.append(p)

        expanded_labels = []
        for label in labels:
            expanded_labels.extend([label] * G)

        future = self.rollout_engine.generate.remote(
            formatted_prompts,
            n_samples=G,
            max_new_tokens=cfg.rollout.max_new_tokens,
            temperature=cfg.rollout.temperature,
        )
        return future, expanded_labels, G

    def finish_experience(
        self,
        rollout_future,
        expanded_labels: list[str],
        G: int,
        tokenizer=None,
    ) -> Experience:
        """Wait for rollouts and build an Experience batch.

        Args:
            rollout_future: Ray ObjectRef from ``start_rollout``.
            expanded_labels: B*G labels (already expanded).
            G: samples per prompt.
            tokenizer: tokenizer for ref log-prob computation.

        Returns:
            Experience batch ready for training.
        """
        cfg = self.config

        rollouts: list[RolloutResult] = ray.get(rollout_future)
        logger.info(
            f"Generated {len(rollouts)} rollouts for "
            f"{len(rollouts) // G} prompts (G={G})"
        )

        responses = [r.response_text for r in rollouts]
        rewards = self.reward_fn(responses, expanded_labels)
        reward_mean = rewards.mean().item()
        reward_nonzero = (rewards > 0).float().mean().item()
        logger.info(f"Rewards: mean={reward_mean:.3f}, pass_rate={reward_nonzero:.3f}")

        advantages = compute_grpo_advantages(rewards, G, eps=cfg.grpo.advantage_eps)

        max_seq_len = cfg.training.max_seq_len
        if max_seq_len > 0:
            from leanrl.experience import truncate_rollout
            rollouts = [truncate_rollout(r, max_seq_len) for r in rollouts]

        ref_log_probs_list = self._compute_ref_logprobs(rollouts, tokenizer)

        experience = build_experience_from_rollouts(
            rollouts=rollouts,
            rewards=rewards,
            advantages=advantages,
            ref_log_probs_list=ref_log_probs_list,
            pad_token_id=tokenizer.pad_token_id if tokenizer else 0,
            max_seq_len=cfg.training.max_seq_len,
        )
        experience.labels = expanded_labels
        return experience

    def execute(
        self,
        prompts: list[str],
        labels: list[str],
        tokenizer=None,
    ) -> Experience:
        """Run single-turn rollout + reward + advantage computation.

        Synchronous wrapper around ``start_rollout`` + ``finish_experience``.
        """
        future, expanded_labels, G = self.start_rollout(prompts, labels, tokenizer)
        return self.finish_experience(future, expanded_labels, G, tokenizer)

    def _compute_ref_logprobs(self, rollouts: list[RolloutResult], tokenizer) -> list[Tensor]:
        """Compute reference model log-probs for each rollout, in mini-batches
        to avoid OOM from large logits tensors (batch * seq_len * vocab_size)."""
        if self.ref_model is None:
            return [torch.zeros_like(r.old_log_probs) for r in rollouts]

        from leanrl.experience import pad_sequences

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
