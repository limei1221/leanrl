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

    def execute(
        self,
        prompts: list[str],
        labels: list[str],
        tokenizer=None,
    ) -> Experience:
        """Run single-turn rollout + reward + advantage computation.

        Args:
            prompts: list of B prompt strings.
            labels: list of B gold label strings.
            tokenizer: tokenizer for formatting prompts (optional).

        Returns:
            Experience batch ready for training.
        """
        G = self.config.grpo.n_samples_per_prompt
        cfg = self.config

        # Format prompts with chat template if tokenizer provided
        formatted_prompts = []
        for p in prompts:
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": p}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted_prompts.append(formatted)
            else:
                formatted_prompts.append(p)

        # 1. Rollout: generate G samples per prompt via vLLM
        rollouts: list[RolloutResult] = ray.get(
            self.rollout_engine.generate.remote(
                formatted_prompts,
                n_samples=G,
                max_new_tokens=cfg.rollout.max_new_tokens,
                temperature=cfg.rollout.temperature,
            )
        )
        logger.info(f"Generated {len(rollouts)} rollouts for {len(prompts)} prompts (G={G})")

        # 2. Reward: score each completion
        # Expand labels to match B*G rollouts
        expanded_labels = []
        for label in labels:
            expanded_labels.extend([label] * G)

        responses = [r.response_text for r in rollouts]
        rewards = self.reward_fn(responses, expanded_labels)
        reward_mean = rewards.mean().item()
        reward_nonzero = (rewards > 0).float().mean().item()
        logger.info(f"Rewards: mean={reward_mean:.3f}, pass_rate={reward_nonzero:.3f}")

        # 3. GRPO advantages
        advantages = compute_grpo_advantages(rewards, G, eps=cfg.grpo.advantage_eps)

        # 4. Reference log-probs (batched)
        ref_log_probs_list = self._compute_ref_logprobs(rollouts, tokenizer)

        # 5. Build Experience
        experience = build_experience_from_rollouts(
            rollouts=rollouts,
            rewards=rewards,
            advantages=advantages,
            ref_log_probs_list=ref_log_probs_list,
            pad_token_id=tokenizer.pad_token_id if tokenizer else 0,
        )
        experience.labels = expanded_labels
        return experience

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
