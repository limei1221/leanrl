"""Experience buffer for collecting rollout data and building training batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class RolloutResult:
    """Output of a single rollout sample (one completion for one prompt)."""

    prompt_ids: Tensor       # (prompt_len,)
    response_ids: Tensor     # (resp_len,)
    full_ids: Tensor         # (prompt_len + resp_len,)
    old_log_probs: Tensor    # (resp_len,) per-token log probs from rollout policy
    response_text: str
    prompt_text: str
    prompt_len: int
    response_len: int
    response_mask: Optional[Tensor] = None


@dataclass
class Experience:
    """Batched training experience with GRPO advantages computed.

    All tensors have batch dimension B*G where B is the number of unique
    prompts and G is n_samples_per_prompt.
    """

    prompt_ids: Tensor          # (B*G, max_prompt_len)
    response_ids: Tensor        # (B*G, max_resp_len)
    input_ids: Tensor           # (B*G, max_seq_len) = prompt + response
    attention_mask: Tensor      # (B*G, max_seq_len)
    old_log_probs: Tensor       # (B*G, max_resp_len)
    ref_log_probs: Tensor       # (B*G, max_resp_len)
    rewards: Tensor             # (B*G,)
    advantages: Tensor          # (B*G,)
    response_mask: Tensor       # (B*G, max_resp_len) 1=model token, 0=pad/env
    labels: Optional[list[str]] = None

    def __len__(self) -> int:
        return self.rewards.shape[0]

    def to(self, device: torch.device) -> Experience:
        return Experience(
            prompt_ids=self.prompt_ids.to(device),
            response_ids=self.response_ids.to(device),
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            ref_log_probs=self.ref_log_probs.to(device),
            rewards=self.rewards.to(device),
            advantages=self.advantages.to(device),
            response_mask=self.response_mask.to(device),
            labels=self.labels,
        )


def pad_sequences(sequences: list[Tensor], pad_value: int = 0, pad_side: str = "right") -> Tensor:
    """Pad a list of 1-D tensors to the same length.

    Args:
        sequences: list of (L_i,) tensors.
        pad_value: value to use for padding.
        pad_side: "right" or "left".

    Returns:
        padded: (N, max_L) tensor.
    """
    max_len = max(s.shape[0] for s in sequences)
    padded = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
    for i, s in enumerate(sequences):
        if pad_side == "right":
            padded[i, : s.shape[0]] = s
        else:
            padded[i, max_len - s.shape[0] :] = s
    return padded


def build_experience_from_rollouts(
    rollouts: list[RolloutResult],
    rewards: Tensor,
    advantages: Tensor,
    ref_log_probs_list: list[Tensor],
    pad_token_id: int = 0,
) -> Experience:
    """Assemble individual rollout results into a batched Experience.

    Args:
        rollouts: list of B*G RolloutResult objects.
        rewards: (B*G,) rewards.
        advantages: (B*G,) GRPO advantages.
        ref_log_probs_list: list of B*G tensors, each (resp_len_i,).
        pad_token_id: token ID used for padding.

    Returns:
        Batched Experience ready for training.
    """
    prompt_ids = pad_sequences([r.prompt_ids for r in rollouts], pad_value=pad_token_id)
    response_ids = pad_sequences([r.response_ids for r in rollouts], pad_value=pad_token_id)
    input_ids = pad_sequences([r.full_ids for r in rollouts], pad_value=pad_token_id)

    max_seq_len = input_ids.shape[1]
    attention_mask = (input_ids != pad_token_id).long()

    max_resp_len = response_ids.shape[1]
    old_log_probs = pad_sequences([r.old_log_probs for r in rollouts], pad_value=0.0)
    ref_log_probs = pad_sequences(ref_log_probs_list, pad_value=0.0)

    response_masks = []
    for r in rollouts:
        if r.response_mask is not None:
            response_masks.append(r.response_mask.to(dtype=torch.float32))
        else:
            response_masks.append(torch.ones(r.response_len, dtype=torch.float32))
    response_mask = pad_sequences(response_masks, pad_value=0.0)

    return Experience(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        rewards=rewards,
        advantages=advantages,
        response_mask=response_mask,
    )


class ExperienceBuffer:
    """Simple buffer that accumulates experiences across rollout rounds."""

    def __init__(self, max_size: int = 0):
        self._buffer: list[Experience] = []
        self._max_size = max_size

    def add(self, exp: Experience):
        self._buffer.append(exp)
        if self._max_size > 0 and len(self._buffer) > self._max_size:
            self._buffer.pop(0)

    def clear(self):
        self._buffer.clear()

    @property
    def size(self) -> int:
        return sum(len(e) for e in self._buffer)

    def iterate_minibatches(self, micro_batch_size: int, device: torch.device):
        """Yield micro-batches by slicing each stored Experience."""
        for exp in self._buffer:
            exp_dev = exp.to(device)
            n = len(exp_dev)
            indices = torch.randperm(n)
            for start in range(0, n, micro_batch_size):
                idx = indices[start : start + micro_batch_size]
                yield Experience(
                    prompt_ids=exp_dev.prompt_ids[idx],
                    response_ids=exp_dev.response_ids[idx],
                    input_ids=exp_dev.input_ids[idx],
                    attention_mask=exp_dev.attention_mask[idx],
                    old_log_probs=exp_dev.old_log_probs[idx],
                    ref_log_probs=exp_dev.ref_log_probs[idx],
                    rewards=exp_dev.rewards[idx],
                    advantages=exp_dev.advantages[idx],
                    response_mask=exp_dev.response_mask[idx],
                )
