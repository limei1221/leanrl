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
    response_lengths: Tensor    # (B*G,) actual response lengths before padding
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
            response_lengths=self.response_lengths.to(device),
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


def _truncate_rollout(r: RolloutResult, ref_lp: Tensor, max_seq_len: int):
    """Truncate a rollout's response to fit within max_seq_len (prompt + response)."""
    max_resp = max_seq_len - r.prompt_len
    if max_resp <= 0:
        max_resp = 1  # keep at least 1 response token
    if r.response_len <= max_resp:
        return r, ref_lp
    return RolloutResult(
        prompt_ids=r.prompt_ids,
        response_ids=r.response_ids[:max_resp],
        full_ids=r.full_ids[:r.prompt_len + max_resp],
        old_log_probs=r.old_log_probs[:max_resp],
        response_text=r.response_text,
        prompt_text=r.prompt_text,
        prompt_len=r.prompt_len,
        response_len=max_resp,
        response_mask=r.response_mask[:max_resp] if r.response_mask is not None else None,
    ), ref_lp[:max_resp]


def build_experience_from_rollouts(
    rollouts: list[RolloutResult],
    rewards: Tensor,
    advantages: Tensor,
    ref_log_probs_list: list[Tensor],
    pad_token_id: int = 0,
    max_seq_len: int = -1,
) -> Experience:
    """Assemble individual rollout results into a batched Experience.

    Args:
        rollouts: list of B*G RolloutResult objects.
        rewards: (B*G,) rewards.
        advantages: (B*G,) GRPO advantages.
        ref_log_probs_list: list of B*G tensors, each (resp_len_i,).
        pad_token_id: token ID used for padding.
        max_seq_len: if > 0, truncate sequences longer than this.

    Returns:
        Batched Experience ready for training.
    """
    if max_seq_len > 0:
        truncated = [
            _truncate_rollout(r, lp, max_seq_len)
            for r, lp in zip(rollouts, ref_log_probs_list)
        ]
        rollouts = [t[0] for t in truncated]
        ref_log_probs_list = [t[1] for t in truncated]

    prompt_ids = pad_sequences([r.prompt_ids for r in rollouts], pad_value=pad_token_id)
    response_ids = pad_sequences([r.response_ids for r in rollouts], pad_value=pad_token_id)
    input_ids = pad_sequences([r.full_ids for r in rollouts], pad_value=pad_token_id)

    # Build attention_mask from actual sequence lengths rather than token
    # comparison, so that legitimate EOS tokens are not masked out when
    # pad_token_id == eos_token_id.
    response_lengths = torch.tensor([r.response_len for r in rollouts], dtype=torch.long)
    seq_lengths = torch.tensor(
        [r.prompt_len + r.response_len for r in rollouts], dtype=torch.long,
    )
    attention_mask = torch.zeros(input_ids.shape, dtype=torch.long)
    for i, sl in enumerate(seq_lengths):
        attention_mask[i, :sl] = 1

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
        response_lengths=response_lengths,
    )


class ExperienceBuffer:
    """Simple buffer that accumulates experiences across rollout rounds."""

    def __init__(self, max_size: int = 0):
        from collections import deque

        self._buffer: deque[Experience] = deque(maxlen=max_size if max_size > 0 else None)

    def add(self, exp: Experience):
        self._buffer.append(exp)

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
                    response_lengths=exp_dev.response_lengths[idx],
                )
