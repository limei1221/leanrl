"""GRPO (Group Relative Policy Optimization) core algorithm.

Implements group-normalized advantage estimation and clipped policy loss
as described in DeepSeek-R1 and OpenRLHF.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_grpo_advantages(
    rewards: Tensor,
    n_samples_per_prompt: int,
    eps: float = 1e-8,
) -> Tensor:
    """Compute group-normalized advantages for GRPO.

    For each prompt group of G samples, advantage = (r - mean) / (std + eps).
    When all rewards in a group are identical (std=0), advantages are zero.

    Args:
        rewards: (B*G,) flat reward tensor.
        n_samples_per_prompt: G, number of samples per prompt.
        eps: small constant for numerical stability.

    Returns:
        advantages: (B*G,) group-normalized advantages.
    """
    G = n_samples_per_prompt
    grouped = rewards.view(-1, G)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


def compute_kl_penalty(
    log_probs: Tensor,
    ref_log_probs: Tensor,
    mask: Tensor,
) -> Tensor:
    """Compute per-sequence KL divergence KL(pi || pi_ref).

    Uses the approximation: KL = mean over tokens of (log_pi - log_pi_ref).

    Args:
        log_probs: (B, T) per-token log probs from current policy.
        ref_log_probs: (B, T) per-token log probs from reference policy.
        mask: (B, T) binary mask for valid response tokens.

    Returns:
        kl: scalar mean KL divergence.
    """
    per_token_kl = log_probs - ref_log_probs
    masked_kl = (per_token_kl * mask).sum() / mask.sum().clamp(min=1)
    return masked_kl


def grpo_policy_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    clip_range: float = 0.2,
) -> Tensor:
    """Clipped surrogate policy loss for GRPO.

    Same as PPO's clipped objective but uses group-normalized advantages
    instead of critic-based advantages.

    Args:
        log_probs: (B, T) per-token log probs under current policy.
        old_log_probs: (B, T) per-token log probs from rollout policy.
        advantages: (B,) per-sequence advantage from GRPO.
        mask: (B, T) binary mask, 1 for response tokens, 0 for prompt/padding.
        clip_range: PPO clipping epsilon.

    Returns:
        loss: scalar policy loss (to be minimized).
    """
    # Per-token importance ratio
    ratio = (log_probs - old_log_probs).exp()
    clipped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range)

    # Broadcast advantages to token level
    adv = advantages.unsqueeze(-1)  # (B, 1)

    # Pessimistic bound
    surr1 = ratio * adv
    surr2 = clipped * adv
    per_token_loss = -torch.min(surr1, surr2)

    # Masked mean over response tokens
    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
    return loss


def grpo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    ref_log_probs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    clip_range: float = 0.2,
    kl_coef: float = 0.01,
    entropy_coef: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Combined GRPO loss: policy loss + KL penalty + entropy bonus.

    Returns both the scalar loss and a dict of metrics for logging.
    """
    policy_loss = grpo_policy_loss(log_probs, old_log_probs, advantages, mask, clip_range)
    kl = compute_kl_penalty(log_probs, ref_log_probs, mask)

    loss = policy_loss + kl_coef * kl

    # Optional entropy bonus (encourages exploration)
    if entropy_coef > 0:
        entropy = -(log_probs.exp() * log_probs * mask).sum() / mask.sum().clamp(min=1)
        loss = loss - entropy_coef * entropy
    else:
        entropy = torch.tensor(0.0, device=loss.device)

    # Approximate clip fraction for monitoring
    with torch.no_grad():
        ratio = (log_probs - old_log_probs).exp()
        clip_frac = ((ratio - 1.0).abs() > clip_range).float()
        clip_frac = (clip_frac * mask).sum() / mask.sum().clamp(min=1)

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl": kl.item(),
        "entropy": entropy.item(),
        "clip_fraction": clip_frac.item(),
        "total_loss": loss.item(),
    }
    return loss, metrics
