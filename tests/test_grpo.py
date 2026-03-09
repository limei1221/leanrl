"""Unit tests for GRPO advantage computation and policy loss."""

import torch
import pytest
from leanrl.grpo import compute_grpo_advantages, grpo_policy_loss, grpo_loss, compute_kl_penalty


class TestComputeGRPOAdvantages:
    def test_basic_normalization(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        adv = compute_grpo_advantages(rewards, n_samples_per_prompt=2)
        assert adv.shape == (4,)
        # Within each group of 2, [1,0] -> positive, negative
        assert adv[0] > 0
        assert adv[1] < 0
        assert adv[2] > 0
        assert adv[3] < 0

    def test_zero_std_group(self):
        """When all rewards in a group are the same, advantages should be ~0."""
        rewards = torch.tensor([1.0, 1.0, 0.0, 1.0])
        adv = compute_grpo_advantages(rewards, n_samples_per_prompt=2)
        assert abs(adv[0].item()) < 1e-4
        assert abs(adv[1].item()) < 1e-4

    def test_shapes_with_larger_groups(self):
        B, G = 4, 8
        rewards = torch.randn(B * G)
        adv = compute_grpo_advantages(rewards, n_samples_per_prompt=G)
        assert adv.shape == (B * G,)

    def test_per_group_zero_mean(self):
        """Advantages within each group should sum to approximately zero."""
        B, G = 3, 4
        rewards = torch.randn(B * G)
        adv = compute_grpo_advantages(rewards, n_samples_per_prompt=G)
        grouped = adv.view(B, G)
        group_sums = grouped.sum(dim=1)
        for s in group_sums:
            assert abs(s.item()) < 1e-4


class TestGRPOPolicyLoss:
    def test_zero_advantage_zero_loss(self):
        B, T = 4, 10
        log_probs = torch.randn(B, T)
        old_log_probs = log_probs.clone()
        advantages = torch.zeros(B)
        mask = torch.ones(B, T)
        loss = grpo_policy_loss(log_probs, old_log_probs, advantages, mask)
        assert abs(loss.item()) < 1e-6

    def test_positive_advantage_encourages(self):
        B, T = 2, 5
        old_lp = torch.zeros(B, T)
        cur_lp = torch.zeros(B, T)
        mask = torch.ones(B, T)

        # Positive advantage should yield negative loss (reward signal)
        adv = torch.tensor([1.0, 1.0])
        loss_pos = grpo_policy_loss(cur_lp, old_lp, adv, mask)
        adv_neg = torch.tensor([-1.0, -1.0])
        loss_neg = grpo_policy_loss(cur_lp, old_lp, adv_neg, mask)
        # With ratio=1, loss = -advantage, so positive adv -> more negative loss
        assert loss_pos < loss_neg

    def test_mask_excludes_padding(self):
        B, T = 2, 10
        log_probs = torch.randn(B, T)
        old_log_probs = log_probs.clone()
        advantages = torch.ones(B)
        mask = torch.zeros(B, T)
        mask[:, :5] = 1.0  # only first 5 tokens count

        loss = grpo_policy_loss(log_probs, old_log_probs, advantages, mask)
        assert loss.isfinite()

    def test_clipping_limits_ratio(self):
        B, T = 2, 5
        old_lp = torch.zeros(B, T)
        # Big shift -> ratio far from 1
        cur_lp = torch.ones(B, T) * 2.0
        mask = torch.ones(B, T)
        advantages = torch.ones(B)

        loss_clipped = grpo_policy_loss(cur_lp, old_lp, advantages, mask, clip_range=0.2)
        loss_unclipped = grpo_policy_loss(cur_lp, old_lp, advantages, mask, clip_range=100.0)
        # Clipping should reduce the magnitude of the loss
        assert abs(loss_clipped.item()) <= abs(loss_unclipped.item()) + 1e-6


class TestKLPenalty:
    def test_identical_distributions_zero_kl(self):
        lp = torch.randn(4, 10)
        mask = torch.ones(4, 10)
        kl = compute_kl_penalty(lp, lp, mask)
        assert abs(kl.item()) < 1e-6

    def test_kl_positive_when_different(self):
        lp = torch.zeros(4, 10)
        ref_lp = -torch.ones(4, 10)
        mask = torch.ones(4, 10)
        kl = compute_kl_penalty(lp, ref_lp, mask)
        assert kl.item() > 0

    def test_kl_positive_when_flipped(self):
        lp = -torch.ones(4, 10)
        ref_lp = torch.zeros(4, 10)
        mask = torch.ones(4, 10)
        kl = compute_kl_penalty(lp, ref_lp, mask)
        assert kl.item() > 0


class TestCombinedLoss:
    def test_combined_loss_returns_metrics(self):
        B, T = 4, 10
        lp = torch.randn(B, T, requires_grad=True)
        old_lp = lp.detach()
        ref_lp = torch.randn(B, T)
        adv = torch.randn(B)
        mask = torch.ones(B, T)

        loss, metrics = grpo_loss(lp, old_lp, ref_lp, adv, mask)
        assert loss.isfinite()
        assert "policy_loss" in metrics
        assert "kl" in metrics
        assert "clip_fraction" in metrics
        assert "total_loss" in metrics
        loss.backward()
        assert lp.grad is not None
