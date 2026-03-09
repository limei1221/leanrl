"""Tests for the rollout engine (mock-based, no GPU required)."""

import pytest
import torch

from leanrl.experience import RolloutResult, pad_sequences, build_experience_from_rollouts


class TestPadSequences:
    def test_right_padding(self):
        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded = pad_sequences(seqs, pad_value=0, pad_side="right")
        assert padded.shape == (2, 3)
        assert padded[1, 2].item() == 0

    def test_left_padding(self):
        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        padded = pad_sequences(seqs, pad_value=0, pad_side="left")
        assert padded.shape == (2, 3)
        assert padded[1, 0].item() == 0
        assert padded[1, 1].item() == 4

    def test_single_sequence(self):
        seqs = [torch.tensor([1, 2, 3, 4])]
        padded = pad_sequences(seqs)
        assert padded.shape == (1, 4)

    def test_equal_length(self):
        seqs = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        padded = pad_sequences(seqs)
        assert padded.shape == (2, 2)
        assert (padded == torch.tensor([[1, 2], [3, 4]])).all()


class TestBuildExperience:
    def _make_rollout(self, prompt_len: int = 5, resp_len: int = 10) -> RolloutResult:
        prompt_ids = torch.arange(prompt_len)
        response_ids = torch.arange(resp_len) + prompt_len
        return RolloutResult(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            full_ids=torch.cat([prompt_ids, response_ids]),
            old_log_probs=torch.randn(resp_len),
            response_text="test response",
            prompt_text="test prompt",
            prompt_len=prompt_len,
            response_len=resp_len,
        )

    def test_basic_build(self):
        rollouts = [self._make_rollout() for _ in range(4)]
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
        ref_lps = [torch.randn(10) for _ in range(4)]

        exp = build_experience_from_rollouts(rollouts, rewards, advantages, ref_lps)
        assert exp.input_ids.shape[0] == 4
        assert exp.rewards.shape == (4,)
        assert exp.response_mask.shape[0] == 4
        assert exp.response_lengths.shape == (4,)
        assert (exp.response_lengths == 10).all()

    def test_variable_length(self):
        r1 = self._make_rollout(prompt_len=3, resp_len=5)
        r2 = self._make_rollout(prompt_len=7, resp_len=12)
        rollouts = [r1, r2]
        rewards = torch.tensor([1.0, 0.0])
        advantages = torch.tensor([0.5, -0.5])
        ref_lps = [torch.randn(5), torch.randn(12)]

        exp = build_experience_from_rollouts(rollouts, rewards, advantages, ref_lps)
        assert exp.input_ids.shape == (2, 19)  # max(3+5, 7+12) = 19
        assert exp.response_ids.shape == (2, 12)  # max(5, 12)
        assert exp.response_lengths.tolist() == [5, 12]

    def test_attention_mask_with_eos_as_pad(self):
        """attention_mask should not mask out EOS tokens when pad==eos."""
        EOS_ID = 2
        prompt_ids = torch.tensor([1, 1, 1])
        response_ids = torch.tensor([1, 1, EOS_ID])  # EOS at the end
        r = RolloutResult(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            full_ids=torch.cat([prompt_ids, response_ids]),
            old_log_probs=torch.randn(3),
            response_text="test",
            prompt_text="test",
            prompt_len=3,
            response_len=3,
        )
        exp = build_experience_from_rollouts(
            [r],
            rewards=torch.tensor([1.0]),
            advantages=torch.tensor([0.0]),
            ref_log_probs_list=[torch.randn(3)],
            pad_token_id=EOS_ID,  # pad == eos
        )
        # All 6 tokens should be unmasked (the EOS is a real token, not padding)
        assert exp.attention_mask.sum().item() == 6
