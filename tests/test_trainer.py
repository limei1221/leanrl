from types import SimpleNamespace

import pytest
import torch

from leanrl.trainer import GRPOTrainer


class _DummyExecutor:
    def __init__(self, reward_batches):
        self.reward_batches = [
            torch.tensor(batch, dtype=torch.float32) for batch in reward_batches
        ]
        self.calls = 0

    def execute(self, prompts, labels, tokenizer=None):
        rewards = self.reward_batches[self.calls]
        self.calls += 1
        return SimpleNamespace(rewards=rewards)


def _make_trainer(task: str, reward_batches: list[list[float]]) -> GRPOTrainer:
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.config = SimpleNamespace(task=task)
    trainer.eval_loader = [
        {"prompts": ["prompt"], "labels": ["label"]}
        for _ in reward_batches
    ]
    trainer.executor = _DummyExecutor(reward_batches)
    trainer.tokenizer = None
    return trainer


def test_swe_evaluate_counts_only_full_reward_as_resolved():
    # SWE rewards > 1.0 indicate test_reward=1.0 + shaping bonus
    trainer = _make_trainer("swe", [[1.1, 0.5, 0.0], [1.2]])
    assert trainer._evaluate() == pytest.approx(0.5)


def test_math_evaluate_counts_positive_reward_as_correct():
    trainer = _make_trainer("math", [[1.0, 0.5, 0.0], [1.0]])
    assert trainer._evaluate() == pytest.approx(0.75)
