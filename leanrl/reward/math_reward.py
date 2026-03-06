"""Rule-based math verification reward function.

Supports:
- GSM8K-style answers with #### delimiter
- LaTeX boxed answers
- Numeric comparison with tolerance
"""

from __future__ import annotations

import re
from typing import Optional

import torch
from torch import Tensor


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from a GSM8K-style response.

    GSM8K gold answers use '#### <number>' format.
    Model responses may use various formats.
    """
    # Try #### format first (gold labels)
    match = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    if match:
        return _clean_number(match.group(1).strip())

    # Try \\boxed{...} format
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return _clean_number(match.group(1).strip())

    # Try "the answer is X" pattern
    match = re.search(r"(?:the\s+)?answer\s+is\s*[:\s]*([+-]?[\d,]+\.?\d*)", text, re.IGNORECASE)
    if match:
        return _clean_number(match.group(1).strip())

    # Fall back to last number in the text
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if numbers:
        return _clean_number(numbers[-1])

    return None


def _clean_number(s: str) -> str:
    """Remove commas and whitespace from a numeric string."""
    return s.replace(",", "").replace(" ", "").strip()


def numbers_equal(a: str, b: str, tol: float = 1e-5) -> bool:
    """Compare two numeric strings with tolerance."""
    try:
        return abs(float(a) - float(b)) < tol
    except (ValueError, TypeError):
        return a.strip() == b.strip()


def compute_math_rewards(
    responses: list[str],
    labels: list[str],
) -> Tensor:
    """Score a batch of model responses against gold labels.

    Args:
        responses: list of model-generated response strings.
        labels: list of gold label strings (may contain ####).

    Returns:
        rewards: (B,) tensor of 0.0 or 1.0.
    """
    rewards = []
    for resp, label in zip(responses, labels):
        pred = extract_gsm8k_answer(resp)
        gold = extract_gsm8k_answer(label) if label else None

        if pred is not None and gold is not None:
            rewards.append(1.0 if numbers_equal(pred, gold) else 0.0)
        else:
            rewards.append(0.0)

    return torch.tensor(rewards, dtype=torch.float32)
