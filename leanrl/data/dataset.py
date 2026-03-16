"""Prompt dataset loading for LeanRL training."""

from __future__ import annotations

import json
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from leanrl.utils.config import DataConfig
from leanrl.utils.logging import logger

# Marker fields that identify a SWE-bench dataset row (princeton-nlp/SWE-bench_Lite uses
# uppercase FAIL_TO_PASS / PASS_TO_PASS stored as JSON-encoded strings)
_SWE_MARKER_FIELDS = ("instance_id", "repo", "base_commit", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")


class PromptDataset(Dataset):
    """Loads prompts (and optional labels) from a HuggingFace dataset."""

    def __init__(self, config: DataConfig, split: Optional[str] = None, tokenizer=None):
        split = split or config.prompt_dataset_split
        logger.info(f"Loading dataset: {config.prompt_dataset} split={split}")

        if config.prompt_dataset == "openai/gsm8k":
            ds = load_dataset("openai/gsm8k", "main", split=split)
        else:
            ds = load_dataset(config.prompt_dataset, split=split)

        if config.max_samples > 0:
            ds = ds.select(range(min(config.max_samples, len(ds))))

        self.prompts = []
        self.labels = []
        self.tokenizer = tokenizer

        for item in ds:
            prompt = item[config.input_key]
            self.prompts.append(prompt)

            if config.label_key and config.label_key in item:
                # For SWE-bench rows, serialize full task metadata as JSON so
                # MultiTurnExecutor._parse_tasks can access repo, base_commit, etc.
                # princeton-nlp/SWE-bench_Lite uses uppercase FAIL_TO_PASS/PASS_TO_PASS
                # stored as JSON-encoded strings; normalize to lowercase lists.
                if all(f in item for f in _SWE_MARKER_FIELDS):
                    self.labels.append(json.dumps({
                        "instance_id": item["instance_id"],
                        "repo": item["repo"],
                        "base_commit": item["base_commit"],
                        "test_patch": item["test_patch"],
                        "fail_to_pass": json.loads(item["FAIL_TO_PASS"]) if isinstance(item["FAIL_TO_PASS"], str) else item["FAIL_TO_PASS"],
                        "pass_to_pass": json.loads(item["PASS_TO_PASS"]) if isinstance(item["PASS_TO_PASS"], str) else item["PASS_TO_PASS"],
                    }))
                else:
                    self.labels.append(item[config.label_key])
            else:
                self.labels.append(None)

        logger.info(f"Loaded {len(self.prompts)} prompts")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        item = {"prompt": self.prompts[idx], "label": self.labels[idx]}
        return item


def build_prompt_dataloader(
    config: DataConfig,
    batch_size: int,
    tokenizer=None,
    split: Optional[str] = None,
) -> DataLoader:
    dataset = PromptDataset(config, split=split, tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=_collate_prompts,
    )


def _collate_prompts(batch: list[dict]) -> dict:
    return {
        "prompts": [b["prompt"] for b in batch],
        "labels": [b["label"] for b in batch],
    }
