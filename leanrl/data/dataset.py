"""Prompt dataset loading for LeanRL training."""

from __future__ import annotations

import json
import random
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


def build_train_and_eval_dataloaders(
    config: DataConfig,
    batch_size: int,
    tokenizer=None,
    seed: int = 42,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Build training and eval dataloaders.

    If an eval split exists (via eval_dataset or the dataset's eval_split),
    use it (capped at max_samples_to_eval). Otherwise, randomly sample
    max_samples_to_eval from training data for eval and use the rest for training.
    """
    max_eval = config.max_samples_to_eval
    if max_eval <= 0:
        # No eval requested
        train_loader = build_prompt_dataloader(config, batch_size, tokenizer)
        return train_loader, None

    # Try to load a separate eval split
    eval_ds = None
    eval_dataset_name = config.eval_dataset or config.prompt_dataset
    eval_split = config.eval_split
    try:
        if eval_dataset_name == "openai/gsm8k":
            eval_ds = load_dataset("openai/gsm8k", "main", split=eval_split)
        else:
            eval_ds = load_dataset(eval_dataset_name, split=eval_split)
        logger.info(f"Found eval split: {eval_dataset_name} split={eval_split} ({len(eval_ds)} samples)")
    except Exception:
        eval_ds = None
        logger.info(f"No eval split '{eval_split}' found for {eval_dataset_name}, will split from training data")

    if eval_ds is not None:
        # Use eval split, cap at max_samples_to_eval
        if len(eval_ds) > max_eval:
            rng = random.Random(seed)
            indices = rng.sample(range(len(eval_ds)), max_eval)
            eval_ds = eval_ds.select(sorted(indices))
        eval_dataset = PromptDataset.__new__(PromptDataset)
        eval_dataset.prompts = [item[config.input_key] for item in eval_ds]
        eval_dataset.labels = []
        eval_dataset.tokenizer = tokenizer
        for item in eval_ds:
            if config.label_key and config.label_key in item:
                if all(f in item for f in _SWE_MARKER_FIELDS):
                    eval_dataset.labels.append(json.dumps({
                        "instance_id": item["instance_id"],
                        "repo": item["repo"],
                        "base_commit": item["base_commit"],
                        "test_patch": item["test_patch"],
                        "fail_to_pass": json.loads(item["FAIL_TO_PASS"]) if isinstance(item["FAIL_TO_PASS"], str) else item["FAIL_TO_PASS"],
                        "pass_to_pass": json.loads(item["PASS_TO_PASS"]) if isinstance(item["PASS_TO_PASS"], str) else item["PASS_TO_PASS"],
                    }))
                else:
                    eval_dataset.labels.append(item[config.label_key])
            else:
                eval_dataset.labels.append(None)
        logger.info(f"Eval set: {len(eval_dataset)} samples")

        # Training data is unchanged
        train_loader = build_prompt_dataloader(config, batch_size, tokenizer)
    else:
        # Split training data: sample max_eval for eval, rest for training
        train_dataset = PromptDataset(config, tokenizer=tokenizer)
        n = len(train_dataset)
        max_eval = min(max_eval, n)

        rng = random.Random(seed)
        all_indices = list(range(n))
        rng.shuffle(all_indices)
        eval_indices = sorted(all_indices[:max_eval])
        train_indices = sorted(all_indices[max_eval:])

        eval_dataset = PromptDataset.__new__(PromptDataset)
        eval_dataset.prompts = [train_dataset.prompts[i] for i in eval_indices]
        eval_dataset.labels = [train_dataset.labels[i] for i in eval_indices]
        eval_dataset.tokenizer = tokenizer

        train_subset = PromptDataset.__new__(PromptDataset)
        train_subset.prompts = [train_dataset.prompts[i] for i in train_indices]
        train_subset.labels = [train_dataset.labels[i] for i in train_indices]
        train_subset.tokenizer = tokenizer

        logger.info(f"Split training data: {len(train_subset)} train, {len(eval_dataset)} eval")

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=_collate_prompts,
        )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_prompts,
    )

    return train_loader, eval_loader


def _collate_prompts(batch: list[dict]) -> dict:
    return {
        "prompts": [b["prompt"] for b in batch],
        "labels": [b["label"] for b in batch],
    }
