"""Prompt dataset loading for LeanRL training."""

from __future__ import annotations

from typing import Optional
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from leanrl.utils.config import DataConfig
from leanrl.utils.logging import logger


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
