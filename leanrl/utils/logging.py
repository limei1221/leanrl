"""Logging utilities for LeanRL."""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional


def setup_logger(name: str = "leanrl", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


class MetricsTracker:
    """Accumulates and logs training metrics, optionally to WandB."""

    def __init__(self, use_wandb: bool = False, project: str = "leanrl", run_name: Optional[str] = None):
        self._use_wandb = use_wandb
        self._run = None
        if use_wandb:
            import wandb

            self._run = wandb.init(project=project, name=run_name)

    def log(self, metrics: dict[str, Any], step: int):
        logger.info(f"step={step} | " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self):
        if self._run is not None:
            self._run.finish()
