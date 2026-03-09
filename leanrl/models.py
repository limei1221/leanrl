"""Policy and Reference model wrappers with DeepSpeed ZeRO integration."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from leanrl.utils.config import ModelConfig, TrainingConfig, InfraConfig
from leanrl.utils.logging import logger


def _extract_response_logprobs(
    per_token_lp: Tensor,
    attention_mask: Tensor,
    response_ids: Tensor,
    response_mask: Tensor,
    pad_token_id: int,
) -> Tensor:
    """Align shifted log-probs to response tokens for variable prompt lengths.

    Args:
        per_token_lp: (B, T-1), log-prob of input_ids[:, 1:].
        attention_mask: (B, T), 1 for non-pad tokens.
        response_ids: (B, R), padded response tokens (model + observations).
        response_mask: (B, R), 1 for model tokens to train on, 0 otherwise.
        pad_token_id: tokenizer pad token id used for padding.

    Returns:
        (B, R) response log-probs, aligned to the response segment and
        left-aligned within the max response length.
    """
    batch_size = per_token_lp.shape[0]
    max_resp_len = response_mask.shape[1]
    max_seq_minus1 = per_token_lp.shape[1]

    seq_lens = attention_mask.long().sum(dim=1)
    resp_lens = (response_ids != pad_token_id).long().sum(dim=1)
    prompt_lens = seq_lens - resp_lens

    resp_lp = per_token_lp.new_zeros((batch_size, max_resp_len))
    for i in range(batch_size):
        resp_len_i = int(resp_lens[i].item())
        if resp_len_i <= 0:
            continue

        start = max(int(prompt_lens[i].item()) - 1, 0)
        end = min(start + resp_len_i, max_seq_minus1)
        width = end - start
        if width <= 0:
            continue

        resp_lp[i, :width] = per_token_lp[i, start:end]
    return resp_lp


def get_deepspeed_config(
    infra: InfraConfig,
    training: TrainingConfig,
    total_steps: int,
) -> dict:
    """Build a DeepSpeed config dict for ZeRO-2 or ZeRO-3."""
    warmup_steps = int(total_steps * training.warmup_ratio)
    config = {
        "train_micro_batch_size_per_gpu": training.micro_batch_size,
        "gradient_accumulation_steps": max(1, training.train_batch_size // training.micro_batch_size),
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": infra.deepspeed_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training.lr,
                "warmup_num_steps": warmup_steps,
                "total_num_steps": total_steps,
            },
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training.lr,
                "betas": [0.9, 0.95],
                "weight_decay": training.weight_decay,
            },
        },
    }
    if infra.deepspeed_stage == 3:
        config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
        })
    return config


class PolicyModel:
    """Wraps a HuggingFace causal LM with DeepSpeed for training."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        infra_cfg: InfraConfig,
        total_steps: int,
    ):
        dtype = (
            torch.bfloat16 if model_cfg.dtype == "bf16"
            else torch.float16 if model_cfg.dtype == "fp16"
            else torch.float32
        )
        logger.info(f"Loading policy model: {model_cfg.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=model_cfg.trust_remote_code,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.model_name_or_path,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if training_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        ds_config = get_deepspeed_config(infra_cfg, training_cfg, total_steps)
        self._init_deepspeed(ds_config)

    def _init_deepspeed(self, ds_config: dict):
        import deepspeed

        self.engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config,
        )

    def forward_logprobs(self, input_ids: Tensor, attention_mask: Tensor, response_start: int) -> Tensor:
        """Compute per-token log-probabilities for the response portion.

        Args:
            input_ids: (B, T) full sequence (prompt + response).
            attention_mask: (B, T).
            response_start: index where the response begins.

        Returns:
            log_probs: (B, T - response_start) per-token log probs for response.
        """
        with torch.no_grad() if not self.model.training else torch.enable_grad():
            outputs = self.engine(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, T, V)

        # Shift: logits[t] predicts token[t+1]
        resp_logits = logits[:, response_start - 1 : -1, :]  # (B, resp_len, V)
        resp_tokens = input_ids[:, response_start:]            # (B, resp_len)

        log_probs = F.log_softmax(resp_logits, dim=-1)
        per_token_lp = log_probs.gather(dim=-1, index=resp_tokens.unsqueeze(-1)).squeeze(-1)
        return per_token_lp

    def forward_logprobs_from_experience(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_ids: Tensor,
        response_mask: Tensor,
        pad_token_id: int,
    ) -> Tensor:
        """Compute log-probs aligned with response_ids using the response_mask.

        Handles variable-length prompts by using the full input_ids and
        extracting log-probs only where response_mask=1.
        """
        outputs = self.engine(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        # Shift logits: logits[:, t, :] predicts input_ids[:, t+1]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_lp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        return _extract_response_logprobs(
            per_token_lp=per_token_lp,
            attention_mask=attention_mask,
            response_ids=response_ids,
            response_mask=response_mask,
            pad_token_id=pad_token_id,
        )

    def train_step(self, loss: Tensor) -> dict:
        """Run a single backward + optimizer step via DeepSpeed."""
        self.engine.backward(loss)
        self.engine.step()
        return {"lr": self.scheduler.get_last_lr()[0] if self.scheduler else 0.0}

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.engine.save_checkpoint(str(path))
        self.tokenizer.save_pretrained(str(path))
        logger.info(f"Policy model saved to {path}")

    def save_hf(self, path: str | Path):
        """Save as a standard HuggingFace model for inference."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state_dict = self.engine.module.state_dict()
        self.model.save_pretrained(str(path), state_dict=state_dict)
        self.tokenizer.save_pretrained(str(path))
        logger.info(f"HF model saved to {path}")

    def get_state_dict_for_vllm(self) -> dict:
        """Extract state dict for syncing to vLLM engine."""
        return {k: v.cpu() for k, v in self.engine.module.state_dict().items()}


class ReferenceModel:
    """Frozen reference model for KL computation. No optimizer, eval-only."""

    def __init__(self, model_cfg: ModelConfig, device: torch.device):
        dtype = (
            torch.bfloat16 if model_cfg.dtype == "bf16"
            else torch.float16 if model_cfg.dtype == "fp16"
            else torch.float32
        )
        logger.info(f"Loading reference model: {model_cfg.ref_model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.ref_model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=model_cfg.trust_remote_code,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def forward_logprobs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_ids: Tensor,
        response_mask: Optional[Tensor] = None,
        pad_token_id: int = 0,
    ) -> Tensor:
        """Compute per-token log-probs for the response portion."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if response_mask is None:
            # Backward-compatible fallback for callers that do not pass a mask.
            response_mask = (response_ids != 0).float()
        response_mask = response_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_lp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        return _extract_response_logprobs(
            per_token_lp=per_token_lp,
            attention_mask=attention_mask,
            response_ids=response_ids,
            response_mask=response_mask,
            pad_token_id=pad_token_id,
        )
