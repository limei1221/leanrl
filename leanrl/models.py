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
    response_lengths: Tensor,
    max_resp_len: int,
) -> Tensor:
    """Align shifted log-probs to response tokens for variable prompt lengths.

    Uses explicit ``response_lengths`` instead of comparing token ids with
    pad_token_id, which breaks when pad_token_id == eos_token_id.

    Args:
        per_token_lp: (B, T-1), per-position values from shifted logits.
        attention_mask: (B, T), 1 for non-pad tokens.
        response_lengths: (B,) actual response lengths before padding.
        max_resp_len: output response dimension R.

    Returns:
        (B, R) values aligned to the response segment and left-aligned
        within the max response length.
    """
    batch_size = per_token_lp.shape[0]
    max_seq_minus1 = per_token_lp.shape[1]

    seq_lens = attention_mask.long().sum(dim=1)
    prompt_lens = seq_lens - response_lengths

    resp_lp = per_token_lp.new_zeros((batch_size, max_resp_len))
    for i in range(batch_size):
        resp_len_i = int(response_lengths[i].item())
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
            **({"offload_optimizer": {"device": "cpu", "pin_memory": True}}
               if infra.offload_optimizer else {}),
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                # WarmupCosineLR in this DeepSpeed version expects ratios, not
                # absolute LRs. It scales the optimizer's base LR (training.lr)
                # by these ratios over the course of training.
                "total_num_steps": total_steps,
                "warmup_min_ratio": 0.0,
                "warmup_num_steps": warmup_steps,
                "cos_min_ratio": 0.0,
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

    def forward_logprobs_from_experience(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_lengths: Tensor,
        max_resp_len: int,
    ) -> tuple[Tensor, Tensor]:
        """Compute log-probs and entropy aligned to response tokens.

        Returns:
            log_probs: (B, R) per-token log-probs for response tokens.
            entropy: (B, R) per-token entropy from the full distribution.
        """
        outputs = self.engine(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        # Shift logits: logits[:, t, :] predicts input_ids[:, t+1]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs_full = F.log_softmax(shift_logits, dim=-1)
        per_token_lp = log_probs_full.gather(
            dim=-1, index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Per-token entropy: H = -sum_v p(v) * log p(v)
        per_token_entropy = -(log_probs_full.exp() * log_probs_full).sum(dim=-1)

        resp_lp = _extract_response_logprobs(
            per_token_lp=per_token_lp,
            attention_mask=attention_mask,
            response_lengths=response_lengths,
            max_resp_len=max_resp_len,
        )
        resp_entropy = _extract_response_logprobs(
            per_token_lp=per_token_entropy,
            attention_mask=attention_mask,
            response_lengths=response_lengths,
            max_resp_len=max_resp_len,
        )
        return resp_lp, resp_entropy

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

    def offload_to_cpu(self):
        """Move model weights to CPU to free GPU memory during training."""
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def reload_to_gpu(self):
        """Move model weights back to GPU for the next rollout."""
        self.model.to(self.device)

    @torch.no_grad()
    def forward_logprobs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_lengths: Tensor,
        max_resp_len: int,
    ) -> Tensor:
        """Compute per-token log-probs for the response portion."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        response_lengths = response_lengths.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_lp = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        return _extract_response_logprobs(
            per_token_lp=per_token_lp,
            attention_mask=attention_mask,
            response_lengths=response_lengths,
            max_resp_len=max_resp_len,
        )
