"""Configuration dataclasses for LeanRL training."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ref_model_name_or_path: Optional[str] = None
    dtype: str = "bf16"
    trust_remote_code: bool = True

    def __post_init__(self):
        if self.ref_model_name_or_path is None:
            self.ref_model_name_or_path = self.model_name_or_path


@dataclass
class GRPOConfig:
    n_samples_per_prompt: int = 8
    kl_coef: float = 0.01
    clip_range: float = 0.2
    max_grad_norm: float = 1.0
    entropy_coef: float = 0.0
    advantage_eps: float = 1e-8


@dataclass
class RolloutConfig:
    rollout_batch_size: int = 64
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1


@dataclass
class TrainingConfig:
    lr: float = 5e-7
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    train_batch_size: int = 8
    micro_batch_size: int = 2
    num_epochs: int = 1
    max_steps: int = -1
    save_steps: int = 50
    logging_steps: int = 1
    gradient_checkpointing: bool = True
    num_ppo_epochs: int = 1
    save_best_only: bool = False
    seed: int = 42


@dataclass
class SWEConfig:
    max_turns: int = 10
    docker_image_prefix: str = "swebench"
    sandbox_timeout: int = 300
    max_concurrent_sandboxes: int = 4
    sandbox_memory_limit: str = "4g"
    sandbox_cpu_limit: float = 2.0


@dataclass
class InfraConfig:
    num_gpus: int = 2
    vllm_num_engines: int = 1
    vllm_gpu_memory_utilization: float = 0.85
    vllm_enforce_eager: bool = True
    vllm_enable_sleep: bool = True
    vllm_tensor_parallel_size: int = 1
    deepspeed_stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False
    ray_address: Optional[str] = None


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "leanrl"
    wandb_run_name: Optional[str] = None
    output_dir: str = "./output"


@dataclass
class DataConfig:
    prompt_dataset: str = "openai/gsm8k"
    prompt_dataset_split: str = "train"
    input_key: str = "question"
    label_key: str = "answer"
    max_samples: int = -1
    eval_dataset: Optional[str] = None
    eval_split: str = "eval"
    max_samples_to_eval: int = 128


@dataclass
class TrainConfig:
    """Top-level configuration that composes all sub-configs."""

    task: str = "math"  # "math" or "swe"
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    swe: SWEConfig = field(default_factory=SWEConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> TrainConfig:
        return cls(
            task=d.get("task", "math"),
            model=ModelConfig(**d.get("model", {})),
            grpo=GRPOConfig(**d.get("grpo", {})),
            rollout=RolloutConfig(**d.get("rollout", {})),
            training=TrainingConfig(**d.get("training", {})),
            swe=SWEConfig(**d.get("swe", {})),
            infra=InfraConfig(**d.get("infra", {})),
            logging=LoggingConfig(**d.get("logging", {})),
            data=DataConfig(**d.get("data", {})),
        )
