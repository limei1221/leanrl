# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LeanRL is a minimal GRPO (Group Relative Policy Optimization) RL post-training framework for LLMs. It trains models on two task types:
- **Math** (GSM8K): single-turn, reward based on numeric answer correctness
- **SWE-bench**: multi-turn, reward based on test suite pass/fail in Docker sandboxes

Key technologies: Ray (distributed rollouts), vLLM (fast inference), DeepSpeed (memory-efficient training), Docker (sandboxed code execution).

## Commands

### Install
```bash
pip install -e ".[dev]"
# DeepSpeed MPI requirement:
apt-get install -y libopenmpi-dev openmpi-bin
# If CUDA tooling is missing:
apt-get install -y nvidia-cuda-toolkit
```

### Lint & Format
```bash
ruff check leanrl/
ruff check --fix leanrl/  # auto-fix
ruff format leanrl/       # auto-format
```

Style: line-length 100, target Python 3.10 (configured in pyproject.toml).

### Test
```bash
pytest tests/
pytest tests/test_grpo.py          # single file
pytest tests/test_grpo.py::test_fn # single test
```

All tests are mock-based and run without a GPU.

### Train (also available as `leanrl-train` CLI)
```bash
# Math (GSM8K) — requires Ray on GPU 1, training on GPU 0
CUDA_VISIBLE_DEVICES=1 ray start --head --num-gpus=1
# Ensure config has infra.ray_address: "auto" so trainer connects to external Ray cluster
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m leanrl.trainer --config configs/math_grpo_1.5b.yaml
# Or via script:
bash scripts/train_math.sh

# SWE-bench — setup Docker images first
bash scripts/setup_swe_docker.sh
bash scripts/train_swe.sh
```

### Evaluate
```bash
python scripts/eval_math.py --model_name_or_path <checkpoint>/final --batch_size 128
python scripts/eval_swe.py --model_name_or_path <checkpoint>/final --num_gpus 1
python scripts/eval_swe_oracle.py  # golden patch baseline (resolve_rate: 0.86)
```

## Architecture

### Training Loop (`leanrl/trainer.py` — `GRPOTrainer`)

The main loop alternates between three phases per batch:

1. **Rollout phase** — vLLM active on GPU 1, training GPU idle. Generates G samples per prompt via `RolloutEngine`, scores them with the reward function, computes GRPO advantages (group-normalized: `(r - mean) / (std + eps)`).

2. **Training phase** — vLLM sleeps to release GPU memory, reference model offloaded to CPU. Runs PPO epochs: computes log-probs from policy, GRPO loss (clipped surrogate + KL penalty + optional entropy bonus), then optimizer step.

3. **Sync phase** — extracts state dict from DeepSpeed policy, merges Q/K/V → `qkv_proj` and gate/up → `gate_up_proj` for vLLM compatibility, then pushes weights to the Ray actor via `collective_rpc`.

### Key Modules

| File | Role |
|------|------|
| `leanrl/trainer.py` | Orchestrator: Ray init, model setup, data loading, training loop, checkpointing |
| `leanrl/grpo.py` | GRPO loss: `compute_grpo_advantages`, `compute_kl_penalty` (k3 estimator), `grpo_loss` |
| `leanrl/models.py` | `PolicyModel` (DeepSpeed ZeRO-2/3), `ReferenceModel` (frozen, CPU-offloadable), weight sync logic |
| `leanrl/rollout.py` | `RolloutEngine` Ray actor wrapping vLLM; `WeightUpdateExtension` for in-place weight sync |
| `leanrl/experience.py` | `Experience` dataclass (batched rollouts), `build_experience_from_rollouts` |
| `leanrl/agent/single_turn.py` | Math executor: formats prompts, generates completions, computes reference log-probs |
| `leanrl/agent/multi_turn.py` | SWE-bench executor: multi-turn agent loop with `parse_action` (XML tags or markdown fences) |
| `leanrl/agent/sandbox.py` | `DockerSandbox` (single container), `SandboxPool` (concurrent), SWE-bench task management |
| `leanrl/reward/math_reward.py` | `extract_gsm8k_answer` (tries `####`, `\boxed{}`, last number), `compute_math_rewards` |
| `leanrl/reward/swe_reward.py` | Runs fail_to_pass/pass_to_pass tests, parses pytest/unittest output |
| `leanrl/data/dataset.py` | `PromptDataset` for GSM8K and SWE-bench_Lite; `build_prompt_dataloader` |
| `leanrl/utils/config.py` | Config dataclasses loaded from YAML (`TrainConfig` → `ModelConfig`, `GRPOConfig`, etc.) |

### GPU Memory Layout (2×GPU setup)

- **GPU 0**: Policy model + Reference model. During training, reference model is offloaded to CPU.
- **GPU 1**: vLLM `RolloutEngine` as Ray actor. During training, enters sleep mode to release VRAM.

Weight sync uses `torch.save`-serialized bytes over `collective_rpc` to survive msgpack serialization limits.

### Async Rollout Prefetching

When `training.async_prefetch: true` and `infra.vllm_enable_sleep: false` (dedicated GPUs), the trainer overlaps the next batch's vLLM generation on GPU 1 with training on GPU 0. Prefetched rollouts use weights one step stale, handled by importance sampling. Only available for single-turn (math) tasks.

### SWE-bench Agent Actions

`parse_action()` in `multi_turn.py` accepts two formats:
- XML: `<bash>cmd</bash>`, `<edit path="file">content</edit>`, `<done/>`
- Markdown fences: ` ```bash `, ` ```python `, etc.

Non-model tokens (observations/system messages) are masked out during training so only model-generated tokens contribute to the loss.

### Configuration

All training hyperparameters are in YAML configs under `configs/`. Key sections:
- `task`: `"math"` or `"swe"` — selects reward function and agent type
- `model`: `model_name_or_path`, optional `ref_model_name_or_path` (defaults to same model)
- `grpo`: `n_samples_per_prompt` (G), `kl_coef`, `clip_range`, `entropy_coef`
- `rollout`: `rollout_batch_size`, `max_new_tokens`, `temperature`
- `training`: `lr`, `micro_batch_size`, `train_batch_size`, `num_ppo_epochs`, `max_steps` (-1 = full dataset), `max_seq_len` (-1 = no truncation)
- `infra`: `deepspeed_stage` (2 or 3), `offload_optimizer`, `vllm_enable_sleep`
- `swe` (SWE-bench only): `max_turns`, `sandbox_timeout`, `max_concurrent_sandboxes`

Baseline GSM8K accuracy for Qwen2.5-1.5B-Instruct: 61.5%, best after training (final): 69.4%.
