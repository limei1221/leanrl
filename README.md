# LeanRL

A minimal GRPO (Group Relative Policy Optimization) post-training framework for LLMs, built on Ray, PyTorch, vLLM, and Docker sandboxes. Designed to run on 2 GPUs.

Supported tasks:
- **Math (GSM8K)** — single-turn generation with rule-based answer verification
- **SWE-bench** — multi-turn agent interaction with Docker-sandboxed test execution

## Features

- GRPO training with clipped policy loss and KL regularization
- vLLM rollout engine as a Ray actor with sleep-mode GPU sharing
- DeepSpeed for memory-efficient policy training (optimizer CPU offload on single GPU, ZeRO-2/3 sharding on multi-GPU)
- Async rollout prefetching: overlaps next batch generation (GPU 1) with training (GPU 0)
- Docker sandboxes for safe, isolated code execution

## Quick Start

```bash
# Install
conda create -n leanrl python=3.12 -y
conda activate leanrl
pip install -e ".[dev]"

# System dependencies required by DeepSpeed:
apt-get update && apt-get install -y libopenmpi-dev openmpi-bin

# If CUDA tooling is missing:
conda install -n leanrl -c nvidia cuda-toolkit=12.8
```

Training requires a 2-GPU setup: GPU 0 for the policy and reference models (DeepSpeed), GPU 1 for the vLLM rollout engine (Ray).

SWE-bench training requires Docker for sandboxed test execution.

```bash
# Math (GSM8K/train)
bash scripts/train_math.sh

# SWE-bench (SWE-bench_Lite/test 284 samples, switch to princeton-nlp/SWE-bench/train or SWE-Gym/SWE-Gym/train if possible)
bash scripts/setup_swe_docker.sh
bash scripts/train_swe.sh
```

## Architecture

```
Trainer (orchestrator)
├── RolloutEngine (vLLM, dedicated GPU via Ray)
├── PolicyModel (DeepSpeed, training GPU)
├── ReferenceModel (frozen, training GPU)
└── RewardRouter
    ├── MathReward (rule-based verification)
    └── SWEReward (Docker sandbox test execution)
```

Each training step follows the GRPO loop:
1. Generate G completions per prompt via vLLM
2. Score completions with the reward function
3. Compute group-normalized advantages
4. Update the policy with clipped surrogate loss + KL penalty
5. Sync updated weights back to the vLLM engine

### GPU Layout (2-GPU)

| GPU | Role |
|-----|------|
| 0   | Policy + Reference models (DeepSpeed ZeRO-2). Optional Adam-state CPU offload via `infra.offload_optimizer`. |
| 1   | vLLM rollout engine. Resident by default; with `infra.vllm_enable_sleep: true` it sleeps between rollouts to free VRAM. |

### Async Rollout Prefetching

When `training.async_prefetch: true` and `infra.vllm_enable_sleep: false` (dedicated GPUs), the trainer overlaps vLLM generation on GPU 1 with training on GPU 0. A bounded queue of up to `training.rollout_prefetch_depth` rollouts is kept ready on GPU 1; weight sync back to vLLM runs every `infra.weight_sync_interval` training steps (and is forced before eval/checkpoint saves).

```
Sequential: [Gen N] [Train N] [Sync] [Gen N+1] [Train N+1] ...
Async:      [Gen N..N+D] [Train N + background gen/sync] [Train N+1 + ...] ...
            where D = rollout_prefetch_depth
```

Prefetched rollouts can be up to `rollout_prefetch_depth + weight_sync_interval − 1` steps stale, which is handled by PPO/GRPO importance sampling — watch the `importance_ratio` metric in wandb for actual drift. Only available for single-turn (math) tasks.

## Configuration

All hyperparameters live in YAML files under `configs/`. See `leanrl/utils/config.py` for the full schema.

## Evaluation

```bash
# Math (GSM8K/test)
python scripts/eval_math.py --model_name_or_path <checkpoint>/final

# SWE-bench (SWE-bench_Lite/test 16 samples, switch to princeton-nlp/SWE-bench_Verified if possible)
python scripts/eval_swe.py --model_name_or_path <checkpoint>/final
python scripts/eval_swe_oracle.py  # golden-patch baseline (resolve rate: 85.7% 257/300)
python scripts/eval_swe_oracle.py --num_samples 16  # golden-patch baseline on 16 test samples (resolve rate: 93.8%  (15/16))
```

## Experiments

### Math

Hardware: 2× A100 (80 GB each), ~465 GB RAM, 27.2 vCPUs, 30 GB disk. Python 3.12.3, PyTorch 2.8.0+cu128.

|  | Model | Accuracy |
|------------|-------|----------|
| baseline | Qwen/Qwen3-1.7B | 74.3%  (980/1319) |
| exp | output/math_grpo_1.5b/final | 84.2%  (1111/1319) |
| ref1 | Qwen/Qwen2.5-Math-1.5B-Instruct | 85.1%  (1123/1319) |
| ref2 | Qwen/Qwen2.5-Math-7B-Instruct | 95.3%  (1257/1319) |

### Coding

Hardware: 2× RTX 4090 (48 GB each), ~198 GB RAM, 61 vCPUs, 500 GB disk. Python 3.12.13, PyTorch 2.8.0+cu128.

default: max_turns=20, max_new_tokens=2048

|  | Model | Resolve Rate |
|------------|-------|--------------|
| baseline | ricdomolm/mini-coder-1.7b | 0.0%  (0/16) (max_turns=10, max_new_tokens=2048) |
| baseline | ricdomolm/mini-coder-1.7b | 0.0%  (0/16) (max_turns=15, max_new_tokens=2048) |
| baseline | ricdomolm/mini-coder-1.7b | 6.2%  (1/16) |
| exp | TODO |  |
| ref | ricdomolm/mini-coder-4b | 6.2%  (1/16) |

## License

MIT
