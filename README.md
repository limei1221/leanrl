# LeanRL

A minimal GRPO (Group Relative Policy Optimization) post-training framework for LLMs, built on Ray, PyTorch, vLLM, and Docker sandboxes. Designed to run on 2–4 GPUs.

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
pip install -e ".[dev]"

# System dependencies required by DeepSpeed:
apt-get update && apt-get install -y libopenmpi-dev openmpi-bin
# If CUDA tooling is missing:
apt-get install -y nvidia-cuda-toolkit
```

Training requires a 2-GPU setup: GPU 0 for the policy and reference models (DeepSpeed), GPU 1 for the vLLM rollout engine (Ray).

SWE-bench training requires a container runtime. Docker or [Podman](https://podman.io/) (rootless, useful when Docker is unavailable, e.g. vast.ai) both work — the scripts auto-detect which is available.

```bash
# If Docker is not available, install Podman instead:
apt-get install -y podman
```

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

| GPU | During Rollout | During Training |
|-----|----------------|-----------------|
| 0   | Idle           | Policy + Reference (DeepSpeed, optimizer offloaded to CPU) |
| 1   | vLLM serving   | vLLM sleeping, memory released |

### Async Rollout Prefetching

When `training.async_prefetch: true` and `infra.vllm_enable_sleep: false` (dedicated GPUs), the trainer overlaps the next batch's vLLM generation on GPU 1 with training on GPU 0:

```
Sequential:  [Generate N] [Train N] [Sync] [Generate N+1] [Train N+1] ...
Async:       [Generate N] [Train N + Generate N+1] [Sync] [Train N+1 + Generate N+2] ...
```

This saves up to `min(T_generate, T_train)` per step. The prefetched rollouts use weights that are one step stale, which is handled by PPO/GRPO importance sampling. Only available for single-turn (math) tasks.

## Configuration

All hyperparameters live in YAML files under `configs/`. See `leanrl/utils/config.py` for the full schema.

## Evaluation

```bash
# Math (GSM8K/test)
python scripts/eval_math.py --model_name_or_path <checkpoint>/final

# SWE-bench (SWE-bench_Lite/test 16 samples, switch to princeton-nlp/SWE-bench_Verified if possible)
python scripts/eval_swe.py --model_name_or_path <checkpoint>/final
python scripts/eval_swe_oracle.py  # golden-patch baseline (resolve rate: 86.0% 258/300)
python scripts/eval_swe_oracle.py --num_samples 16  # golden-patch baseline on 16 test samples (resolve rate: 93.8% 15/16)
```

## Experiments

Hardware: 2× RTX 4090 (24 GB each), 197 GB RAM, 15 CPU cores, 485 GB disk. Python 3.12.13, PyTorch 2.10.0+cu128.

### Math

|  | Model | Accuracy |
|------------|-------|----------|
| baseline | Qwen/Qwen2.5-1.5B-Instruct | 61.5%  (811/1319) |
| exp | output/math_grpo_1.5b/final | 69.4%  (916/1319) |
| ref1 | Qwen/Qwen2.5-Math-1.5B-Instruct | 84.6%  (1116/1319) |
| ref2 | Qwen/Qwen2.5-Math-7B-Instruct | 94.1%  (1241/1319) |

### Coding
max_turns=10, max_new_tokens=512

|  | Model | Resolve Rate |
|------------|-------|--------------|
| baseline | ricdomolm/mini-coder-1.7b | 0.0%  (0/16) |
| baseline | ricdomolm/mini-coder-1.7b | 6.2%  (1/16) (max_turns=15, max_new_tokens=1024)|
| baseline | ricdomolm/mini-coder-1.7b | 12.5%  (2/16) (max_turns=20, max_new_tokens=2048) |
| exp |  | (too difficult to train on current GPU machine) |

## License

MIT
