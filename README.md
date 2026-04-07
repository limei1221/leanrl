# LeanRL

A minimal GRPO (Group Relative Policy Optimization) post-training framework for LLMs, built on Ray, PyTorch, vLLM, and Docker sandboxes. Designed to run on 2–4 GPUs.

Supported tasks:
- **Math (GSM8K)** — single-turn generation with rule-based answer verification
- **SWE-bench** — multi-turn agent interaction with Docker-sandboxed test execution

## Features

- GRPO training with clipped policy loss and KL regularization
- vLLM rollout engine as a Ray actor with sleep-mode GPU sharing
- DeepSpeed ZeRO-2/3 for memory-efficient policy training
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

```bash
# Math (GSM8K/train)
bash scripts/train_math.sh

# SWE-bench (SWE-bench_Lite/test, switch to princeton-nlp/SWE-bench/train or SWE-Gym/SWE-Gym/train if possible)
bash scripts/setup_swe_docker.sh
bash scripts/train_swe.sh
```

## Architecture

```
Trainer (orchestrator)
├── RolloutEngine (vLLM, dedicated GPU via Ray)
├── PolicyModel (DeepSpeed ZeRO, training GPU)
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
| 0   | Idle           | Policy + Reference (DeepSpeed ZeRO-2) |
| 1   | vLLM serving   | vLLM sleeping, memory released |

## Configuration

All hyperparameters live in YAML files under `configs/`. See `leanrl/utils/config.py` for the full schema.

## Evaluation

```bash
# Math (GSM8K/test)
python eval_math.py --model_name_or_path <checkpoint>/final --batch_size 128

# SWE-bench (SWE-bench_Lite/dev, switch to princeton-nlp/SWE-bench_Verified if possible)
bash scripts/setup_swe_docker.sh dev
python eval_swe.py --model_name_or_path <checkpoint>/final --num_gpus 1
python eval_swe_oracle.py  # golden-patch baseline (resolve rate: 0.8567 257/300)
```

## Experiments

Hardware: 2× RTX 4090 (24 GB each), 197 GB RAM, 15 CPU cores, 485 GB disk. Python 3.12.13, PyTorch 2.10.0+cu128.

### Math

| Checkpoint | Model | Accuracy |
|------------|-------|----------|
| baseline | Qwen/Qwen2.5-1.5B-Instruct | 66.9% (882/1319) |
| step 50 | output/math_grpo_1.5b/step_50 | 67.9% (896/1319) |
| step 100 | output/math_grpo_1.5b/step_100 | 69.7% (918/1319) |
| step 150 | output/math_grpo_1.5b/step_150 | 68.5% (903/1319) |
| step 200 | output/math_grpo_1.5b/step_200 | 68.5% (904/1319) |
| final | output/math_grpo_1.5b/final | 68.6% (905/1319) |
| ref1 | Qwen/Qwen2.5-Math-1.5B-Instruct | 83.0% (1095/1319) |
| ref2 | Qwen/Qwen2.5-Math-7B-Instruct | 93.9% (1239/1319) |

### Coding

| Checkpoint | Model | Resolve Rate |
|------------|-------|--------------|
| baseline | Qwen/Qwen2.5-1.5B-Instruct | |
| ref1 | Qwen/Qwen2.5-Coder-1.5B-Instruct | |
| ref2 | Qwen/Qwen2.5-Coder-3B-Instruct | |
| ref3 | Qwen/Qwen2.5-Coder-7B-Instruct | |
| ref4 | ricdomolm/mini-coder-1.7b | |
| ref5 | ricdomolm/mini-coder-4b | |

## License

MIT
