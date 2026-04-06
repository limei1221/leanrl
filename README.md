# LeanRL

A minimal RL post-training framework for LLMs using Ray + PyTorch + vLLM + Docker sandboxes.

LeanRL implements **GRPO (Group Relative Policy Optimization)** for both math verification (single-turn) and SWE-bench (multi-turn agent) training, designed to run on 2-4 GPUs.

## Features

- **GRPO training** with clipped policy loss and KL regularization
- **vLLM rollout engine** as a Ray actor with sleep-mode GPU sharing
- **Single-turn math training** with rule-based reward verification
- **Multi-turn SWE-bench training** with Docker sandbox execution
- **DeepSpeed ZeRO-2/3** for memory-efficient policy training

## Quick Start

```bash
# Install (editable + dev deps)
# On a fresh GPU machine, system MPI libraries are required by DeepSpeed at runtime:
#   apt-get update && apt-get install -y libopenmpi-dev openmpi-bin
# If CUDA tooling is missing, install it:
#   apt-get install -y nvidia-cuda-toolkit
pip install -e ".[dev]"

# 2-GPU setup (recommended)
#
# GPU 0: Policy + Reference model (DeepSpeed)
# GPU 1: vLLM rollout engine (via Ray)

# Train on GSM8K (math)
bash scripts/train_math.sh

# Train on SWE-bench
bash scripts/setup_swe_docker.sh
bash scripts/train_swe.sh
```

## Architecture

```
Trainer (orchestrator)
  ├── RolloutEngine (vLLM on dedicated GPU, Ray actor)
  ├── PolicyModel (DeepSpeed ZeRO, training GPU)
  ├── ReferenceModel (frozen, training GPU)
  └── RewardRouter
       ├── MathReward (rule-based verification)
       └── SWEReward (Docker sandbox test execution)
```

**Training loop** (GRPO):
1. Generate G samples per prompt via vLLM
2. Score with reward function (math verifier or Docker sandbox)
3. Compute group-normalized advantages
4. Update policy with clipped surrogate loss + KL penalty
5. Sync weights back to vLLM engine

## Configuration

All settings live in YAML config files under `configs/`. See `leanrl/utils/config.py` for the full schema.

## GPU Layout (2-GPU)

| GPU | During Rollout | During Training |
|-----|---------------|-----------------|
| 0 | idle | Policy + Reference (DeepSpeed ZeRO-2) |
| 1 | vLLM serving | vLLM sleeping, memory released | 

## License

MIT
