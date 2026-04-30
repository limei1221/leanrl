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
# Use Python 3.12 — the pinned flash-attn wheel is cp312-only.
conda create -n leanrl python=3.12 -y && conda activate leanrl
pip install -e ".[dev]"

# DeepSpeed MPI requirement:
apt-get install -y libopenmpi-dev openmpi-bin

# If CUDA tooling is missing (Ubuntu's apt nvidia-cuda-toolkit is CUDA 11.5,
# wrong major for torch cu12 — use conda instead so it scopes to the env):
conda install -n leanrl -c nvidia cuda-toolkit=12.8
export CUDA_HOME=$CONDA_PREFIX
```

**Version pins that matter**:
- `torch==2.8.0` and `flash-attn 2.8.3` (cp312/cu12 prebuilt wheel) are coupled — both need to move together. With a different torch minor, the wheel marker stops matching and you must source-build (`MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="<arch>" FLASH_ATTENTION_FORCE_BUILD=TRUE pip install -v --no-build-isolation flash-attn==2.8.3`; restrict `TORCH_CUDA_ARCH_LIST` to the target arch — `"8.0"` A100, `"8.9"` RTX 4090 — to cut build time, and lower `MAX_JOBS` on cgroup-limited containers to avoid backward-hdim OOM).
- `transformers>=4.55.2,<5.0`: vLLM 0.11.0 was built against transformers 4.x. transformers 5.0 tightened tokenizer `__getattr__` and breaks `Qwen2Tokenizer.all_special_tokens_extended`, which vLLM accesses on every rollout — symptom is `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended` from the `RolloutEngine` actor.

### Lint & Format
```bash
ruff check .
ruff check --fix .   # auto-fix
ruff format .        # auto-format
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

2. **Training phase** — reference model offloaded to CPU. vLLM releases VRAM only if `infra.vllm_enable_sleep: true`; with dedicated GPUs it stays resident. Runs PPO epochs: computes log-probs from policy, GRPO loss (clipped surrogate + KL penalty + optional entropy bonus), then optimizer step.

3. **Sync phase** — extracts state dict from DeepSpeed policy, merges Qwen-style split projections (Q/K/V → `qkv_proj`, gate/up → `gate_up_proj`) for vLLM compatibility, then serializes via `torch.save` to bytes and pushes to the Ray actor via `collective_rpc`.

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

- **GPU 0**: Policy model + Reference model (DeepSpeed ZeRO-2 by default). Reference model is always offloaded to CPU during the training phase. Adam states offload to CPU iff `infra.offload_optimizer: true`.
- **GPU 1**: vLLM `RolloutEngine` as Ray actor. Resident by default; only sleeps between rollouts when `infra.vllm_enable_sleep: true`.

Weight sync uses `torch.save`-serialized bytes over `collective_rpc` because vLLM's msgpack round-trip clears tensor aux buffers, making raw `torch.Tensor` reconstruction impossible.

### Async Rollout Prefetching

When `training.async_prefetch: true` and `infra.vllm_enable_sleep: false` (dedicated GPUs), `_train_async` maintains a bounded queue of up to `training.rollout_prefetch_depth` pre-generated rollouts on GPU 1 and overlaps generation with training on GPU 0. Weight sync back to vLLM runs every `infra.weight_sync_interval` training steps (forced before eval/checkpoint saves); a new sync is silently skipped if the previous one is still pending, tracked via `self._vllm_sync_skipped_total` and exposed in the `vllm_sync_skipped_total` wandb metric.

Max rollout staleness = `rollout_prefetch_depth + weight_sync_interval − 1` steps. `grpo.py` emits `importance_ratio` (masked mean pre-clip `exp(log_pi − log_pi_old)`) and `clip_fraction` so drift is visible in wandb. Only available for single-turn (math) tasks.

### SWE-bench Agent Actions

`parse_action()` in `multi_turn.py` accepts two formats:
- XML: `<bash>cmd</bash>`, `<edit path="file">content</edit>`, `<done/>`
- Markdown fences: ` ```bash `, ` ```python `, etc.

Non-model tokens (observations/system messages) are masked out via `response_mask` in `Experience` so only model-generated tokens contribute to the loss. This masking is critical for multi-turn SWE-bench where environment outputs are interleaved with model generations.

### Configuration

All training hyperparameters are in YAML configs under `configs/`. Key sections:
- `task`: `"math"` or `"swe"` — selects reward function and agent type
- `model`: `model_name_or_path`, optional `ref_model_name_or_path` (defaults to same model)
- `grpo`: `n_samples_per_prompt` (G), `kl_coef`, `clip_range`, `entropy_coef`
- `rollout`: `rollout_batch_size`, `max_new_tokens`, `temperature`
- `training`: `lr`, `micro_batch_size`, `train_batch_size` (→ `grad_accum = train_batch_size / micro_batch_size`), `num_ppo_epochs`, `max_steps` (-1 = full dataset), `max_seq_len` (-1 = no truncation), `async_prefetch`, `rollout_prefetch_depth` (queue depth when async)
- `infra`: `deepspeed_stage` (2 or 3), `offload_optimizer`, `vllm_enable_sleep`, `vllm_gpu_memory_utilization`, `weight_sync_interval` (sync vLLM weights every N training steps when async)
- `swe` (SWE-bench only): `max_turns`, `sandbox_timeout`, `max_concurrent_sandboxes`

### Success Thresholds

Success is task-dependent: math uses `reward > 0`, SWE uses `reward > 1.0` (test reward is up to 1.0 and reward shaping adds up to 0.3, so only fully resolved instances exceed 1.0).

### Baselines

GSM8K accuracy for Qwen/Qwen3-1.7B (thinking disabled): 74.3% baseline → 84.2% after GRPO training. See `README.md` for the full experiment table including math-specialized reference models.
