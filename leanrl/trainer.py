"""GRPO Trainer: top-level orchestrator for LeanRL training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ray
import torch

from leanrl.utils.config import TrainConfig
from leanrl.utils.logging import logger, MetricsTracker
from leanrl.grpo import grpo_loss
from leanrl.experience import Experience


class GRPOTrainer:
    """Orchestrates the GRPO training loop.

    Manages: vLLM rollout engine (Ray), policy model (DeepSpeed),
    reference model, reward functions, and agent executors.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self._setup_infrastructure()
        self._setup_models()
        self._setup_data()
        self._setup_executor()

        self.metrics = MetricsTracker(
            use_wandb=config.logging.use_wandb,
            project=config.logging.wandb_project,
            run_name=config.logging.wandb_run_name,
        )
        self.global_step = 0

    def _setup_infrastructure(self):
        """Initialize Ray cluster."""
        cfg = self.config.infra
        if not ray.is_initialized():
            ray.init(address=cfg.ray_address, ignore_reinit_error=True)
        logger.info(f"Ray initialized: {ray.cluster_resources()}")

    def _setup_models(self):
        """Load policy, reference, and rollout models."""
        cfg = self.config

        # Rollout engine (vLLM on dedicated GPU)
        from leanrl.rollout import RolloutEngine

        self.rollout_engine = RolloutEngine.remote(
            model_path=cfg.model.model_name_or_path,
            rollout_cfg=cfg.rollout,
            infra_cfg=cfg.infra,
        )
        logger.info("Rollout engine created")

        # Policy model (DeepSpeed on training GPU)
        from leanrl.models import PolicyModel, ReferenceModel

        total_steps = self._estimate_total_steps()
        self.policy = PolicyModel(
            model_cfg=cfg.model,
            training_cfg=cfg.training,
            infra_cfg=cfg.infra,
            total_steps=total_steps,
        )
        self.tokenizer = self.policy.tokenizer
        logger.info("Policy model loaded")

        # Reference model (frozen, shares GPU with policy)
        device = torch.device("cuda:0")
        self.ref_model = ReferenceModel(cfg.model, device=device)
        logger.info("Reference model loaded")

    def _estimate_total_steps(self) -> int:
        cfg = self.config
        if cfg.training.max_steps > 0:
            return cfg.training.max_steps
        # Rough estimate based on dataset size
        try:
            from datasets import load_dataset

            if cfg.data.prompt_dataset == "openai/gsm8k":
                ds = load_dataset("openai/gsm8k", "main", split=cfg.data.prompt_dataset_split)
            else:
                ds = load_dataset(cfg.data.prompt_dataset, split=cfg.data.prompt_dataset_split)
            n_samples = len(ds)
            if cfg.data.max_samples > 0:
                n_samples = min(n_samples, cfg.data.max_samples)
        except Exception:
            n_samples = 10000

        batch_size = cfg.rollout.rollout_batch_size
        steps_per_epoch = max(1, n_samples // batch_size)
        return steps_per_epoch * cfg.training.num_epochs

    def _setup_data(self):
        """Build prompt dataloader."""
        from leanrl.data.dataset import build_prompt_dataloader

        self.train_loader = build_prompt_dataloader(
            config=self.config.data,
            batch_size=self.config.rollout.rollout_batch_size,
            tokenizer=self.tokenizer,
        )
        logger.info(f"Dataloader: {len(self.train_loader)} batches")

    def _setup_executor(self):
        """Build the appropriate agent executor based on task type."""
        cfg = self.config

        if cfg.task == "math":
            from leanrl.reward.math_reward import compute_math_rewards
            from leanrl.agent.single_turn import SingleTurnExecutor

            self.executor = SingleTurnExecutor(
                rollout_engine=self.rollout_engine,
                reward_fn=compute_math_rewards,
                ref_model=self.ref_model,
                config=cfg,
            )
        elif cfg.task == "swe":
            from leanrl.agent.multi_turn import MultiTurnExecutor

            self.executor = MultiTurnExecutor(
                rollout_engine=self.rollout_engine,
                ref_model=self.ref_model,
                config=cfg,
            )
        else:
            raise ValueError(f"Unknown task: {cfg.task}")

        logger.info(f"Executor: {cfg.task} mode")

    def train(self):
        """Main GRPO training loop."""
        cfg = self.config
        logger.info("=" * 60)
        logger.info("Starting GRPO training")
        logger.info(f"  Task: {cfg.task}")
        logger.info(f"  Model: {cfg.model.model_name_or_path}")
        logger.info(f"  G (samples/prompt): {cfg.grpo.n_samples_per_prompt}")
        logger.info(f"  KL coef: {cfg.grpo.kl_coef}")
        logger.info(f"  Clip range: {cfg.grpo.clip_range}")
        logger.info(f"  LR: {cfg.training.lr}")
        logger.info("=" * 60)

        for epoch in range(cfg.training.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")

            for batch_idx, batch in enumerate(self.train_loader):
                if cfg.training.max_steps > 0 and self.global_step >= cfg.training.max_steps:
                    break

                step_start = time.time()

                # --- Rollout phase (vLLM GPU active, training GPU idle) ---
                prompts = batch["prompts"]
                labels = batch["labels"]

                experience = self.executor.execute(
                    prompts=prompts,
                    labels=labels,
                    tokenizer=self.tokenizer,
                )

                # --- Training phase (vLLM sleeping, training GPU active) ---
                try:
                    ray.get(self.rollout_engine.sleep.remote())
                except Exception:
                    pass

                step_metrics = self._train_on_experience(experience)

                # --- Sync weights back to vLLM ---
                try:
                    ray.get(self.rollout_engine.wake_up.remote())
                except Exception:
                    pass

                state_dict = self.policy.get_state_dict_for_vllm()
                ray.get(self.rollout_engine.update_weights.remote(state_dict))

                # --- Logging ---
                step_time = time.time() - step_start
                step_metrics["step_time"] = step_time
                step_metrics["reward_mean"] = experience.rewards.mean().item()
                step_metrics["reward_std"] = experience.rewards.std().item()
                step_metrics["pass_rate"] = (experience.rewards > 0).float().mean().item()
                step_metrics["epoch"] = epoch + 1

                if self.global_step % cfg.training.logging_steps == 0:
                    self.metrics.log(step_metrics, step=self.global_step)

                # --- Checkpointing ---
                if cfg.training.save_steps > 0 and (self.global_step + 1) % cfg.training.save_steps == 0:
                    self._save_checkpoint()

                self.global_step += 1

        # Final save
        self._save_checkpoint(final=True)
        self.metrics.finish()
        logger.info("Training complete!")

    def _train_on_experience(self, experience: Experience) -> dict:
        """Run multiple PPO epochs over the experience batch."""
        cfg = self.config
        device = torch.device("cuda:0")
        exp = experience.to(device)
        all_metrics = {}

        for ppo_epoch in range(cfg.training.num_ppo_epochs):
            n = len(exp)
            indices = torch.randperm(n, device=device)
            mbs = cfg.training.micro_batch_size

            for start in range(0, n, mbs):
                idx = indices[start : start + mbs]

                mb_input_ids = exp.input_ids[idx]
                mb_attn_mask = exp.attention_mask[idx]
                mb_response_ids = exp.response_ids[idx]
                mb_old_lp = exp.old_log_probs[idx]
                mb_ref_lp = exp.ref_log_probs[idx]
                mb_advantages = exp.advantages[idx]
                mb_resp_mask = exp.response_mask[idx]

                # Forward pass to get current log-probs
                cur_lp = self.policy.forward_logprobs_from_experience(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attn_mask,
                    response_ids=mb_response_ids,
                    response_mask=mb_resp_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                # Compute GRPO loss
                loss, metrics = grpo_loss(
                    log_probs=cur_lp,
                    old_log_probs=mb_old_lp,
                    ref_log_probs=mb_ref_lp,
                    advantages=mb_advantages,
                    mask=mb_resp_mask,
                    clip_range=cfg.grpo.clip_range,
                    kl_coef=cfg.grpo.kl_coef,
                    entropy_coef=cfg.grpo.entropy_coef,
                )

                # Backward + step
                step_info = self.policy.train_step(loss)
                metrics.update(step_info)
                all_metrics = metrics

        return all_metrics

    def _save_checkpoint(self, final: bool = False):
        output_dir = Path(self.config.logging.output_dir)
        if final:
            save_path = output_dir / "final"
        else:
            save_path = output_dir / f"step_{self.global_step}"

        self.policy.save_hf(save_path)
        logger.info(f"Checkpoint saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="LeanRL GRPO Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    trainer = GRPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
