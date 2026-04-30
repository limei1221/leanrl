"""GRPO Trainer: top-level orchestrator for LeanRL training."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
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
        self.best_eval_metric = -1.0
        self._vllm_sync_skipped_total = 0

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
        logger.info(f"Estimated total steps: {total_steps}")

        # Reference model (frozen, shares GPU with policy)
        device = torch.device("cuda")
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
        rollout_steps_per_epoch = max(1, n_samples // batch_size)
        total_rollout_steps = rollout_steps_per_epoch * cfg.training.num_epochs

        # One rollout step runs `num_ppo_epochs` over the sampled batch.
        # DeepSpeed increments optimizer steps once every
        # `gradient_accumulation_steps` micro-batches, so convert accordingly.
        samples_per_rollout = batch_size * cfg.grpo.n_samples_per_prompt
        microbatches_per_epoch = max(1, samples_per_rollout // cfg.training.micro_batch_size)
        grad_accum = max(1, cfg.training.train_batch_size // cfg.training.micro_batch_size)
        optimizer_steps_per_rollout = (
            cfg.training.num_ppo_epochs * microbatches_per_epoch // grad_accum
        )
        return total_rollout_steps * max(1, optimizer_steps_per_rollout)

    def _setup_data(self):
        """Build prompt dataloaders (train + optional eval)."""
        from leanrl.data.dataset import build_train_and_eval_dataloaders

        self.train_loader, self.eval_loader = build_train_and_eval_dataloaders(
            config=self.config.data,
            batch_size=self.config.rollout.rollout_batch_size,
            tokenizer=self.tokenizer,
            seed=self.config.training.seed,
        )
        logger.info(f"Train dataloader: {len(self.train_loader)} batches")
        if self.eval_loader is not None:
            logger.info(f"Eval dataloader: {len(self.eval_loader)} batches")

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
                policy_model=self.policy,
            )
        else:
            raise ValueError(f"Unknown task: {cfg.task}")

        logger.info(f"Executor: {cfg.task} mode")

    def _use_async_prefetch(self) -> bool:
        """Whether async rollout prefetching is enabled and applicable."""
        cfg = self.config
        if not cfg.training.async_prefetch:
            return False
        if cfg.task != "math":
            logger.warning("async_prefetch only supports single-turn (math) tasks, disabling")
            return False
        if cfg.infra.vllm_enable_sleep:
            logger.warning(
                "async_prefetch requires vllm_enable_sleep=False "
                "(vLLM must stay awake for prefetch), disabling"
            )
            return False
        return True

    def _rollout_prefetch_depth(self) -> int:
        return max(1, int(getattr(self.config.training, "rollout_prefetch_depth", 1)))

    def _weight_sync_interval(self) -> int:
        return max(1, int(getattr(self.config.infra, "weight_sync_interval", 1)))

    def _should_sync_vllm_after_step(self) -> bool:
        interval = self._weight_sync_interval()
        return (self.global_step + 1) % interval == 0

    def _needs_fresh_vllm_for_checkpoint(self) -> bool:
        cfg = self.config
        return (
            cfg.training.save_steps > 0
            and self.global_step > 0
            and self.global_step % cfg.training.save_steps == 0
            and cfg.training.save_best_only
            and self.eval_loader is not None
        )

    def _start_vllm_weight_sync(self):
        state_dict = self.policy.get_state_dict_for_vllm()
        future = self.rollout_engine.update_weights.remote(state_dict)
        del state_dict  # Ray serializes synchronously before returning.
        return future

    def _clear_completed_weight_sync(self, sync_future):
        if sync_future is None:
            return None
        ready, _ = ray.wait([sync_future], timeout=0)
        if ready:
            ray.get(ready[0])
            return None
        return sync_future

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

        # Ensure output directory exists and refresh its mtime.
        output_dir = Path(self.config.logging.output_dir)
        if output_dir.exists():
            logger.warning(f"Output directory already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        os.utime(output_dir, None)

        if self._use_async_prefetch():
            logger.info("Async rollout prefetching enabled")
            self._train_async()
        else:
            self._train_sync()

        # Final save — always save regardless of save_best_only
        self._save_checkpoint(final=True)
        self.metrics.finish()
        logger.info("Training complete!")

    def _train_sync(self):
        """Sequential training loop (original flow)."""
        cfg = self.config

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
                if self.config.infra.vllm_enable_sleep:
                    try:
                        ray.get(self.rollout_engine.sleep.remote())
                    except Exception:
                        pass

                # Free GPU memory held by the frozen reference model
                self.ref_model.offload_to_cpu()

                train_metrics = self._train_on_experience(experience)

                # Bring reference model back for next rollout
                self.ref_model.reload_to_gpu()

                # --- Wake vLLM and sync weights back when scheduled ---
                if self.config.infra.vllm_enable_sleep:
                    try:
                        ray.get(self.rollout_engine.wake_up.remote())
                    except Exception:
                        pass

                if self._should_sync_vllm_after_step() or self._needs_fresh_vllm_for_checkpoint():
                    ray.get(self._start_vllm_weight_sync())

                self._log_and_checkpoint(experience, train_metrics, step_start, epoch)

    def _train_async(self):
        """Async prefetch loop with a bounded rollout queue.

        Timeline per step:
          [GPU 0] train on N             [GPU 1] generate queued batches
          [GPU 0] ref_logprobs for N+1   [GPU 1] weight sync / more generation
        """
        cfg = self.config
        prefetch_depth = self._rollout_prefetch_depth()
        sync_interval = self._weight_sync_interval()
        logger.info(
            f"Async rollout prefetch depth={prefetch_depth}, "
            f"vLLM weight_sync_interval={sync_interval}"
        )

        for epoch in range(cfg.training.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")

            batches = list(self.train_loader)
            if not batches:
                continue
            if cfg.training.max_steps > 0:
                remaining_steps = cfg.training.max_steps - self.global_step
                if remaining_steps <= 0:
                    break
                batches = batches[:remaining_steps]

            # First batch: synchronous rollout (no overlap possible yet)
            first_batch = batches[0]
            experience = self.executor.execute(
                prompts=first_batch["prompts"],
                labels=first_batch["labels"],
                tokenizer=self.tokenizer,
            )

            pending_rollouts = deque()
            next_prefetch_idx = 1
            pending_sync = None

            def fill_rollout_queue():
                nonlocal next_prefetch_idx
                while next_prefetch_idx < len(batches) and len(pending_rollouts) < prefetch_depth:
                    batch = batches[next_prefetch_idx]
                    future, expanded_labels, G = self.executor.start_rollout(
                        prompts=batch["prompts"],
                        labels=batch["labels"],
                        tokenizer=self.tokenizer,
                    )
                    pending_rollouts.append((future, expanded_labels, G))
                    next_prefetch_idx += 1

            fill_rollout_queue()

            for batch_idx in range(len(batches)):
                step_start = time.time()
                has_next = batch_idx + 1 < len(batches)

                # Train on current experience (GPU 0).
                self.ref_model.offload_to_cpu()
                train_metrics = self._train_on_experience(experience)
                self.ref_model.reload_to_gpu()

                pending_sync = self._clear_completed_weight_sync(pending_sync)
                fresh_sync_required = self._needs_fresh_vllm_for_checkpoint()
                sync_due = self._should_sync_vllm_after_step() or fresh_sync_required

                if fresh_sync_required:
                    if pending_sync is not None:
                        ray.get(pending_sync)
                    pending_sync = self._start_vllm_weight_sync()
                elif sync_due:
                    if pending_sync is None:
                        pending_sync = self._start_vllm_weight_sync()
                    else:
                        self._vllm_sync_skipped_total += 1
                        logger.info(
                            "Previous vLLM weight sync is still pending; "
                            "skipping this scheduled sync "
                            f"(total skipped={self._vllm_sync_skipped_total})"
                        )

                # Build next experience on GPU 0 while GPU 1 continues with
                # queued generation or the weight sync above.
                next_experience = None
                if has_next:
                    if not pending_rollouts:
                        fill_rollout_queue()
                    rollout_future, expanded_labels, G = pending_rollouts.popleft()
                    next_experience = self.executor.finish_experience(
                        rollout_future,
                        expanded_labels,
                        G,
                        self.tokenizer,
                    )

                # Eval/checkpoint rollouts should see weights from the just
                # completed train step. Normal training can tolerate bounded
                # staleness from the prefetch queue.
                if fresh_sync_required and pending_sync is not None:
                    ray.get(pending_sync)
                    pending_sync = None

                self._log_and_checkpoint(experience, train_metrics, step_start, epoch)

                if has_next:
                    fill_rollout_queue()
                    experience = next_experience

            if pending_sync is not None:
                ray.get(pending_sync)

    def _log_and_checkpoint(
        self,
        experience: Experience,
        train_metrics: dict,
        step_start: float,
        epoch: int,
    ):
        """Shared logging and checkpointing logic for both sync and async loops."""
        cfg = self.config
        success_mask = self._success_mask(experience.rewards)

        step_time = time.time() - step_start
        step_metrics = dict(train_metrics)
        step_metrics["step_time"] = step_time
        step_metrics["reward_mean"] = experience.rewards.mean().item()
        step_metrics["reward_std"] = experience.rewards.std().item()
        step_metrics["pass_rate"] = success_mask.float().mean().item()
        step_metrics["epoch"] = epoch + 1
        step_metrics["vllm_sync_skipped_total"] = self._vllm_sync_skipped_total

        if self.global_step % cfg.training.logging_steps == 0:
            self.metrics.log(step_metrics, step=self.global_step)

        if (
            cfg.training.save_steps > 0
            and self.global_step > 0
            and self.global_step % cfg.training.save_steps == 0
        ):
            if cfg.training.save_best_only and self.eval_loader is not None:
                eval_metric = self._evaluate()
                metric_name = "accuracy" if self.config.task == "math" else "resolve_rate"
                eval_metrics = {
                    f"eval_{metric_name}": eval_metric,
                }
                self.metrics.log(eval_metrics, step=self.global_step)
                if eval_metric > self.best_eval_metric:
                    logger.info(
                        f"New best eval metric: {eval_metric:.4f} "
                        f"(prev {self.best_eval_metric:.4f})"
                    )
                    self.best_eval_metric = eval_metric
                    self._save_checkpoint()
                else:
                    logger.info(
                        f"Eval metric {eval_metric:.4f} did not improve "
                        f"(best {self.best_eval_metric:.4f}), skipping checkpoint"
                    )
            else:
                self._save_checkpoint()

        self.global_step += 1

    def _success_mask(self, rewards: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask for successful samples under the active task."""
        if self.config.task == "swe":
            # SWE test reward is up to 1.0, shaping adds up to 0.3.
            # A resolved instance has test_reward == 1.0, so total > 1.0.
            return rewards > 1.0
        return rewards > 0

    def _train_on_experience(self, experience: Experience) -> dict:
        """Run multiple PPO epochs over the experience batch."""
        cfg = self.config
        device = torch.device("cuda")
        exp = experience.to(device)
        all_metrics: dict[str, float] = {}
        num_updates = 0

        for ppo_epoch in range(cfg.training.num_ppo_epochs):
            n = len(exp)
            indices = torch.randperm(n, device=device)
            mbs = cfg.training.micro_batch_size

            for start in range(0, n, mbs):
                idx = indices[start : start + mbs]

                mb_input_ids = exp.input_ids[idx]
                mb_attn_mask = exp.attention_mask[idx]
                mb_old_lp = exp.old_log_probs[idx]
                mb_ref_lp = exp.ref_log_probs[idx]
                mb_advantages = exp.advantages[idx]
                mb_resp_mask = exp.response_mask[idx]
                mb_resp_lens = exp.response_lengths[idx]

                # Forward pass to get current log-probs and entropy
                cur_lp, cur_entropy = self.policy.forward_logprobs_from_experience(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attn_mask,
                    response_lengths=mb_resp_lens,
                    max_resp_len=mb_resp_mask.shape[1],
                    compute_entropy=(cfg.grpo.entropy_coef > 0),
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
                    entropy=cur_entropy,
                )

                # Backward + step
                step_info = self.policy.train_step(loss)
                metrics.update(step_info)

                for k, v in metrics.items():
                    all_metrics[k] = all_metrics.get(k, 0.0) + v
                num_updates += 1

        if num_updates > 0:
            all_metrics = {k: v / num_updates for k, v in all_metrics.items()}
        return all_metrics

    def _evaluate(self) -> float:
        """Run eval and return the metric (accuracy for math, resolve_rate for swe)."""
        logger.info("Running evaluation...")
        total_correct = 0
        total_samples = 0

        for batch in self.eval_loader:
            prompts = batch["prompts"]
            labels = batch["labels"]

            experience = self.executor.execute(
                prompts=prompts,
                labels=labels,
                tokenizer=self.tokenizer,
            )

            # Each prompt generates G samples; rewards are per-sample.
            # For SWE, only a full reward counts as resolved even if the
            # training reward uses partial credit.
            total_correct += self._success_mask(experience.rewards).float().sum().item()
            total_samples += len(experience.rewards)

        metric = total_correct / total_samples if total_samples > 0 else 0.0
        metric_name = "accuracy" if self.config.task == "math" else "resolve_rate"
        logger.info(f"Eval {metric_name}: {metric:.4f} ({int(total_correct)}/{total_samples})")
        return metric

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
