"""vLLM-based rollout engine wrapped as a Ray actor."""

from __future__ import annotations

from typing import Optional

import ray
import torch
from torch import Tensor

from leanrl.experience import RolloutResult
from leanrl.utils.config import RolloutConfig, InfraConfig
from leanrl.utils.logging import logger


@ray.remote(num_gpus=1)
class RolloutEngine:
    """Ray actor that wraps a vLLM engine for high-throughput generation.

    Supports:
    - Batch generation with N samples per prompt
    - Weight synchronization from the policy model
    - Sleep mode to release GPU memory during training
    """

    def __init__(
        self,
        model_path: str,
        rollout_cfg: RolloutConfig,
        infra_cfg: InfraConfig,
        tokenizer_path: Optional[str] = None,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self._model_path = model_path
        self._rollout_cfg = rollout_cfg

        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=infra_cfg.vllm_gpu_memory_utilization,
            enforce_eager=infra_cfg.vllm_enforce_eager,
            enable_sleep_mode=infra_cfg.vllm_enable_sleep,
            tensor_parallel_size=infra_cfg.vllm_tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
        )

        tok_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompts: list[str],
        n_samples: int = 1,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> list[RolloutResult]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: list of B prompt strings.
            n_samples: G, number of completions per prompt.
            max_new_tokens: override from config.
            temperature: override from config.
            top_p: override from config.

        Returns:
            list of B*G RolloutResult objects (prompts interleaved:
            [p0_s0, p0_s1, ..., p0_sG, p1_s0, ...]).
        """
        from vllm import SamplingParams

        cfg = self._rollout_cfg
        sampling_params = SamplingParams(
            n=n_samples,
            max_tokens=max_new_tokens or cfg.max_new_tokens,
            temperature=temperature or cfg.temperature,
            top_p=top_p or cfg.top_p,
            top_k=cfg.top_k if cfg.top_k > 0 else -1,
            logprobs=1,
        )

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        results = []
        for prompt_output in outputs:
            prompt_text = prompt_output.prompt
            prompt_ids = torch.tensor(prompt_output.prompt_token_ids, dtype=torch.long)

            for completion in prompt_output.outputs:
                resp_ids = torch.tensor(completion.token_ids, dtype=torch.long)
                full_ids = torch.cat([prompt_ids, resp_ids])

                # Extract per-token log probs
                log_probs_list = []
                if completion.logprobs:
                    for lp_dict in completion.logprobs:
                        if lp_dict:
                            token_id = list(lp_dict.keys())[0]
                            log_probs_list.append(lp_dict[token_id].logprob)
                        else:
                            log_probs_list.append(0.0)
                else:
                    log_probs_list = [0.0] * len(completion.token_ids)

                old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32)

                results.append(RolloutResult(
                    prompt_ids=prompt_ids,
                    response_ids=resp_ids,
                    full_ids=full_ids,
                    old_log_probs=old_log_probs,
                    response_text=completion.text,
                    prompt_text=prompt_text,
                    prompt_len=len(prompt_ids),
                    response_len=len(resp_ids),
                ))

        return results

    def update_weights(self, state_dict: dict[str, Tensor]):
        """Sync policy weights into the vLLM engine.

        Uses vLLM's weight update mechanism via the model runner.
        """
        try:
            from vllm.worker.worker import Worker

            model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            model.load_weights(state_dict.items())
            logger.info("vLLM weights updated successfully")
        except Exception as e:
            logger.warning(f"Weight update via model_runner failed, trying collective_rpc: {e}")
            try:
                self.llm.collective_rpc("update_weights", args=(state_dict,))
            except Exception as e2:
                logger.error(f"Weight sync failed: {e2}")
                raise

    def sleep(self):
        """Release GPU memory (sleep mode)."""
        try:
            self.llm.sleep()
        except AttributeError:
            pass

    def wake_up(self):
        """Reclaim GPU memory after sleeping."""
        try:
            self.llm.wake_up()
        except AttributeError:
            pass

    def get_tokenizer(self):
        return self.tokenizer
