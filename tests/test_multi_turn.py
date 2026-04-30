"""Tests for SWE multi-turn action parsing and token masking."""

import sys
from types import SimpleNamespace

import torch

from leanrl.agent.multi_turn import (
    parse_action,
    ACTION_BASH,
    ACTION_DONE,
    _append_prompt_delta,
    _pool_task,
    MultiTurnExecutor,
)
from leanrl.experience import RolloutResult
from leanrl.agent.sandbox import SandboxResult, TaskInstance


class TestParseActionBashTag:
    def test_simple_bash(self):
        assert parse_action("<bash>ls /testbed</bash>") == (ACTION_BASH, "ls /testbed")

    def test_multiline_bash(self):
        text = "<bash>\ncd /testbed\npython -m pytest tests/\n</bash>"
        action, content = parse_action(text)
        assert action == ACTION_BASH
        assert "cd /testbed" in content
        assert "python -m pytest" in content

    def test_bash_strips_whitespace(self):
        assert parse_action("<bash>  echo hi  </bash>") == (ACTION_BASH, "echo hi")


class TestParseActionBashFence:
    def test_bash_fence(self):
        text = "Let me check:\n```bash\nls /testbed\n```"
        assert parse_action(text) == (ACTION_BASH, "ls /testbed")

    def test_sh_fence(self):
        text = "```sh\ncat file.py\n```"
        assert parse_action(text) == (ACTION_BASH, "cat file.py")

    def test_done_inside_bash_fence(self):
        text = "```bash\n<done/>\n```"
        assert parse_action(text) == (ACTION_DONE, "")


class TestParseActionDone:
    def test_done_tag(self):
        assert parse_action("All fixed. <done/>") == (ACTION_DONE, "")

    def test_done_tag_with_space(self):
        assert parse_action("<done />") == (ACTION_DONE, "")

    def test_no_action_is_done(self):
        assert parse_action("I think the issue is...") == (ACTION_DONE, "")


class TestEditActionsRemoved:
    """Verify that edit-style actions are no longer parsed."""

    def test_edit_tag_ignored(self):
        text = '<edit path="foo.py">print("hi")</edit>'
        action, _ = parse_action(text)
        # Should NOT return an edit action — falls through to done
        assert action == ACTION_DONE

    def test_python_fence_not_converted(self):
        text = "```python\nimport os\nprint(os.getcwd())\n```"
        action, _ = parse_action(text)
        # Without a bash/sh fence, this is not recognized — falls through to done
        assert action == ACTION_DONE

    def test_python_fence_with_path_ignored(self):
        text = 'Edit `/testbed/foo.py`:\n```python\nprint("hi")\n```'
        action, _ = parse_action(text)
        assert action == ACTION_DONE

    def test_bash_tag_preferred_over_edit_tag(self):
        text = '<edit path="x.py">stuff</edit>\n<bash>echo ok</bash>'
        action, content = parse_action(text)
        assert action == ACTION_BASH
        assert content == "echo ok"


class TestActionConstantRemoved:
    def test_no_action_edit_constant(self):
        import leanrl.agent.multi_turn as mod

        assert not hasattr(mod, "ACTION_EDIT")


class TestPromptDeltaMasking:
    def test_prompt_delta_is_added_as_masked_context(self):
        response_ids = [201]
        old_log_probs = [-0.5]
        response_mask = [1.0]

        ok = _append_prompt_delta(
            response_ids=response_ids,
            old_log_probs=old_log_probs,
            response_mask=response_mask,
            new_prompt_ids=[101, 102, 201, 301, 302],
            initial_prompt_ids=[101, 102],
        )

        assert ok
        assert response_ids == [201, 301, 302]
        assert old_log_probs == [-0.5, 0.0, 0.0]
        assert response_mask == [1.0, 0.0, 0.0]

    def test_prompt_delta_returns_false_on_drift(self):
        response_ids = [201]
        old_log_probs = [-0.5]
        response_mask = [1.0]

        ok = _append_prompt_delta(
            response_ids=response_ids,
            old_log_probs=old_log_probs,
            response_mask=response_mask,
            new_prompt_ids=[101, 999, 201, 301],
            initial_prompt_ids=[101, 102],
        )

        assert ok is False
        # Inputs untouched on drift
        assert response_ids == [201]
        assert old_log_probs == [-0.5]
        assert response_mask == [1.0]

    def test_prompt_delta_returns_false_when_new_prompt_shorter(self):
        ok = _append_prompt_delta(
            response_ids=[201, 202],
            old_log_probs=[-0.1, -0.2],
            response_mask=[1.0, 1.0],
            new_prompt_ids=[101, 102, 201],
            initial_prompt_ids=[101, 102],
        )
        assert ok is False

    def test_empty_delta_is_noop(self):
        response_ids = [201]
        old_log_probs = [-0.5]
        response_mask = [1.0]

        ok = _append_prompt_delta(
            response_ids=response_ids,
            old_log_probs=old_log_probs,
            response_mask=response_mask,
            new_prompt_ids=[101, 102, 201],
            initial_prompt_ids=[101, 102],
        )

        assert ok
        assert response_ids == [201]
        assert old_log_probs == [-0.5]
        assert response_mask == [1.0]

    def test_sequential_multi_turn_accumulation(self):
        """Three simulated turns of delta+generation, verifying invariants."""
        initial_prompt = [101, 102]
        response_ids: list[int] = []
        old_log_probs: list[float] = []
        response_mask: list[float] = []

        # Turn 0: no delta; just append generation.
        gen_0 = [201, 202]
        response_ids.extend(gen_0)
        old_log_probs.extend([-0.1, -0.2])
        response_mask.extend([1.0, 1.0])

        # Turn 1: prompt re-renders as initial + gen_0 + delta_1.
        ok = _append_prompt_delta(
            response_ids=response_ids,
            old_log_probs=old_log_probs,
            response_mask=response_mask,
            new_prompt_ids=[101, 102, 201, 202, 301, 302],
            initial_prompt_ids=initial_prompt,
        )
        assert ok
        gen_1 = [401]
        response_ids.extend(gen_1)
        old_log_probs.extend([-0.3])
        response_mask.extend([1.0])

        # Turn 2: prompt re-renders as initial + gen_0 + delta_1 + gen_1 + delta_2.
        ok = _append_prompt_delta(
            response_ids=response_ids,
            old_log_probs=old_log_probs,
            response_mask=response_mask,
            new_prompt_ids=[101, 102, 201, 202, 301, 302, 401, 501, 502],
            initial_prompt_ids=initial_prompt,
        )
        assert ok
        assert response_ids == [201, 202, 301, 302, 401, 501, 502]
        assert response_mask == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        assert old_log_probs == [-0.1, -0.2, 0.0, 0.0, -0.3, 0.0, 0.0]


class _ToyChatTokenizer:
    """Small deterministic tokenizer for chat-template round-trip tests."""

    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) + 1 for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i != self.eos_token_id]
        return "".join(chr(i - 1) for i in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs,
    ) -> list[int] | str:
        tokens: list[int] = []
        text_parts: list[str] = []
        for msg in messages:
            header = f"<|{msg['role']}|>\n"
            text_parts.extend([header, msg["content"]])
            tokens.extend(self.encode(header))
            tokens.extend(self.encode(msg["content"]))
            if msg["role"] == "assistant":
                tokens.append(self.eos_token_id)
                text_parts.append("<|eos|>")
            text_parts.append("\n")
            tokens.extend(self.encode("\n"))

        if add_generation_prompt:
            generation_prompt = "<|assistant|>\n"
            tokens.extend(self.encode(generation_prompt))
            text_parts.append(generation_prompt)

        if tokenize:
            return tokens
        return "".join(text_parts)


class TestPromptDeltaChatTokenizer:
    """Round-trip smoke test with a chat-template tokenizer.

    Exercises the path that breaks in production: decode(gen_ids) placed back
    into apply_chat_template should retokenize to the original gen_ids.
    """

    def test_clean_round_trip_across_turns(self):
        tok = _ToyChatTokenizer()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix the bug in foo.py."},
        ]
        initial_prompt = list(
            tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )

        gen_text_0 = "Let me check the code.\n<bash>ls /testbed</bash>"
        gen_ids_0 = tok.encode(gen_text_0, add_special_tokens=False) + [tok.eos_token_id]
        gen_text_0_decoded = tok.decode(gen_ids_0, skip_special_tokens=True)

        response_ids = list(gen_ids_0)
        old_log_probs = [-0.1] * len(gen_ids_0)
        response_mask = [1.0] * len(gen_ids_0)

        messages_t1 = messages + [
            {"role": "assistant", "content": gen_text_0_decoded},
            {"role": "user", "content": "[Observation]\nfile1.py\nfile2.py\n\nContinue."},
        ]
        prompt_t1 = list(
            tok.apply_chat_template(messages_t1, tokenize=True, add_generation_prompt=True)
        )

        ok = _append_prompt_delta(
            response_ids=response_ids,
            old_log_probs=old_log_probs,
            response_mask=response_mask,
            new_prompt_ids=prompt_t1,
            initial_prompt_ids=initial_prompt,
        )

        assert ok, (
            "Plain-ASCII assistant content should round-trip through the chat "
            "tokenizer; drift here indicates a real regression."
        )
        delta_len = len(response_ids) - len(gen_ids_0)
        assert delta_len > 0
        assert response_mask[-delta_len:] == [0.0] * delta_len
        assert old_log_probs[-delta_len:] == [0.0] * delta_len
        # Full accumulated stream equals the re-rendered turn-1 prompt.
        assert initial_prompt + response_ids == prompt_t1


class TestBatchedMultiTurnRollout:
    def test_pool_task_uses_unique_pool_key_and_original_image(self):
        task = TaskInstance(
            instance_id="django__django-12345",
            repo="django/django",
            base_commit="abc123",
            test_patch="",
            fail_to_pass=[],
            pass_to_pass=[],
            problem_statement="Fix it",
        )

        sb_task = _pool_task(task, "django__django-12345__sample_1_9")

        assert sb_task.instance_id == "django__django-12345__sample_1_9"
        assert sb_task.docker_image == "sweb.eval.x86_64.django__django-12345:latest"
        assert sb_task.problem_statement == task.problem_statement

    def test_batched_rollout_calls_vllm_once_with_all_active_prompts(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda value: value))

        class DummyGenerate:
            def __init__(self):
                self.batch_sizes = []

            def remote(self, prompts, n_samples, max_new_tokens, temperature):
                self.batch_sizes.append(len(prompts))
                return [
                    RolloutResult(
                        prompt_ids=torch.tensor([100 + i]),
                        response_ids=torch.tensor([200 + i]),
                        full_ids=torch.tensor([100 + i, 200 + i]),
                        old_log_probs=torch.tensor([-0.1 * (i + 1)]),
                        response_text="<done/>",
                        prompt_text=prompt,
                        prompt_len=1,
                        response_len=1,
                    )
                    for i, prompt in enumerate(prompts)
                ]

        class DummySandbox:
            def execute(self, cmd):
                return SandboxResult(stdout="", stderr="", exit_code=0)

        generate = DummyGenerate()
        executor = MultiTurnExecutor.__new__(MultiTurnExecutor)
        executor.rollout_engine = SimpleNamespace(generate=generate)
        executor.sandbox_pool = SimpleNamespace(
            get_sandbox=lambda task: DummySandbox(),
            release_sandbox=lambda _id: None,
        )
        executor.config = SimpleNamespace(
            swe=SimpleNamespace(max_turns=1, max_concurrent_sandboxes=8),
            rollout=SimpleNamespace(max_new_tokens=8, temperature=0.7),
        )

        monkeypatch.setattr(
            "leanrl.agent.multi_turn.compute_swe_reward",
            lambda sandbox, task, stats: (1.0, {}),
        )

        tasks = [
            TaskInstance(
                instance_id=f"repo-{i}",
                repo="repo",
                base_commit="abc123",
                test_patch="",
                fail_to_pass=[],
                pass_to_pass=[],
                problem_statement=f"Fix it {i}",
            )
            for i in range(3)
        ]

        rollouts, rewards = executor._run_batched_rollouts(tasks, n_samples=1, tokenizer=None)

        assert generate.batch_sizes == [3]
        assert len(rollouts) == 3
        assert rewards == [1.0, 1.0, 1.0]


class TestPolicyLogprobRefresh:
    def test_refresh_old_logprobs_uses_policy_model(self):
        class DummyPolicy:
            def __init__(self):
                self.calls = []

            def forward_logprobs_no_grad(
                self,
                input_ids,
                attention_mask,
                response_lengths,
                max_resp_len,
            ):
                self.calls.append(
                    {
                        "input_ids": input_ids.clone(),
                        "attention_mask": attention_mask.clone(),
                        "response_lengths": response_lengths.clone(),
                        "max_resp_len": max_resp_len,
                    }
                )
                return torch.tensor(
                    [[-1.0, -2.0], [-3.0, -4.0]],
                    dtype=torch.float32,
                )

        policy = DummyPolicy()
        executor = MultiTurnExecutor.__new__(MultiTurnExecutor)
        executor.policy_model = policy
        executor.config = SimpleNamespace(training=SimpleNamespace(micro_batch_size=2))

        rollouts = [
            RolloutResult(
                prompt_ids=torch.tensor([10, 11]),
                response_ids=torch.tensor([12, 13]),
                full_ids=torch.tensor([10, 11, 12, 13]),
                old_log_probs=torch.zeros(2),
                response_text="a",
                prompt_text="p1",
                prompt_len=2,
                response_len=2,
            ),
            RolloutResult(
                prompt_ids=torch.tensor([20]),
                response_ids=torch.tensor([21]),
                full_ids=torch.tensor([20, 21]),
                old_log_probs=torch.zeros(1),
                response_text="b",
                prompt_text="p2",
                prompt_len=1,
                response_len=1,
            ),
        ]

        executor._refresh_old_logprobs(
            rollouts,
            tokenizer=SimpleNamespace(pad_token_id=0),
        )

        assert len(policy.calls) == 1
        assert policy.calls[0]["input_ids"].tolist() == [
            [10, 11, 12, 13],
            [20, 21, 0, 0],
        ]
        assert policy.calls[0]["attention_mask"].tolist() == [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ]
        assert policy.calls[0]["response_lengths"].tolist() == [2, 1]
        assert policy.calls[0]["max_resp_len"] == 2
        assert rollouts[0].old_log_probs.tolist() == [-1.0, -2.0]
        assert rollouts[1].old_log_probs.tolist() == [-3.0]
