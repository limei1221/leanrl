"""Tests for SWE multi-turn action parsing and token masking."""

import pytest
from leanrl.agent.multi_turn import (
    parse_action,
    ACTION_BASH,
    ACTION_DONE,
    _append_prompt_delta,
)


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
        text = '```python\nimport os\nprint(os.getcwd())\n```'
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


class TestPromptDeltaRealTokenizer:
    """Round-trip smoke test with a real Qwen chat template + tokenizer.

    Exercises the path that breaks in production: decode(gen_ids) placed back
    into apply_chat_template should retokenize to the original gen_ids.
    """

    def test_clean_round_trip_across_turns(self):
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        except Exception as e:
            pytest.skip(f"Qwen tokenizer unavailable: {e}")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix the bug in foo.py."},
        ]
        initial_prompt = list(tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ))

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
        prompt_t1 = list(tok.apply_chat_template(
            messages_t1, tokenize=True, add_generation_prompt=True
        ))

        ok = _append_prompt_delta(
            response_ids=response_ids,
            old_log_probs=old_log_probs,
            response_mask=response_mask,
            new_prompt_ids=prompt_t1,
            initial_prompt_ids=initial_prompt,
        )

        assert ok, (
            "Plain-ASCII assistant content should round-trip through the Qwen "
            "tokenizer; drift here indicates a real regression."
        )
        delta_len = len(response_ids) - len(gen_ids_0)
        assert delta_len > 0
        assert response_mask[-delta_len:] == [0.0] * delta_len
        assert old_log_probs[-delta_len:] == [0.0] * delta_len
        # Full accumulated stream equals the re-rendered turn-1 prompt.
        assert initial_prompt + response_ids == prompt_t1
