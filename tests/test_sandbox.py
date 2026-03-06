"""Tests for the Docker sandbox manager (mock-based)."""

import pytest
from unittest.mock import MagicMock, patch
from leanrl.agent.sandbox import DockerSandbox, TaskInstance, SandboxResult


@pytest.fixture
def sample_task():
    return TaskInstance(
        instance_id="django__django-12345",
        repo="django/django",
        base_commit="abc123",
        test_patch="",
        fail_to_pass=["tests/test_foo.py::test_bar"],
        pass_to_pass=["tests/test_foo.py::test_existing"],
        problem_statement="Fix the bug in views.py",
    )


class TestTaskInstance:
    def test_fields(self, sample_task):
        assert sample_task.instance_id == "django__django-12345"
        assert sample_task.repo == "django/django"

    def test_defaults(self):
        task = TaskInstance(
            instance_id="test",
            repo="test/repo",
            base_commit="abc",
            test_patch="",
            fail_to_pass=[],
            pass_to_pass=[],
            problem_statement="test",
        )
        assert task.docker_image is None


class TestDockerSandbox:
    def test_image_name_default(self, sample_task):
        sandbox = DockerSandbox(sample_task)
        assert "swebench" in sandbox.image_name
        assert "django__django" in sandbox.image_name

    def test_image_name_custom(self, sample_task):
        sample_task.docker_image = "custom/image:v1"
        sandbox = DockerSandbox(sample_task)
        assert sandbox.image_name == "custom/image:v1"


class TestSandboxResult:
    def test_fields(self):
        result = SandboxResult(stdout="ok", stderr="", exit_code=0)
        assert result.stdout == "ok"
        assert not result.timed_out

    def test_timeout(self):
        result = SandboxResult(stdout="", stderr="timeout", exit_code=-1, timed_out=True)
        assert result.timed_out


class TestParseAction:
    def test_bash_action(self):
        from leanrl.agent.multi_turn import parse_action, ACTION_BASH

        action_type, content = parse_action("<bash>ls -la</bash>")
        assert action_type == ACTION_BASH
        assert content == "ls -la"

    def test_edit_action(self):
        from leanrl.agent.multi_turn import parse_action, ACTION_EDIT

        text = '<edit path="foo.py">print("hello")</edit>'
        action_type, content = parse_action(text)
        assert action_type == ACTION_EDIT
        assert "foo.py" in content

    def test_done_action(self):
        from leanrl.agent.multi_turn import parse_action, ACTION_DONE

        action_type, _ = parse_action("<done/>")
        assert action_type == ACTION_DONE

    def test_fallback_bash(self):
        from leanrl.agent.multi_turn import parse_action, ACTION_BASH

        action_type, content = parse_action("git diff")
        assert action_type == ACTION_BASH
        assert content == "git diff"


class TestMathReward:
    def test_gsm8k_extraction(self):
        from leanrl.reward.math_reward import extract_gsm8k_answer

        assert extract_gsm8k_answer("#### 42") == "42"
        assert extract_gsm8k_answer("The answer is 3.14") == "3.14"
        assert extract_gsm8k_answer("\\boxed{100}") == "100"
        assert extract_gsm8k_answer("So we get 1,234") == "1234"

    def test_math_rewards(self):
        from leanrl.reward.math_reward import compute_math_rewards

        rewards = compute_math_rewards(
            responses=["The answer is 42", "The answer is 99", "\\boxed{7}"],
            labels=["#### 42", "#### 100", "#### 7"],
        )
        assert rewards[0].item() == 1.0
        assert rewards[1].item() == 0.0
        assert rewards[2].item() == 1.0


class TestSWEReward:
    def test_parse_pytest_results(self):
        from leanrl.reward.swe_reward import parse_pytest_results

        output = """
tests/test_foo.py::test_bar PASSED
tests/test_foo.py::test_baz FAILED
tests/test_foo.py::test_qux ERROR
"""
        results = parse_pytest_results(output)
        assert results["tests/test_foo.py::test_bar"] == "passed"
        assert results["tests/test_foo.py::test_baz"] == "failed"
        assert results["tests/test_foo.py::test_qux"] == "error"
