"""Tests for SWE multi-turn action parsing (bash-only scaffold)."""

import pytest
from leanrl.agent.multi_turn import parse_action, ACTION_BASH, ACTION_DONE


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
