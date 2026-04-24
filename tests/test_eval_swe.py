import json
import sys

import eval_swe
from eval_swe import build_task
from leanrl.agent.sandbox import TaskInstance


def test_build_task_returns_task_instance():
    row = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc123",
        "test_patch": "diff --git a/x b/x",
        "FAIL_TO_PASS": ["tests/test_foo.py::test_bar"],
        "PASS_TO_PASS": ["tests/test_foo.py::test_existing"],
        "problem_statement": "Fix the bug in views.py",
    }

    task = build_task(row)

    assert isinstance(task, TaskInstance)
    assert task.instance_id == "django__django-12345"
    assert task.repo == "django/django"
    assert task.base_commit == "abc123"
    assert task.test_patch == "diff --git a/x b/x"
    assert task.fail_to_pass == ["tests/test_foo.py::test_bar"]
    assert task.pass_to_pass == ["tests/test_foo.py::test_existing"]
    assert task.problem_statement == "Fix the bug in views.py"


def test_build_task_parses_json_encoded_test_lists():
    row = {
        "instance_id": "sympy__sympy-1",
        "repo": "sympy/sympy",
        "base_commit": "def456",
        "test_patch": "",
        "FAIL_TO_PASS": json.dumps(["test_bugfix"]),
        "PASS_TO_PASS": json.dumps(["test_existing"]),
        "problem_statement": "Fix a SymPy regression",
    }

    task = build_task(row)

    assert task.fail_to_pass == ["test_bugfix"]
    assert task.pass_to_pass == ["test_existing"]


def test_build_task_uses_defaults_for_optional_fields():
    row = {
        "instance_id": "astropy__astropy-1",
        "problem_statement": "Fix the issue",
    }

    task = build_task(row)

    assert task.repo == ""
    assert task.base_commit == ""
    assert task.test_patch == ""
    assert task.fail_to_pass == []
    assert task.pass_to_pass == []


class _DummyDataset(list):
    def shuffle(self, seed):
        return self

    def select(self, indices):
        return _DummyDataset([self[i] for i in list(indices)])


def test_main_uses_training_aligned_swe_defaults(monkeypatch):
    dataset = _DummyDataset([
        {"instance_id": "repo-1", "problem_statement": "Fix the bug"},
    ])
    called = {}

    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_swe.py", "--model_name_or_path", "stub-model", "--output_json", ""],
    )
    monkeypatch.setattr(eval_swe, "load_dataset", lambda dataset_name, split: dataset)
    monkeypatch.setattr(eval_swe, "build_task", lambda row: row)

    def fake_evaluate(**kwargs):
        called.update(kwargs)
        return 0.0, [{"resolved": False}]

    monkeypatch.setattr(eval_swe, "evaluate", fake_evaluate)

    eval_swe.main()

    assert called["model_path"] == "stub-model"
    assert called["max_turns"] == 15
    assert called["max_new_tokens"] == 1024
