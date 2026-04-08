"""Docker sandbox manager for SWE-bench task execution.

Manages a pool of Docker containers, each providing an isolated environment
for running code edits and test suites against SWE-bench task instances.
"""

from __future__ import annotations

import re
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from leanrl.utils.logging import logger


@dataclass
class SandboxResult:
    """Result from executing a command inside a Docker sandbox."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


@dataclass
class TaskInstance:
    """A SWE-bench task instance with metadata needed for sandbox setup."""

    instance_id: str
    repo: str
    base_commit: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    problem_statement: str
    docker_image: Optional[str] = None


class DockerSandbox:
    """Manages a single Docker container for executing SWE-bench actions."""

    def __init__(
        self,
        task: TaskInstance,
        image_prefix: str = "swebench",
        timeout: int = 300,
        memory_limit: str = "4g",
        cpu_limit: float = 2.0,
    ):
        self.task = task
        self.timeout = timeout
        self._container = None
        self._image_prefix = image_prefix
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit

    @property
    def image_name(self) -> str:
        if self.task.docker_image:
            return self.task.docker_image
        # swebench 4.x instance images: sweb.eval.x86_64.{instance_id}:latest
        return f"sweb.eval.x86_64.{self.task.instance_id}:latest"

    def start(self):
        """Create and start the Docker container."""
        import docker

        client = docker.from_env()

        try:
            self._container = client.containers.run(
                self.image_name,
                command="sleep infinity",
                detach=True,
                mem_limit=self._memory_limit,
                nano_cpus=int(self._cpu_limit * 1e9),
                network_disabled=True,
                remove=True,
                stdin_open=True,
                tty=True,
            )
            logger.info(f"Sandbox started: {self.task.instance_id} ({self._container.short_id})")
        except Exception as e:
            logger.error(f"Failed to start sandbox for {self.task.instance_id}: {e}")
            raise

    def execute(self, command: str) -> SandboxResult:
        """Execute a shell command inside the container.

        Args:
            command: shell command string to execute.

        Returns:
            SandboxResult with stdout, stderr, exit_code.
        """
        if self._container is None:
            raise RuntimeError("Sandbox not started")

        # docker-py's exec_run has no timeout parameter; enforce via shell timeout(1).
        # Exit code 124 means the timeout was reached.
        # Prepend the SWE-bench conda env so 'python', 'pytest', etc. resolve correctly.
        env_prefix = "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH && "
        wrapped = f"timeout {self.timeout} bash -c {shlex.quote(env_prefix + command)}"

        try:
            exit_code, output = self._container.exec_run(
                ["bash", "-c", wrapped],
                demux=True,
            )
            stdout = (output[0] or b"").decode("utf-8", errors="replace")
            stderr = (output[1] or b"").decode("utf-8", errors="replace")
            timed_out = exit_code == 124
            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=f"Execution error: {e}",
                exit_code=-1,
                timed_out=False,
            )

    def write_file(self, path: str, content: str) -> SandboxResult:
        """Write content to a file inside the container."""
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        cmd = f"echo '{encoded}' | base64 -d > {shlex.quote(path)}"
        return self.execute(cmd)

    def read_file(self, path: str) -> SandboxResult:
        """Read a file from inside the container."""
        return self.execute(f"cat {path}")

    def apply_patch(self, patch: str) -> SandboxResult:
        """Apply a git diff patch inside the container."""
        self.write_file("/tmp/patch.diff", patch)
        return self.execute("cd /testbed && git apply /tmp/patch.diff")

    def run_tests(self, test_names: Optional[list[str]] = None) -> SandboxResult:
        """Run the test suite (or specific tests) inside the container.

        Detects unittest-style names (``test_foo (module.Class)``) and routes
        them through Django's ``tests/runtests.py``; all other names go to pytest.
        """
        if test_names and any("(" in t for t in test_names):
            return self._run_django_tests(test_names)

        # Bare names with spaces = Django docstring tests with no module context
        if test_names and all("::" not in t for t in test_names) and any(" " in t for t in test_names):
            return self._run_django_bare_tests(test_names)

        # Bare Python identifiers (no spaces, no ::) → sympy-style runner
        if test_names and all("::" not in t for t in test_names):
            return self._run_sympy_tests(test_names)

        if test_names:
            # Do NOT individually shlex.quote here — execute() wraps the entire
            # command in a single-quoted bash -c string, so [param] brackets are
            # already safe.  Extra quoting would produce nested quotes that break
            # pytest argument parsing.
            test_args = " ".join(test_names)
            cmd = f"cd /testbed && python -m pytest {test_args} --tb=short -v 2>&1"
        else:
            cmd = "cd /testbed && python -m pytest --tb=short -v 2>&1"
        result = self.execute(cmd)

        # Exit code 4: pytest collected 0 items because some node IDs don't exist.
        # Extract the not-found IDs, drop them, and retry with the remaining tests.
        if result.exit_code == 4 and test_names:
            not_found = set(re.findall(r"ERROR: not found: \S+::(\S+)", result.stdout))
            surviving = [t for t in test_names if not any(nf in t for nf in not_found)]
            if surviving and len(surviving) < len(test_names):
                test_args2 = " ".join(surviving)
                cmd2 = f"cd /testbed && python -m pytest {test_args2} --tb=short -v 2>&1"
                return self.execute(cmd2)

        return result

    def _run_django_tests(self, test_names: list[str]) -> SandboxResult:
        """Run Django-style unittest tests via tests/runtests.py.

        Runs the top-level module(s) inferred from parseable test names rather
        than individual dotted paths.  This ensures bare-docstring test names
        (which have no ``(module.Class)`` suffix) also appear in the output.
        """
        _pat = re.compile(r"^.+?\s+\(([a-zA-Z0-9_.]+)\)$")
        modules: set[str] = set()
        for name in test_names:
            m = _pat.match(name.strip())
            if m:
                # Use top-level module component (e.g. "test_utils" from "test_utils.tests.Cls")
                modules.add(m.group(1).split(".")[0])

        if modules:
            args = " ".join(shlex.quote(mod) for mod in sorted(modules))
        else:
            # Fallback: pass names directly (will likely error, but best-effort)
            args = " ".join(shlex.quote(t) for t in test_names)

        cmd = f"cd /testbed && PYTHONIOENCODING=utf-8 python tests/runtests.py --verbosity=2 {args} 2>&1"
        return self.execute(cmd)

    def _run_django_bare_tests(self, test_names: list[str]) -> SandboxResult:
        """Run Django bare-docstring tests (no module.Class context) via runtests.py.

        Finds which Django test module contains each docstring by grepping the
        tests/ directory, then runs those modules so the bare-docstring output
        lines are captured and can be parsed.
        """
        modules: set[str] = set()
        for test_name in test_names:
            # Search for the docstring text in Django test files
            safe = test_name.replace("'", r"'\''")  # escape for shell single-quote
            r = self.execute(
                f"grep -rl '{safe}' /testbed/tests --include='*.py' 2>/dev/null | head -3"
            )
            for path in r.stdout.strip().splitlines():
                # /testbed/tests/model_fields/tests.py → model_fields
                rel = path.replace("/testbed/tests/", "").split("/")[0]
                if rel.endswith(".py"):
                    rel = rel[:-3]
                modules.add(rel)

        if not modules:
            return SandboxResult(stdout="", stderr="no modules found", exit_code=1)

        args = " ".join(shlex.quote(m) for m in sorted(modules))
        cmd = f"cd /testbed && PYTHONIOENCODING=utf-8 python tests/runtests.py --verbosity=2 {args} 2>&1"
        return self.execute(cmd)

    def _run_sympy_tests(self, test_names: list[str]) -> SandboxResult:
        """Run sympy bare-name tests via /testbed/bin/test with -k filtering.

        Emits synthetic ``SYMPY_RESULT: <name> PASSED/FAILED`` lines so
        ``parse_pytest_results`` can score them by test name.
        """
        lines: list[str] = []
        for test_name in test_names:
            # Find all sympy test files that define this function
            r_find = self.execute(
                f"grep -rl 'def {test_name}' /testbed/sympy --include='*.py' 2>/dev/null"
            )
            test_files = [
                f.replace("/testbed/", "")
                for f in r_find.stdout.strip().splitlines()
                if f.strip()
            ]
            if not test_files:
                lines.append(f"SYMPY_RESULT: {test_name} FAILED")
                continue

            files_arg = " ".join(test_files)
            r = self.execute(
                f"cd /testbed && bin/test {files_arg} -k {test_name} 2>&1 | tail -5"
            )
            status = "PASSED" if "[OK]" in r.stdout else "FAILED"
            lines.append(f"SYMPY_RESULT: {test_name} {status}")

        return SandboxResult(stdout="\n".join(lines), stderr="", exit_code=0)

    def stop(self):
        """Stop and remove the container."""
        if self._container is not None:
            try:
                self._container.stop(timeout=5)
            except Exception:
                try:
                    self._container.kill()
                except Exception:
                    pass
            self._container = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class SandboxPool:
    """Manages a pool of Docker sandboxes for parallel SWE-bench execution."""

    def __init__(
        self,
        max_concurrent: int = 4,
        image_prefix: str = "swebench",
        timeout: int = 300,
        memory_limit: str = "4g",
        cpu_limit: float = 2.0,
    ):
        self.max_concurrent = max_concurrent
        self.image_prefix = image_prefix
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._active: dict[str, DockerSandbox] = {}
        self._lock = threading.Lock()

    def get_sandbox(self, task: TaskInstance) -> DockerSandbox:
        """Get or create a sandbox for a task instance."""
        with self._lock:
            if task.instance_id in self._active:
                return self._active[task.instance_id]

            sandbox = DockerSandbox(
                task=task,
                image_prefix=self.image_prefix,
                timeout=self.timeout,
                memory_limit=self.memory_limit,
                cpu_limit=self.cpu_limit,
            )
            self._active[task.instance_id] = sandbox

        sandbox.start()
        return sandbox

    def release_sandbox(self, instance_id: str):
        """Stop and remove a sandbox."""
        with self._lock:
            if instance_id in self._active:
                self._active[instance_id].stop()
                del self._active[instance_id]

    def release_all(self):
        """Stop all active sandboxes."""
        for instance_id in list(self._active.keys()):
            self.release_sandbox(instance_id)

    def execute_parallel(
        self,
        tasks_and_commands: list[tuple[TaskInstance, str]],
    ) -> list[SandboxResult]:
        """Execute commands in parallel across sandboxes."""
        futures = []
        for task, command in tasks_and_commands:
            sandbox = self.get_sandbox(task)
            future = self._executor.submit(sandbox.execute, command)
            futures.append(future)

        return [f.result() for f in futures]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release_all()
