"""Docker sandbox manager for SWE-bench task execution.

Manages a pool of Docker containers, each providing an isolated environment
for running code edits and test suites against SWE-bench task instances.
"""

from __future__ import annotations

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

        try:
            exit_code, output = self._container.exec_run(
                ["bash", "-c", command],
                demux=True,
                timeout=self.timeout,
            )
            stdout = (output[0] or b"").decode("utf-8", errors="replace")
            stderr = (output[1] or b"").decode("utf-8", errors="replace")
            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=False,
            )
        except Exception as e:
            error_msg = str(e)
            timed_out = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
            return SandboxResult(
                stdout="",
                stderr=f"Execution error: {error_msg}",
                exit_code=-1,
                timed_out=timed_out,
            )

    def write_file(self, path: str, content: str) -> SandboxResult:
        """Write content to a file inside the container."""
        import base64

        import shlex
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
        """Run the test suite (or specific tests) inside the container."""
        if test_names:
            test_args = " ".join(test_names)
            cmd = f"cd /testbed && python -m pytest {test_args} -x --tb=short 2>&1"
        else:
            cmd = "cd /testbed && python -m pytest --tb=short 2>&1"
        return self.execute(cmd)

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
        sandbox.start()
        with self._lock:
            self._active[task.instance_id] = sandbox
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
