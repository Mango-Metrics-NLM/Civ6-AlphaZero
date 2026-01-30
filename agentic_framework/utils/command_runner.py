"""
Command Runner for Agentic Framework

Provides safe, configurable command execution with timeout handling,
output capture, and retry logic.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .logging_config import get_logger

logger = get_logger("command_runner")


class CommandError(Exception):
    """Base exception for command execution errors."""

    def __init__(
        self,
        message: str,
        command: str,
        return_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.command = command
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class CommandTimeoutError(CommandError):
    """Raised when a command times out."""

    pass


class OutputMode(Enum):
    """Output capture modes."""

    CAPTURE = "capture"  # Capture stdout/stderr
    STREAM = "stream"  # Stream to console
    BOTH = "both"  # Capture and stream


@dataclass
class CommandResult:
    """Result of a command execution."""

    command: str
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    success: bool = field(init=False)
    timed_out: bool = False

    def __post_init__(self) -> None:
        """Calculate success after initialization."""
        self.success = self.return_code == 0 and not self.timed_out

    def __bool__(self) -> bool:
        """Return True if command succeeded."""
        return self.success

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "timed_out": self.timed_out,
        }


@dataclass
class RetryConfig:
    """Configuration for command retry behavior."""

    max_retries: int = 0
    retry_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 30.0
    retry_on_codes: list[int] = field(default_factory=lambda: [])
    retry_on_timeout: bool = True


class CommandRunner:
    """Executes shell commands with configurable options.

    Features:
    - Timeout handling
    - Output capture modes
    - Retry logic with exponential backoff
    - Environment variable management
    - Working directory control
    - Signal handling
    """

    def __init__(
        self,
        default_timeout_seconds: float = 300.0,
        default_cwd: Path | None = None,
        default_env: dict[str, str] | None = None,
        shell: bool = True,
        output_mode: OutputMode = OutputMode.CAPTURE,
    ) -> None:
        """Initialize the command runner.

        Args:
            default_timeout_seconds: Default timeout for commands.
            default_cwd: Default working directory.
            default_env: Default environment variables (merged with current env).
            shell: Whether to use shell execution.
            output_mode: Default output capture mode.
        """
        self.default_timeout_seconds = default_timeout_seconds
        self.default_cwd = default_cwd or Path.cwd()
        self.default_env = default_env or {}
        self.shell = shell
        self.output_mode = output_mode

    def _build_env(self, extra_env: dict[str, str] | None = None) -> dict[str, str]:
        """Build environment variables for command execution.

        Args:
            extra_env: Additional environment variables.

        Returns:
            Complete environment dictionary.
        """
        env = os.environ.copy()
        env.update(self.default_env)
        if extra_env:
            env.update(extra_env)
        return env

    def run(
        self,
        command: str,
        timeout_seconds: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        output_mode: OutputMode | None = None,
        retry_config: RetryConfig | None = None,
        check: bool = False,
    ) -> CommandResult:
        """Execute a command synchronously.

        Args:
            command: Command to execute.
            timeout_seconds: Timeout in seconds (None = use default).
            cwd: Working directory (None = use default).
            env: Extra environment variables.
            output_mode: Output capture mode (None = use default).
            retry_config: Retry configuration.
            check: Raise exception on non-zero exit code.

        Returns:
            CommandResult with execution details.

        Raises:
            CommandError: If check=True and command fails.

        Note:
            Timeouts are handled gracefully via CommandResult.timed_out flag.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        working_dir = cwd or self.default_cwd
        full_env = self._build_env(env)
        mode = output_mode or self.output_mode
        retry = retry_config or RetryConfig()

        attempt = 0
        delay = retry.retry_delay_seconds
        last_result: CommandResult | None = None

        while attempt <= retry.max_retries:
            if attempt > 0:
                logger.info(
                    f"Retrying command (attempt {attempt + 1}/{retry.max_retries + 1}): {command}"
                )
                time.sleep(delay)
                delay = min(delay * retry.backoff_multiplier, retry.max_delay_seconds)

            result = self._execute_once(
                command=command,
                timeout=timeout,
                cwd=working_dir,
                env=full_env,
                mode=mode,
            )
            last_result = result

            # Check if we should retry
            should_retry = False
            if result.timed_out and retry.retry_on_timeout:
                should_retry = True
            elif result.return_code in retry.retry_on_codes:
                should_retry = True

            if not should_retry or attempt >= retry.max_retries:
                break

            attempt += 1

        if last_result is None:
            last_result = CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr="Command did not execute",
                duration_seconds=0.0,
            )

        if check and not last_result.success:
            raise CommandError(
                message=f"Command failed with return code {last_result.return_code}",
                command=command,
                return_code=last_result.return_code,
                stdout=last_result.stdout,
                stderr=last_result.stderr,
            )

        return last_result

    def _execute_once(
        self,
        command: str,
        timeout: float,
        cwd: Path,
        env: dict[str, str],
        mode: OutputMode,
    ) -> CommandResult:
        """Execute a command once without retry.

        Args:
            command: Command to execute.
            timeout: Timeout in seconds.
            cwd: Working directory.
            env: Environment variables.
            mode: Output capture mode.

        Returns:
            CommandResult with execution details.
        """
        logger.debug(f"Executing command: {command}")
        logger.debug(f"Working directory: {cwd}")

        start_time = time.time()
        stdout_data = ""
        stderr_data = ""
        timed_out = False

        try:
            if mode == OutputMode.STREAM:
                # Stream output directly to console
                process = subprocess.Popen(
                    command,
                    shell=self.shell,
                    cwd=str(cwd),
                    env=env,
                    stdout=None,
                    stderr=None,
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._kill_process(process)
                    timed_out = True

                return_code = process.returncode if not timed_out else -1

            else:
                # Capture output
                process = subprocess.Popen(
                    command,
                    shell=self.shell,
                    cwd=str(cwd),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                try:
                    stdout_data, stderr_data = process.communicate(timeout=timeout)

                    if mode == OutputMode.BOTH:
                        if stdout_data:
                            print(stdout_data, end="")
                        if stderr_data:
                            print(stderr_data, end="", file=__import__("sys").stderr)

                except subprocess.TimeoutExpired:
                    self._kill_process(process)
                    stdout_data, stderr_data = process.communicate()
                    timed_out = True

                return_code = process.returncode if not timed_out else -1

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=time.time() - start_time,
            )

        duration = time.time() - start_time

        result = CommandResult(
            command=command,
            return_code=return_code,
            stdout=stdout_data,
            stderr=stderr_data,
            duration_seconds=duration,
            timed_out=timed_out,
        )

        if result.success:
            logger.debug(f"Command completed successfully in {duration:.2f}s")
        elif timed_out:
            logger.warning(f"Command timed out after {timeout}s: {command}")
        else:
            logger.warning(f"Command failed with code {return_code}: {command}")

        return result

    def _kill_process(self, process: subprocess.Popen) -> None:
        """Kill a process and its children.

        Args:
            process: Process to kill.
        """
        try:
            # Try graceful termination first
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                process.kill()
                process.wait()
        except Exception as e:
            logger.error(f"Error killing process: {e}")

    async def run_async(
        self,
        command: str,
        timeout_seconds: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Execute a command asynchronously.

        Args:
            command: Command to execute.
            timeout_seconds: Timeout in seconds.
            cwd: Working directory.
            env: Extra environment variables.

        Returns:
            CommandResult with execution details.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        working_dir = cwd or self.default_cwd
        full_env = self._build_env(env)

        logger.debug(f"Executing async command: {command}")
        start_time = time.time()
        stdout_data = ""
        stderr_data = ""
        timed_out = False
        return_code = -1

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(working_dir),
                env=full_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                stdout_data = stdout_bytes.decode() if stdout_bytes else ""
                stderr_data = stderr_bytes.decode() if stderr_bytes else ""
                return_code = process.returncode or 0

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                timed_out = True

        except Exception as e:
            logger.error(f"Async command execution error: {e}")
            stderr_data = str(e)

        duration = time.time() - start_time

        return CommandResult(
            command=command,
            return_code=return_code,
            stdout=stdout_data,
            stderr=stderr_data,
            duration_seconds=duration,
            timed_out=timed_out,
        )

    def run_multiple(
        self,
        commands: list[str],
        stop_on_failure: bool = False,
        **kwargs: Any,
    ) -> list[CommandResult]:
        """Execute multiple commands sequentially.

        Args:
            commands: List of commands to execute.
            stop_on_failure: Stop execution on first failure.
            **kwargs: Additional arguments passed to run().

        Returns:
            List of CommandResult objects.
        """
        results: list[CommandResult] = []

        for command in commands:
            result = self.run(command, **kwargs)
            results.append(result)

            if stop_on_failure and not result.success:
                logger.warning(f"Stopping execution due to failure: {command}")
                break

        return results

    async def run_multiple_async(
        self,
        commands: list[str],
        max_concurrent: int = 5,
        **kwargs: Any,
    ) -> list[CommandResult]:
        """Execute multiple commands concurrently.

        Args:
            commands: List of commands to execute.
            max_concurrent: Maximum concurrent executions.
            **kwargs: Additional arguments passed to run_async().

        Returns:
            List of CommandResult objects in order.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(cmd: str) -> CommandResult:
            async with semaphore:
                return await self.run_async(cmd, **kwargs)

        tasks = [run_with_semaphore(cmd) for cmd in commands]
        return await asyncio.gather(*tasks)


# Convenience function for simple command execution
def run_command(
    command: str,
    timeout_seconds: float = 300.0,
    cwd: Path | None = None,
    check: bool = False,
) -> CommandResult:
    """Execute a command with sensible defaults.

    Args:
        command: Command to execute.
        timeout_seconds: Timeout in seconds.
        cwd: Working directory.
        check: Raise exception on failure.

    Returns:
        CommandResult with execution details.
    """
    runner = CommandRunner(default_timeout_seconds=timeout_seconds)
    if cwd is not None:
        runner.default_cwd = cwd
    return runner.run(command, check=check)
