"""SSH to the orchestrator VM and run a command.

Usage:
    python scripts/vm_ssh.py "echo hello"
    python scripts/vm_ssh.py "ls -la ~/Civ6-AlphaZero/"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

GCLOUD = (
    os.environ.get("CIV6AZ_GCLOUD_BIN")
    or shutil.which("gcloud")
    or shutil.which("gcloud.cmd")
    or "gcloud"
)
VM_NAME = os.environ.get("CIV6AZ_ORCHESTRATOR_VM", "civ6az-orchestrator")
ZONE = os.environ.get("CIV6AZ_VERTEX_ZONE", "us-central1-a")
PROJECT = os.environ.get("CIV6AZ_VERTEX_PROJECT")
if not PROJECT:
    raise SystemExit("CIV6AZ_VERTEX_PROJECT is not set (there is no default project).")


def ssh_command(command: str, timeout: int = 120) -> tuple[str, str, int]:
    """Run a command on the VM via SSH.

    Args:
        command: Shell command to execute on the VM.
        timeout: SSH timeout in seconds.

    Returns:
        Tuple of (stdout, stderr, return_code).
    """
    result = subprocess.run(
        [
            GCLOUD,
            "compute",
            "ssh",
            VM_NAME,
            f"--zone={ZONE}",
            f"--project={PROJECT}",
            "--tunnel-through-iap",
            f"--command={command}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def main() -> None:
    """Run SSH command from CLI argument."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/vm_ssh.py <command>")
        sys.exit(1)

    command = sys.argv[1]
    print(f"Running on {VM_NAME}: {command}")
    stdout, stderr, rc = ssh_command(command)
    if stdout:
        print(stdout)
    if rc != 0:
        print(f"STDERR: {stderr[:500]}")
        print(f"Return code: {rc}")


if __name__ == "__main__":
    main()
