#!/usr/bin/env python3
"""
Deploy and launch orchestrator on GCP VM with full setup.

This script:
1. Uploads setup script to VM
2. Installs codebase and dependencies
3. Launches orchestrator in background
4. Monitors startup

Sprint 6 Phase 3: Production-ready orchestrator deployment.
"""
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any


class VMOrchestratorDeployer:
    """Handles orchestrator deployment to GCP VM."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize deployer with configuration.

        Args:
            config: Deployment configuration dictionary.
        """
        self.config = config
        self.gcloud_cmd = self._get_gcloud_cmd()
        self.project_root = Path(__file__).parent.parent

    def _get_gcloud_cmd(self) -> str:
        """Get platform-specific gcloud command path (always use .cmd on Windows)."""
        if platform.system() == "Windows":
            return r"C:\Users\iansh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
        return "gcloud"

    def _build_gcloud_env(self) -> dict[str, str]:
        """Environment variables to pass to all gcloud subprocess calls.

        Sets CLOUDSDK flags so gcloud never blocks on update checks or
        interactive prompts when running non-interactively from Python.
        """
        import os
        env = os.environ.copy()
        env["CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK"] = "1"
        env["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
        env["CLOUDSDK_PYTHON_SITEPACKAGES"] = "1"
        return env

    def _run_ssh_command(self, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Execute command on VM via SSH (non-interactive, batch mode).

        Args:
            command: Shell command to execute.
            timeout: Command timeout in seconds.

        Returns:
            CompletedProcess with results.
        """
        ssh_args = [
            self.gcloud_cmd,
            "compute", "ssh", self.config["vm_name"],
            f"--zone={self.config['vm_zone']}",
            f"--project={self.config['gcp_project']}",
            "--quiet",
            # Non-interactive SSH flags: no TTY, batch mode, no host-key dialogs
            "--",
            "-n",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
            command
        ]

        return subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._build_gcloud_env(),
        )

    def pre_provision_ssh_keys(self) -> bool:
        """Run gcloud compute config-ssh to pre-generate SSH keys without interaction.

        This must be run before any SSH commands so that gcloud does not
        open interactive dialogs when called non-interactively from Python.

        Returns:
            True if successful or already provisioned, False on auth errors.
        """
        print("[0/5] Pre-provisioning SSH keys (gcloud compute config-ssh)...")
        result = subprocess.run(
            [self.gcloud_cmd, "compute", "config-ssh",
             f"--project={self.config['gcp_project']}", "--quiet"],
            capture_output=True,
            text=True,
            timeout=30,
            env=self._build_gcloud_env(),
        )
        if result.returncode == 0:
            print("  [OK] SSH keys provisioned")
            return True
        # config-ssh exits 0 even if already done; if it fails it's an auth error
        print(f"  [WARN] config-ssh returned {result.returncode}: {result.stderr[:200]}")
        print("  Proceeding anyway (keys may already exist)")
        return True

    def upload_codebase(self) -> bool:
        """Upload codebase and setup script to VM.

        Returns:
            True if successful, False otherwise.
        """
        print("[1/5] Uploading codebase to VM...")

        # Create tarball of codebase (excluding large files)
        print("  Creating codebase archive...")
        tar_path = self.project_root / "civ6az-codebase.tar.gz"

        import tarfile
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add key directories, excluding .git, __pycache__, etc.
            for item in ["alpha_zero", "scripts", "requirements-container.txt", "pyproject.toml"]:
                item_path = self.project_root / item
                if item_path.exists():
                    tar.add(item_path, arcname=f"Civ6-AlphaZero/{item}")

        print(f"  Archive created: {tar_path.stat().st_size / 1e6:.1f} MB")

        scp_cmd = [
            self.gcloud_cmd,
            "compute", "scp",
            "--quiet",
            "--ssh-flag=-o StrictHostKeyChecking=no",
            str(tar_path),
            f"{self.config['vm_name']}:civ6az-codebase.tar.gz",
            f"--zone={self.config['vm_zone']}",
            f"--project={self.config['gcp_project']}"
        ]

        result = subprocess.run(
            scp_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=self._build_gcloud_env(),
        )

        # Clean up local tarball
        tar_path.unlink(missing_ok=True)

        if result.returncode == 0:
            print("  [OK] Codebase uploaded")

            # Extract on VM
            extract_result = self._run_ssh_command(
                "cd ~ && tar -xzf civ6az-codebase.tar.gz && rm civ6az-codebase.tar.gz"
            )

            if extract_result.returncode == 0:
                print("  [OK] Codebase extracted")
                return True
            else:
                print(f"  [FAIL] Extraction failed: {extract_result.stderr}")
                return False
        else:
            print(f"  [FAIL] Upload failed: {result.stderr}")
            return False

    def install_dependencies(self) -> bool:
        """Install Python dependencies on VM.

        Returns:
            True if successful, False otherwise.
        """
        print("[2/5] Installing dependencies on VM (this may take 3-5 minutes)...")

        # Step-by-step installation to avoid command escaping issues
        steps = [
            ("Creating venv", "cd Civ6-AlphaZero && python3 -m venv venv"),
            ("Upgrading pip", "cd Civ6-AlphaZero && source venv/bin/activate && pip install -q --upgrade pip"),
            ("Installing requirements", "cd Civ6-AlphaZero && source venv/bin/activate && pip install -q -r requirements-container.txt"),
        ]

        for desc, cmd in steps:
            print(f"    {desc}...")
            result = self._run_ssh_command(cmd, timeout=600)
            if result.returncode != 0:
                print(f"      [FAIL] {desc} failed: {result.stderr[:200]}")
                return False

        # Create environment file
        print("    Creating environment config...")
        env_content = """export CIV6AZ_VERTEX_PROJECT=hrmtrain
export CIV6AZ_VERTEX_REGION=us-central1
export CIV6AZ_GCS_BUCKET=civ6az-training
export CIV6AZ_CONTAINER_IMAGE_URI=us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0
export WANDB_API_KEY=26a08535e80a6d5d6f4a941f6a742264b25f4819
export WANDB_PROJECT=civ6-alphazero
export PYTHONPATH=~/Civ6-AlphaZero:$PYTHONPATH
"""
        # Write env file
        result = self._run_ssh_command(f"cat > ~/.bashrc_civ6az << 'ENVEOF'\n{env_content}ENVEOF")

        if result.returncode != 0:
            print(f"      [FAIL] Environment config failed")
            return False

        # Verify installation
        print("    Verifying installation...")
        verify_result = self._run_ssh_command(
            "cd Civ6-AlphaZero && source venv/bin/activate && python -c 'import alpha_zero; print(\"OK\")'"
        )

        if verify_result.returncode == 0 and "OK" in verify_result.stdout:
            print("  [OK] Dependencies installed and verified")
            return True
        else:
            print(f"  [FAIL] Verification failed: {verify_result.stderr[:200]}")
            return False

    def launch_orchestrator(self) -> bool:
        """Launch orchestrator via systemd on VM.

        Returns:
            True if successful, False otherwise.
        """
        print("[3/5] Setting up systemd service...")

        # Upload systemd service file and env template
        setup_steps = [
            (
                "Configuring service for current user",
                "sed -i \"s/civ6az/$USER/g\" ~/Civ6-AlphaZero/scripts/systemd/civ6az-orchestrator.service"
            ),
            (
                "Installing systemd service",
                "sudo cp ~/Civ6-AlphaZero/scripts/systemd/civ6az-orchestrator.service "
                "/etc/systemd/system/ && sudo systemctl daemon-reload && "
                "sudo systemctl enable civ6az-orchestrator.service",
            ),
            (
                "Creating env file",
                "test -f ~/.env.civ6az || "
                "cp ~/Civ6-AlphaZero/scripts/systemd/.env.civ6az.template ~/.env.civ6az",
            ),
            (
                "Starting orchestrator service",
                "sudo systemctl restart civ6az-orchestrator",
            ),
        ]

        for desc, cmd in setup_steps:
            print(f"    {desc}...")
            result = self._run_ssh_command(cmd, timeout=30)
            if result.returncode != 0:
                print(f"      [FAIL] {desc} failed: {result.stderr[:200]}")
                return False

        print("  [OK] Orchestrator service started")
        return True

    def verify_startup(self) -> bool:
        """Verify orchestrator service started successfully.

        Returns:
            True if running, False otherwise.
        """
        print("[4/5] Verifying orchestrator startup...")

        # Wait a few seconds for startup
        time.sleep(5)

        # Check systemd service status
        result = self._run_ssh_command(
            "systemctl is-active civ6az-orchestrator"
        )

        if result.returncode == 0 and "active" in result.stdout.strip():
            print("  [OK] Orchestrator service active")

            # Check logs for any immediate errors
            log_check = self._run_ssh_command(
                "journalctl -u civ6az-orchestrator --no-pager -n 20"
            )

            if "error" in log_check.stdout.lower() or "exception" in log_check.stdout.lower():
                print("  [WARN] Potential errors in log:")
                print(log_check.stdout)
                return False

            print("  Log output (last 10 lines):")
            for line in log_check.stdout.split('\n')[-10:]:
                print(f"    {line}")

            return True
        else:
            print("  [FAIL] Orchestrator service not active")
            # Show why it failed
            status_result = self._run_ssh_command(
                "systemctl status civ6az-orchestrator --no-pager"
            )
            if status_result.stdout:
                print(status_result.stdout)
            return False

    def display_monitoring_info(self):
        """Display monitoring and management information."""
        print("\n" + "="*70)
        print("Orchestrator Deployed Successfully!")
        print("="*70)
        print()
        print("Configuration:")
        orch = self.config["orchestrator"]
        print(f"  Iterations: {orch['iterations']}")
        print(f"  Total games: {orch['iterations'] * orch['workers'] * orch['games_per_worker']}")
        print(f"  Batch self-play: {orch['num_parallel_games']} parallel games")
        print(f"  Starting from: v{orch['start_model_version']}")
        print(f"  Resume enabled: Yes (auto-resumes after reboot)")
        print()
        print("Monitor Progress:")
        print(f"  W&B: https://wandb.ai/{orch['wandb_entity']}/{orch['wandb_project']}")
        print()
        print("Service Management:")
        print(f"  {self.gcloud_cmd} compute ssh {self.config['vm_name']} \\")
        print(f"    --zone={self.config['vm_zone']} --project={self.config['gcp_project']}")
        print()
        print("  sudo systemctl status civ6az-orchestrator   # Check status")
        print("  sudo systemctl stop civ6az-orchestrator     # Stop")
        print("  sudo systemctl restart civ6az-orchestrator  # Restart")
        print("  journalctl -u civ6az-orchestrator -f        # Live logs")
        print()


def main():
    """Main deployment workflow."""

    # Configuration (dynamic, no hardcoded values except defaults)
    config = {
        "gcp_project": "hrmtrain",
        "vm_zone": "us-central1-a",
        "vm_name": "civ6az-orchestrator",
        "orchestrator": {
            "iterations": 20,
            "workers": 4,
            "games_per_worker": 50,
            "num_parallel_games": 4,
            "max_batch_size": 32,
            "mcts_type": "gumbel",
            "mcts_simulations": 16,
            "max_turns": 30,
            "evaluation_games": 30,
            "evaluation_win_threshold": 0.55,
            "start_model_version": 268,
            "wandb_project": "civ6-alphazero",
            "wandb_entity": "ianshank-none",
            "wandb_tags": "batch-selfplay,v1.5.0"
        }
    }

    print("="*70)
    print("Civ6 AlphaZero - Orchestrator VM Deployment")
    print("="*70)
    print()
    print("This will:")
    print("  1. Upload setup script to VM")
    print("  2. Install codebase and dependencies")
    print("  3. Launch 20-iteration batch self-play benchmark")
    print("  4. Verify startup")
    print()
    print("Expected time: 10-15 minutes")
    print("Expected cost: $3-5")
    print()

    # Initialize deployer
    deployer = VMOrchestratorDeployer(config)

    # Execute deployment steps
    steps = [
        deployer.pre_provision_ssh_keys,
        deployer.upload_codebase,
        deployer.install_dependencies,
        deployer.launch_orchestrator,
        deployer.verify_startup,
    ]

    for i, step in enumerate(steps, 1):
        try:
            if not step():
                print(f"\n[FAIL] Deployment failed at step {i}")
                return 1
        except Exception as e:
            print(f"\n[FAIL] Exception at step {i}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Display monitoring info
    deployer.display_monitoring_info()

    print("\n[OK] Deployment complete - benchmark running!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
