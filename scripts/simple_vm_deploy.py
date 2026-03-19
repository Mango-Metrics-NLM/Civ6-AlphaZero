#!/usr/bin/env python3
"""Simple VM deployment using uploaded scripts."""
import platform
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def get_gcloud_cmd():
    """Get platform-specific gcloud command."""
    if platform.system() == "Windows":
        return r"C:\Users\iansh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    return "gcloud"


def run_ssh(vm_name, zone, project, command):
    """Execute SSH command on VM."""
    gcloud = get_gcloud_cmd()
    result = subprocess.run(
        [gcloud, "compute", "ssh", vm_name, f"--zone={zone}",
         f"--project={project}", f"--command={command}"],
        capture_output=True, text=True
    )
    return result


def main():
    """Deploy orchestrator to VM."""
    project_root = Path(__file__).parent.parent
    gcloud = get_gcloud_cmd()

    # Configuration
    vm_name = "civ6az-orchestrator"
    zone = "us-central1-a"
    project = "hrmtrain"

    print("="*70)
    print("Civ6 AlphaZero - Simple VM Deployment")
    print("="*70)
    print()

    # Step 1: Create and upload codebase
    print("[1/4] Creating codebase archive...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "codebase.tar.gz"

        with tarfile.open(tar_path, "w:gz") as tar:
            for item in ["alpha_zero", "scripts", "requirements-container.txt", "pyproject.toml"]:
                item_path = project_root / item
                if item_path.exists():
                    tar.add(item_path, arcname=f"Civ6-AlphaZero/{item}",
                           filter=lambda ti: None if '__pycache__' in ti.name or '.pyc' in ti.name else ti)

        print(f"  Archive: {tar_path.stat().st_size / 1e6:.1f} MB")
        print("  Uploading to VM...")

        result = subprocess.run([
            gcloud, "compute", "scp", str(tar_path),
            f"{vm_name}:codebase.tar.gz", f"--zone={zone}", f"--project={project}"
        ], capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"  [FAIL] {result.stderr}")
            return 1

    print("  [OK] Codebase uploaded")

    # Step 2: Clean up existing installation
    print("[2/5] Cleaning up existing installation...")
    result = run_ssh(vm_name, zone, project, "sudo rm -rf Civ6-AlphaZero")
    if result.returncode != 0:
        print(f"  [WARN] Cleanup had issues (may not exist): {result.stderr}")
    else:
        print("  [OK] Cleanup complete")

    # Step 3: Extract codebase
    print("[3/5] Extracting codebase on VM...")
    result = run_ssh(vm_name, zone, project, "tar -xzf codebase.tar.gz && rm codebase.tar.gz")
    if result.returncode != 0:
        print(f"  [FAIL] {result.stderr}")
        return 1
    print("  [OK] Extracted")

    # Step 3.5: Fix ownership (files might have been owned by root previously)
    print("  Fixing file permissions...")
    result = run_ssh(vm_name, zone, project, "sudo chown -R $(whoami):$(whoami) Civ6-AlphaZero")
    if result.returncode != 0:
        print(f"  [WARN] Ownership fix had issues: {result.stderr}")

    # Step 4: Upload install script (convert line endings to Unix)
    print("[4/5] Uploading install script...")
    install_script = project_root / "scripts" / "vm_install.sh"

    # Convert CRLF to LF for bash compatibility
    with tempfile.TemporaryDirectory() as tmpdir:
        unix_script = Path(tmpdir) / "install.sh"
        with open(install_script, 'rb') as src:
            content = src.read()
        # Explicitly convert Windows CRLF (\r\n) to Unix LF (\n)
        content = content.replace(b'\r\n', b'\n')
        with open(unix_script, 'wb') as dst:
            dst.write(content)

        result = subprocess.run([
            gcloud, "compute", "scp", str(unix_script),
            f"{vm_name}:install.sh", f"--zone={zone}", f"--project={project}"
        ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [FAIL] {result.stderr}")
        return 1
    print("  [OK] Script uploaded")

    # Step 5: Execute install script
    print("[5/5] Running installation (this may take 3-5 minutes)...")
    print("  Installing dependencies and launching orchestrator...")

    result = run_ssh(vm_name, zone, project, "bash install.sh")

    if result.returncode == 0:
        print()
        print("="*70)
        print("[OK] Deployment Complete!")
        print("="*70)
        print()
        print(result.stdout)
        print()
        print("Monitor Progress:")
        print("  W&B: https://wandb.ai/ianshank-none/civ6-alphazero")
        print()
        print("Check Status:")
        print(f"  {gcloud} compute ssh {vm_name} --zone={zone}")
        print("  tail -f ~/orchestrator.log")
        print("  ps aux | grep vertex_orchestrator")
        print()
        return 0
    else:
        print()
        print("="*70)
        print("[FAIL] Deployment Failed")
        print("="*70)
        print(f"Error: {result.stderr}")
        print(f"Output: {result.stdout}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
