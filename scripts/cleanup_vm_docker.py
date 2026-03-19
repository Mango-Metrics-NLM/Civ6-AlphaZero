#!/usr/bin/env python3
"""Clean up Docker containers and images on VM to free disk space."""
import subprocess
import platform

def main():
    gcloud = r"C:\Users\iansh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" if platform.system() == "Windows" else "gcloud"

    print("=" * 70)
    print("Cleaning up VM Docker artifacts")
    print("=" * 70)
    print()

    # Remove stopped containers
    print("[1/3] Removing stopped Docker containers...")
    result = subprocess.run([
        gcloud, "compute", "ssh", "civ6az-orchestrator",
        "--zone=us-central1-a",
        "--project=hrmtrain",
        "--command=docker rm -f $(docker ps -aq) 2>/dev/null || true"
    ], capture_output=True, text=True)
    if result.stdout:
        print(f"  Removed: {result.stdout.strip()}")
    print("  [OK]")

    # Remove all images
    print("[2/3] Removing Docker images...")
    result = subprocess.run([
        gcloud, "compute", "ssh", "civ6az-orchestrator",
        "--zone=us-central1-a",
        "--project=hrmtrain",
        "--command=docker rmi -f $(docker images -q) 2>/dev/null || true"
    ], capture_output=True, text=True)
    if result.stdout:
        print(f"  {result.stdout.strip()}")
    print("  [OK]")

    # Check new disk usage
    print("[3/3] Checking disk space...")
    result = subprocess.run([
        gcloud, "compute", "ssh", "civ6az-orchestrator",
        "--zone=us-central1-a",
        "--project=hrmtrain",
        "--command=df -h /mnt/stateful_partition"
    ], capture_output=True, text=True)
    print(result.stdout)

    print("=" * 70)
    print("[OK] Cleanup complete!")
    print("=" * 70)

    return 0

if __name__ == "__main__":
    exit(main())
