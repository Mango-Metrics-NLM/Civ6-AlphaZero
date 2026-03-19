#!/usr/bin/env python3
"""Check where /home is mounted on VM."""
import subprocess
import platform

def main():
    gcloud = r"C:\Users\iansh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" if platform.system() == "Windows" else "gcloud"

    cmd = [
        gcloud, "compute", "ssh", "civ6az-orchestrator",
        "--zone=us-central1-a",
        "--project=hrmtrain",
        "--command=mount | grep -E '(home|stateful)' && echo '---' && df -h / /home && echo '---' && du -sh /home/iansh/* 2>/dev/null || true"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode

if __name__ == "__main__":
    exit(main())
