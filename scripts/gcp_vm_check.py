#!/usr/bin/env python3
"""
GCP VM status check + create VM if missing.

Uses Python SDK only (no gcloud subprocess = no hangs).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

# ── Config (env-driven, no hardcoded values) ────────────────────────────
PROJECT = os.environ.get("CIV6AZ_VERTEX_PROJECT", "hrmtrain")
ZONE = os.environ.get("CIV6AZ_VM_ZONE", "us-central1-a")
REGION = os.environ.get("CIV6AZ_VERTEX_REGION", "us-central1")
VM_NAME = os.environ.get("CIV6AZ_VM_NAME", "civ6az-orchestrator")
BUCKET = os.environ.get("CIV6AZ_GCS_BUCKET", "civ6az-training")
MACHINE_TYPE = os.environ.get("CIV6AZ_VM_MACHINE_TYPE", "e2-small")
IMAGE_FAMILY = "debian-12"
IMAGE_PROJECT = "debian-cloud"
STATE_BLOB = "orchestrator/state.json"
HEARTBEAT_BLOB = "orchestrator/heartbeat.json"
DISK_SIZE_GB = 20


def check_vm() -> dict:
    """Check VM existence and status via Compute Engine API."""
    try:
        from google.cloud import compute_v1  # type: ignore[import-untyped]
        client = compute_v1.InstancesClient()
        instance = client.get(project=PROJECT, zone=ZONE, instance=VM_NAME)
        ext_ip = "none"
        if instance.network_interfaces and instance.network_interfaces[0].access_configs:
            ext_ip = instance.network_interfaces[0].access_configs[0].nat_i_p or "none"
        return {
            "exists": True,
            "status": instance.status,
            "name": instance.name,
            "machine_type": instance.machine_type.split("/")[-1],
            "external_ip": ext_ip,
            "creation_timestamp": instance.creation_timestamp,
        }
    except Exception as e:
        err = str(e)
        if "404" in err or "NOT_FOUND" in err or "was not found" in err.lower():
            return {"exists": False, "status": "NOT_FOUND"}
        return {"exists": False, "status": "ERROR", "error": err}


def create_vm() -> bool:
    """Create the orchestrator VM via Compute Engine API."""
    try:
        from google.cloud import compute_v1  # type: ignore[import-untyped]

        images_client = compute_v1.ImagesClient()
        latest_image = images_client.get_from_family(
            project=IMAGE_PROJECT, family=IMAGE_FAMILY
        )

        instance = compute_v1.Instance()
        instance.name = VM_NAME
        instance.machine_type = f"zones/{ZONE}/machineTypes/{MACHINE_TYPE}"

        # Boot disk
        disk = compute_v1.AttachedDisk()
        disk.auto_delete = True
        disk.boot = True
        init_params = compute_v1.AttachedDiskInitializeParams()
        init_params.source_image = latest_image.self_link
        init_params.disk_size_gb = DISK_SIZE_GB
        disk.initialize_params = init_params
        instance.disks = [disk]

        # Network (default VPC, external IP)
        net_iface = compute_v1.NetworkInterface()
        net_iface.name = "global/networks/default"
        access_config = compute_v1.AccessConfig()
        access_config.name = "External NAT"
        access_config.type_ = "ONE_TO_ONE_NAT"
        net_iface.access_configs = [access_config]
        instance.network_interfaces = [net_iface]

        # Service account scopes (cloud-platform = full API access)
        sa = compute_v1.ServiceAccount()
        sa.email = "default"
        sa.scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        instance.service_accounts = [sa]

        # Metadata: auto-restart, auto-resume-on-boot label
        metadata = compute_v1.Metadata()
        metadata.items = [
            compute_v1.Items(key="enable-oslogin", value="TRUE"),
        ]
        instance.metadata = metadata

        client = compute_v1.InstancesClient()
        op = client.insert(project=PROJECT, zone=ZONE, instance_resource=instance)

        print(f"  VM creation started (operation: {op.name})")
        print(f"  Waiting for VM to be RUNNING...")

        # Poll until done
        ops_client = compute_v1.ZoneOperationsClient()
        for _ in range(60):  # max 5 minutes
            op_status = ops_client.get(project=PROJECT, zone=ZONE, operation=op.name)
            if op_status.status == compute_v1.Operation.Status.DONE:
                if op_status.error:
                    print(f"  [FAIL] VM creation error: {op_status.error}")
                    return False
                print(f"  [OK] VM creation completed")
                return True
            time.sleep(5)

        print(f"  [WARN] Timed out waiting for VM creation — check GCP console")
        return False

    except Exception as e:
        print(f"  [FAIL] Exception creating VM: {e}")
        return False


def check_gcs_blob(bucket_name: str, blob_path: str) -> dict:
    """Download and parse a GCS JSON blob."""
    try:
        from google.cloud import storage  # type: ignore[import-untyped]
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return {"exists": False}
        content = blob.download_as_text()
        return {"exists": True, "content": json.loads(content)}
    except Exception as e:
        return {"exists": False, "error": str(e)}


def main() -> int:
    create_if_missing = "--create" in sys.argv

    print("=" * 60)
    print("  GCP VM & Orchestrator Status Check")
    print(f"  Project: {PROJECT}  Zone: {ZONE}")
    print(f"  {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    print("=" * 60)

    # ── VM status ──────────────────────────────────────────────────────
    print("\n[1/3] Checking VM status...")
    vm = check_vm()
    if vm["exists"]:
        print(f"  [OK] VM '{VM_NAME}' EXISTS  status={vm['status']}")
        print(f"       machine_type={vm.get('machine_type')}  ip={vm.get('external_ip')}")
        print(f"       created={vm.get('creation_timestamp')}")
    else:
        print(f"  [--] VM '{VM_NAME}' NOT FOUND (status={vm['status']})")
        if create_if_missing:
            print(f"\n  Creating VM '{VM_NAME}'...")
            ok = create_vm()
            if not ok:
                print("  [FAIL] VM creation failed. Exiting.")
                return 1
            # Re-check after creation
            time.sleep(5)
            vm = check_vm()
            if vm["exists"]:
                print(f"  [OK] VM now RUNNING  ip={vm.get('external_ip')}")
            else:
                print(f"  [FAIL] VM still not found after creation.")
                return 1
        else:
            print()
            print("  Run with --create to provision the VM automatically:")
            print(f"    python scripts/gcp_vm_check.py --create")
            print()
            print("  Or manually:")
            print(f"    gcloud.cmd compute instances create {VM_NAME} \\")
            print(f"      --zone={ZONE} --project={PROJECT} \\")
            print(f"      --machine-type={MACHINE_TYPE} \\")
            print(f"      --image-family={IMAGE_FAMILY} \\")
            print(f"      --image-project={IMAGE_PROJECT} \\")
            print(f"      --scopes=cloud-platform \\")
            print(f"      --boot-disk-size={DISK_SIZE_GB}GB")

    # ── GCS heartbeat ──────────────────────────────────────────────────
    print(f"\n[2/3] Checking heartbeat (gs://{BUCKET}/{HEARTBEAT_BLOB})...")
    hb = check_gcs_blob(BUCKET, HEARTBEAT_BLOB)
    if hb["exists"]:
        c = hb["content"]
        print(f"  [OK] Heartbeat found")
        print(f"       status={c.get('status')}  iter={c.get('iteration')}  "
              f"model=v{c.get('model_version')}  uptime={c.get('uptime_seconds')}s")
        print(f"       timestamp={c.get('timestamp_utc')}")
    else:
        print(f"  [--] No heartbeat -- orchestrator not yet running")

    # ── GCS state ──────────────────────────────────────────────────────
    print(f"\n[3/3] Checking state (gs://{BUCKET}/{STATE_BLOB})...")
    st = check_gcs_blob(BUCKET, STATE_BLOB)
    if st["exists"]:
        c = st["content"]
        pct = round(c.get("current_iteration", 0) / max(c.get("total_iterations", 1), 1) * 100, 1)
        print(f"  [OK] State found -- resume available")
        print(f"       iter={c.get('current_iteration')}/{c.get('total_iterations')} ({pct}%)  "
              f"model=v{c.get('model_version')}  cost=${c.get('total_estimated_cost', 0):.2f}")
        print(f"       last_checkpoint={c.get('last_checkpoint_utc')}")
    else:
        print(f"  [--] No state -- fresh run (no prior progress)")

    print("\n" + "=" * 60)
    return 0 if vm.get("exists") else 2


if __name__ == "__main__":
    sys.exit(main())
