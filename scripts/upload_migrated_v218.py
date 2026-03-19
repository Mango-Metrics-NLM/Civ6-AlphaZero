#!/usr/bin/env python3
"""Download v218, migrate to N-player, and upload to GCS.

Sprint 6 Phase 3: Prepare migrated v218 for fine-tuning.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Download, migrate, and upload v218."""
    import platform

    project_root = Path(__file__).parent.parent
    v218_path = project_root / "v218.pt"
    v218_migrated_path = project_root / "v218_migrated.pt"

    gcs_source = "gs://civ6az-training/models/civ6az/v218/model.pt"
    gcs_dest = "gs://civ6az-training/models/civ6az/v218_migrated/model.pt"

    # Determine gsutil command based on platform
    if platform.system() == "Windows":
        gsutil_cmd = r"C:\Users\iansh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd"
    else:
        gsutil_cmd = "gsutil"

    # Step 1: Download v218 from GCS
    print("Step 1: Downloading v218 from GCS...")
    cmd = [gsutil_cmd, "cp", gcs_source, str(v218_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] Download failed: {result.stderr}")
        return 1
    print(f"[OK] Downloaded: {v218_path.stat().st_size / 1e6:.1f} MB")

    # Step 2: Migrate v218 to N-player
    print("\nStep 2: Migrating v218 to N-player...")
    sys.path.insert(0, str(project_root))
    from scripts.migrate_model_to_nplayer import migrate_checkpoint

    try:
        migrate_checkpoint(
            input_path=str(v218_path),
            output_path=str(v218_migrated_path),
            dry_run=False
        )
    except Exception as e:
        print(f"[FAIL] Migration failed: {e}")
        return 1

    print(f"[OK] Migrated: {v218_migrated_path.stat().st_size / 1e6:.1f} MB")

    # Step 3: Upload migrated model to GCS
    print("\nStep 3: Uploading migrated model to GCS...")
    cmd = [gsutil_cmd, "cp", str(v218_migrated_path), gcs_dest]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] Upload failed: {result.stderr}")
        return 1

    print(f"[OK] Uploaded to: {gcs_dest}")

    # Step 4: Create metadata file
    print("\nStep 4: Creating metadata...")
    metadata = '{"migrated_from": "v218", "migration_version": "1.5.0", "spatial_channels": 28, "value_output_dim": 6}'
    cmd = [gsutil_cmd, "cp", "-", "gs://civ6az-training/models/civ6az/v218_migrated/metadata.json"]
    result = subprocess.run(cmd, input=metadata, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] Metadata upload failed: {result.stderr}")
        # Don't fail on metadata upload
    else:
        print("[OK] Metadata uploaded")

    # Cleanup local files
    print("\nCleaning up local files...")
    v218_path.unlink(missing_ok=True)
    v218_migrated_path.unlink(missing_ok=True)
    print("[OK] Cleanup complete")

    print("\n" + "="*60)
    print("SUCCESS: Migrated v218 ready for fine-tuning!")
    print("="*60)
    print(f"\nGCS location: {gcs_dest}")
    print("\nNext steps:")
    print("  1. Launch 10-iteration fine-tuning:")
    print("     bash scripts/launch_finetune_v218.sh")
    print("  2. Or run 20-iteration batch self-play benchmark:")
    print("     bash scripts/launch_prod_v1.5.0.sh 20")

    return 0

if __name__ == "__main__":
    sys.exit(main())
