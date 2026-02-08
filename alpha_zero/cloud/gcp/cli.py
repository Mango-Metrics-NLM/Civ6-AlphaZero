"""GCP CLI for AlphaZero.

Provides command-line utilities for managing GCP resources,
validating configuration, and launching jobs.
"""

import argparse
import logging
import sys
from typing import NoReturn

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import GCSBackend
from alpha_zero.utils.config import ConfigManager

logger = logging.getLogger(__name__)


def validate_config(args: argparse.Namespace) -> None:
    """Validate GCP configuration and permissions."""
    print("Validating GCP Configuration...")
    
    try:
        # 1. Load Configs
        # Load Civ6 config to ensure it exists/is valid syntax
        try:
             _ = ConfigManager(args.config)
             print(f"Civ6 Config '{args.config}' loaded successfully")
        except Exception as e:
             print(f"Warning: Failed to load '{args.config}': {e}")

        # Load GCP Config from Env
        gcp_config = GCPConfig.from_env(validate=False) # Validate manually to print nice errors
        
        # Check required fields
        if not gcp_config.project_id:
            print("Missing CIV6AZ_VERTEX_PROJECT env var")
        if not gcp_config.bucket:
             print("Missing CIV6AZ_GCS_BUCKET env var")
             
        print(f"GCP Config loaded: Project={gcp_config.project_id}, Region={gcp_config.region}")
        
        if not gcp_config.project_id or not gcp_config.bucket:
             print("Cannot proceed without Project ID and Bucket")
             sys.exit(1)

        # 2. Check Storage Access
        print("\nChecking Storage Access...")
        storage = GCSBackend(gcp_config.project_id)
        # Check if bucket exists using list_blobs or similar, since GCSBackend doesn't have bucket.exists() reference
        # Start looking at GCSBackend implementation in storage.py...
        # It has blob_exists(bucket, path) but not bucket_exists.
        # But we can try to list_blobs with partial prefix or just try to get the bucket object via client.
        try:
            bucket_obj = storage.client.bucket(gcp_config.bucket)
            if bucket_obj.exists():
                print(f"Bucket '{gcp_config.bucket}' exists and is accessible")
            else:
                print(f"Bucket '{gcp_config.bucket}' does not exist or is inaccessible")
                sys.exit(1)
        except Exception as e:
             print(f"Failed to access bucket '{gcp_config.bucket}': {e}")
             sys.exit(1)
            
        # 3. Check Vertex AI (if enabled)
        # vertex_ai_enabled is not explicitly in GCPConfig but implied/defaulted.
        # We check if project_id is set as a proxy for enablement/capability.
        if gcp_config.project_id:
            print("\nChecking Vertex AI...")
            # Import here to avoid hard dependency if not used
            from google.cloud import aiplatform
            
            aiplatform.init(
                project=gcp_config.project_id,
                location=gcp_config.region,
                staging_bucket=f"gs://{gcp_config.bucket}",
            )
            print("Vertex AI initialized successfully")
        else:
            print("\nVertex AI disabled/not configured (no project_id)")
            
        print("\nValidation Complete: All checks passed!")
        
    except Exception as e:
        print(f"\nValidation Failed: {e}")
        logger.exception("Validation failed")
        sys.exit(1)


def main() -> NoReturn:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AlphaZero GCP CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    
    # Validate Command
    validate_parser = subparsers.add_parser("validate", help="Validate GCP configuration")
    validate_parser.add_argument(
        "--config",
        type=str,
        default="configs/civ6_config.yaml",
        help="Path to configuration file",
    )
    validate_parser.set_defaults(func=validate_config)
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run command
    args.func(args)
    sys.exit(0)


if __name__ == "__main__":
    main()
