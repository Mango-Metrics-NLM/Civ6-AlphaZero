"""Find Vertex AI Custom Jobs by display name.

Usage:
    python scripts/find_job.py [--display-name NAME] [--project PROJECT] [--region REGION]
"""

from __future__ import annotations

import argparse
import os

from google.cloud import aiplatform


def find_jobs(
    project: str,
    region: str,
    display_name: str | None = None,
) -> None:
    """List Vertex AI custom jobs, optionally filtered by display name.

    Args:
        project: GCP project ID.
        region: GCP region.
        display_name: Optional display name filter.
    """
    aiplatform.init(project=project, location=region)

    filter_str = f'display_name="{display_name}"' if display_name else ""
    jobs = aiplatform.CustomJob.list(
        filter=filter_str,
        order_by="create_time desc",
    )
    for job in jobs:
        print(f"Name: {job.display_name}")
        print(f"ID: {job.name}")
        print(f"Resource Name: {job.resource_name}")
        print(f"State: {job.state}")
        print(f"Create Time: {job.create_time}")
        print("-" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Vertex AI Custom Jobs")
    parser.add_argument(
        "--project",
        default=os.environ.get("CIV6AZ_VERTEX_PROJECT"),
        required=not os.environ.get("CIV6AZ_VERTEX_PROJECT"),
        help="GCP project ID (default: $CIV6AZ_VERTEX_PROJECT; there is no built-in default)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("CIV6AZ_VERTEX_REGION", "us-central1"),
        help="GCP region (default: $CIV6AZ_VERTEX_REGION or 'us-central1')",
    )
    parser.add_argument(
        "--display-name",
        default=None,
        help="Filter by display name (e.g. 'civ6az-tournament-51models')",
    )
    args = parser.parse_args()
    find_jobs(
        project=args.project,
        region=args.region,
        display_name=args.display_name,
    )
