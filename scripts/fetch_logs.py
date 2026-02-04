"""Quick script to fetch Cloud Logging entries for a Vertex AI job."""
import sys
from google.cloud import logging_v2

client = logging_v2.Client(project="hrmtrain")
job_id = "1239650621028564992"
filter_str = f'resource.type="ml_job" AND resource.labels.job_id="{job_id}"'

print(f"Fetching logs for job {job_id}...")
entries = list(client.list_entries(filter_=filter_str, max_results=50))

if not entries:
    print("No log entries found")
    sys.exit(0)

print(f"Found {len(entries)} log entries:\n")
for entry in entries:
    ts = entry.timestamp.strftime("%H:%M:%S") if entry.timestamp else "??:??:??"
    payload = entry.payload
    if isinstance(payload, dict):
        text = payload.get("message", str(payload))
    else:
        text = str(payload)
    print(f"[{ts}] {text[:200]}")
