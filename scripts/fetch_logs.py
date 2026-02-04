"""Quick script to fetch Cloud Logging entries for a Vertex AI job - ERROR FOCUS."""
import sys
from google.cloud import logging_v2

client = logging_v2.Client(project="hrmtrain")
job_id = "2594389683936952320"
filter_str = f'resource.type="ml_job" AND resource.labels.job_id="{job_id}"'

print(f"Fetching logs for job {job_id}...")
entries = list(client.list_entries(filter_=filter_str, max_results=200))

if not entries:
    print("No log entries found")
    sys.exit(0)

print(f"Found {len(entries)} log entries. Showing last 50 and all errors:\n")

# Show last 50 entries
for entry in entries[-50:]:
    ts = entry.timestamp.strftime("%H:%M:%S") if entry.timestamp else "??:??:??"
    payload = entry.payload
    if isinstance(payload, dict):
        text = payload.get("message", str(payload))
    else:
        text = str(payload)
    # Highlight errors
    if "error" in text.lower() or "exception" in text.lower() or "traceback" in text.lower():
        print(f"!!! [{ts}] {text[:300]}")
    else:
        print(f"[{ts}] {text[:150]}")
