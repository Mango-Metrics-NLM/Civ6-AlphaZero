"""Fast targeted extraction of training failure details.

Uses the already-known job data from the earlier query to fetch
individual jobs by resource name instead of listing all 1,441.
"""
import json
import sys

from google.cloud import aiplatform

aiplatform.init(project="hrmtrain", location="us-central1")

# From our earlier query, we know the most recent jobs. Let's get just
# the 3 most recent training jobs directly.
# State codes: 1=QUEUED, 2=RUNNING, 3=CANCELLING, 4=SUCCEEDED, 5=FAILED
print("Fetching last 3 training jobs (server-filtered by display name)...")
sys.stdout.flush()

try:
    jobs = aiplatform.CustomJob.list(
        filter='display_name:"civ6az-training"',
        order_by="create_time desc",
    )
    # Take only first 5 to avoid long processing
    recent = jobs[:5] if len(jobs) > 5 else jobs
except Exception as e:
    print(f"List failed: {e}")
    sys.exit(1)

print(f"Got {len(recent)} recent training jobs")
sys.stdout.flush()

for j in recent:
    print("=" * 70)
    print(f"Name: {j.display_name}")
    print(f"State: {j.state} (5=FAILED, 4=SUCCEEDED)")
    print(f"Created: {j.create_time}")
    print(f"Ended: {j.end_time}")

    # Get error info
    err = getattr(j, "error", None)
    if err:
        code_val = getattr(err, "code", "N/A")
        msg_val = getattr(err, "message", "N/A")
        print(f"Error Code: {code_val}")
        print(f"Error Msg: {msg_val[:800]}")
    else:
        print("Error: (none)")

    # Raw job spec details
    try:
        raw = j.to_dict()
        specs = raw.get("jobSpec", {}).get("workerPoolSpecs", [])
        for i, spec in enumerate(specs):
            container = spec.get("containerSpec", {})
            img = container.get("imageUri", "N/A")
            args_list = container.get("args", [])
            cmd_list = container.get("command", [])
            print(f"Container Image: {img}")
            if cmd_list:
                print(f"Command: {' '.join(cmd_list)}")
            if args_list:
                print(f"Args: {' '.join(args_list[:30])}")
    except Exception as ex:
        print(f"Raw parse error: {ex}")

    print()
    sys.stdout.flush()

print("Done.")
