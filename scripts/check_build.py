"""Fetch Cloud Build status and logs via REST API.

.. deprecated::
    Use ``scripts/monitor_build.py`` instead, which supports polling,
    step verification, and structured CLI arguments.
"""
import sys
import warnings

warnings.warn(
    "check_build.py is deprecated — use monitor_build.py for polling and "
    "step-count verification.",
    DeprecationWarning,
    stacklevel=1,
)
import json
import google.auth
import google.auth.transport.requests
import urllib.request

BUILD_ID = sys.argv[1] if len(sys.argv) > 1 else "283ca181-47bb-4029-b1c0-5a81690b8756"
PROJECT = sys.argv[2] if len(sys.argv) > 2 else "hrmtrain"

creds, _ = google.auth.default()
creds.refresh(google.auth.transport.requests.Request())

url = "https://cloudbuild.googleapis.com/v1/projects/{}/builds/{}".format(PROJECT, BUILD_ID)
req = urllib.request.Request(url, headers={"Authorization": "Bearer " + creds.token})

try:
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
except Exception as e:
    print("ERROR fetching build:", e)
    sys.exit(1)

print("BUILD STATUS:", data.get("status"))
print("LOG URL:", data.get("logUrl", "N/A"))
print("LOGS BUCKET:", data.get("logsBucket", "N/A"))
print()

# Show step statuses
steps = data.get("steps", [])
for i, s in enumerate(steps):
    step_id = s.get("id", "unknown")
    step_status = s.get("status", "unknown")
    print("  Step {}: {} -> {}".format(i, step_id, step_status))

# Show failure info
print()
if "statusDetail" in data:
    print("STATUS DETAIL:", data["statusDetail"])
if "failureInfo" in data:
    print("FAILURE INFO:", json.dumps(data["failureInfo"], indent=2))

# Try to read the log from GCS
logs_bucket = data.get("logsBucket", "")
if logs_bucket:
    log_url = "{}/log-{}.txt".format(logs_bucket, BUILD_ID)
    # Convert gs:// to https://
    if log_url.startswith("gs://"):
        https_url = log_url.replace("gs://", "https://storage.googleapis.com/", 1)
    else:
        https_url = log_url
    
    print("\nFetching build log from:", https_url)
    try:
        log_req = urllib.request.Request(https_url, headers={"Authorization": "Bearer " + creds.token})
        log_resp = urllib.request.urlopen(log_req, timeout=30)
        log_text = log_resp.read().decode("utf-8", errors="replace")
        # Show last 100 lines
        lines = log_text.strip().split("\n")
        print("\n=== BUILD LOG (last 100 lines) ===")
        for line in lines[-100:]:
            print(line)
    except Exception as e:
        print("Could not fetch log:", e)
