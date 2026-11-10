"""List GCP service accounts and diagnose Vertex AI scope issues."""

import json
import os
import socket
import sys

import requests

# Force IPv4
_orig_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _orig_gai(*a, **k) if r[0] == socket.AF_INET]  # type: ignore[assignment]

PROJECT = os.environ.get("CIV6AZ_VERTEX_PROJECT")
if not PROJECT:
    raise SystemExit("CIV6AZ_VERTEX_PROJECT is not set (there is no default project).")

creds_path = os.path.join(os.environ["APPDATA"], "gcloud", "application_default_credentials.json")
with open(creds_path) as f:
    creds = json.load(f)

token = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    },
    timeout=15,
).json()["access_token"]

print("=== Service Accounts ===")
r = requests.get(
    f"https://iam.googleapis.com/v1/projects/{PROJECT}/serviceAccounts",
    headers={"Authorization": f"Bearer {token}"},
    timeout=15,
)
data = r.json()
for sa in data.get("accounts", []):
    email = sa.get("email", "")
    name = sa.get("displayName", "")
    print(f"  {email}  ({name})")

# Also check what the project number is (for default compute SA)
print("\n=== Default Compute SA would be: ===")
print(f"  {PROJECT} project — check number")

# Check if there's a Vertex AI SA
print("\n=== Vertex AI job details ===")
job_id = sys.argv[1] if len(sys.argv) > 1 else "3572076797744906240"
region = "us-central1"
url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{region}/customJobs/{job_id}"
r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
job = r.json()
# Print the job's service account and scheduling info
job_spec = job.get("jobSpec", {})
scheduling = job_spec.get("scheduling", {})
sa = job_spec.get("serviceAccount", "NOT SET")
print(f"  Service Account: {sa}")
print(f"  Scheduling: {json.dumps(scheduling, indent=2) if scheduling else 'default'}")

# Show worker pool spec env vars (look for CIV6AZ_CONTAINER_IMAGE_URI)
wps = job_spec.get("workerPoolSpecs", [{}])[0]
container = wps.get("containerSpec", {})
env_list = container.get("env", [])
print("\n=== Container Env Vars ===")
for e in env_list:
    print(f"  {e.get('name', '')}: {e.get('value', '')}")
