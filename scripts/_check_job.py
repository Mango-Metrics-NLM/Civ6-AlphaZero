"""Quick script to check Vertex AI job status."""

import json
import os
import pathlib
import socket

_o = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _o(*a, **k) if r[0] == socket.AF_INET]
import requests  # noqa: E402

adc_path = pathlib.Path(os.environ["APPDATA"], "gcloud", "application_default_credentials.json")
adc = json.loads(adc_path.read_text(encoding="utf-8"))
tok = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": adc["client_id"],
        "client_secret": adc["client_secret"],
        "refresh_token": adc["refresh_token"],
        "grant_type": "refresh_token",
    },
    timeout=10,
).json()["access_token"]

project = os.environ.get("CIV6AZ_VERTEX_PROJECT")
if not project:
    raise SystemExit("CIV6AZ_VERTEX_PROJECT is not set (there is no default project).")
region = "us-central1"

# List recent jobs
url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/customJobs"
r = requests.get(
    url, headers={"Authorization": f"Bearer {tok}"}, params={"pageSize": 5}, timeout=15
)
data = r.json()

for job in data.get("customJobs", []):
    name = job.get("displayName", "?")
    state = job.get("state", "?")
    create_time = job.get("createTime", "?")[:19]
    job_id = job.get("name", "").split("/")[-1]
    error = job.get("error", {}).get("message", "")
    image = ""
    try:
        image = job["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]["imageUri"].split("/")[-1]
    except (KeyError, IndexError):
        pass
    line = f"  {name} | {state} | {create_time} | {image}"
    if error:
        line += f" | ERR: {error[:80]}"
    print(line)
