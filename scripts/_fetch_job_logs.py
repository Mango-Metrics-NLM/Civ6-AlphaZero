"""Fetch the LATEST container logs for a Vertex AI job, focusing on application output."""

import json
import os
import socket
import sys

import requests

# Force IPv4
_orig_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _orig_gai(*a, **k) if r[0] == socket.AF_INET]

JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "492670183786151936"

creds_path = os.path.join(os.environ["APPDATA"], "gcloud", "application_default_credentials.json")
with open(creds_path) as f:
    creds = json.load(f)

token = requests.post("https://oauth2.googleapis.com/token", data={
    "client_id": creds["client_id"],
    "client_secret": creds["client_secret"],
    "refresh_token": creds["refresh_token"],
    "grant_type": "refresh_token",
}, timeout=15).json()["access_token"]

url = "https://logging.googleapis.com/v2/entries:list"
# Get latest logs (desc order) filtering for container stdout/stderr
resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json={
    "resourceNames": ["projects/hrmtrain"],
    "filter": f'resource.type="ml_job" AND resource.labels.job_id="{JOB_ID}" AND resource.labels.task_name="workerpool0-0"',
    "orderBy": "timestamp desc",
    "pageSize": 50,
}, timeout=30)

data = resp.json()
entries = data.get("entries", [])
if not entries:
    # Try broader filter
    print("No entries with ml_job filter. Trying broader search...")
    resp = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json={
        "resourceNames": ["projects/hrmtrain"],
        "filter": f'"{JOB_ID}" AND NOT textPayload:"provisioning"',
        "orderBy": "timestamp desc",
        "pageSize": 30,
    }, timeout=30)
    data = resp.json()
    entries = data.get("entries", [])

if not entries:
    print(f"No log entries found for job {JOB_ID}")
else:
    # Print in chronological order (oldest first)
    for entry in reversed(entries):
        ts = entry.get("timestamp", "")[11:23]
        msg = entry.get("textPayload", "")
        if not msg:
            jp = entry.get("jsonPayload", {})
            msg = jp.get("message", str(jp) if jp else "")
        if msg and "provisioning" not in msg.lower():
            print(f"{ts} {msg[:300]}")
