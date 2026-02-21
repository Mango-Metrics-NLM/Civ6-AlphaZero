# PRD: Bypass & Mitigation of GCP Connectivity Issues

**Epic**: Phase 2: Bypass & Mitigation
**Status**: Draft

## 1. User Story

As a **Developer**,  
I want to **implement workarounds for the blocked `gcloud` CLI**,  
So that **I can continue interacting with Google Cloud resources (Vertex AI, Storage) without being blocked by local shell restrictions.**

## 2. Acceptance Criteria

### Scenario 1: Direct Binary Execution

- **Given** PowerShell blocks `gcloud.ps1`
- **When** I execute `gcloud.cmd` directly or use `cmd /c "gcloud ..."`
- **Then** the command runs successfully, bypassing the Execution Policy.

### Scenario 2: Disable Update Checks

- **Given** `gcloud` hangs on network calls due to background update checks
- **When** I set `CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=1`
- **Then** commands return response immediately without attempting to contact update servers.

### Scenario 3: Python SDK Fallback

- **Given** `gcloud` CLI is unusable
- **When** I run a Python script importing `google.cloud.storage`
- **Then** it successfully authenticates using ADC (or explicit key) and lists buckets.

## 3. Scope

- **In-Scope**: Environment variable configuration, shell aliases/wrappers, verifying Python SDK connectivity.
- **Out-of-Scope**: Permanent system-wide policy changes (e.g. changing global Execution Policy via GPO).

## 4. Success Metrics

- **Availability**: At least one method (CLI or Python) allows listing GCS buckets.
- **Latency**: Command execution time < 5s for basic metadata ops.
