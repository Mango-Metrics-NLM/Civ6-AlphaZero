# PRD: Rapid Diagnosis of GCP Connectivity

**Epic**: Phase 1: Rapid Diagnosis
**Status**: Draft

## 1. User Story

As a **Developer**,  
I want to **identify why `gcloud` commands are hanging or failing** on my local Windows machine,  
So that **I can understand the root cause (auth vs network vs execution policy) and determine the correct fix.**

## 2. Acceptance Criteria

### Scenario 1: Identify Execution Block

- **Given** I run `gcloud info` in PowerShell
- **When** the command fails with a SecurityError
- **Then** I confirm that the Restricted Execution Policy is blocking the `gcloud.ps1` script.

### Scenario 2: Identify Network Hang

- **Given** I bypass the execution policy (e.g. via `cmd /c`)
- **When** I run a basic command like `gcloud info`
- **Then** the command should output configuration details within 5 seconds OR provide a specific error message (not just hang indefinitely).

### Scenario 3: Verify Auth Status

- **Given** `gcloud info` runs successfully
- **When** I inspect the output
- **Then** I can see the currently active account and project, and whether ADC checks pass.

## 3. Scope

- **In-Scope**: Checking `gcloud` executable path, verifying PowerShell Execution Policy, checking for update-check hangs.
- **Out-of-Scope**: Fixing the issue (that is Phase 2/3), deploying new infrastructure.

## 4. Success Metrics

- **Time to Diagnosis**: Root cause identified within < 10 minutes of active investigation.
- **Clarity**: A definitive "Yes/No" on whether the issue is Execution Policy, Auth Token, or Network.
