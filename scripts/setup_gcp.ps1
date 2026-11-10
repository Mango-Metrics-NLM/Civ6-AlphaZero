<#
.SYNOPSIS
    GCP Resource Provisioning for Civ6 AlphaZero (Windows wrapper).

.DESCRIPTION
    PowerShell wrapper that provisions all GCP resources using gcloud CLI.
    Mirrors the functionality of setup_gcp.sh for Windows users.

    Creates: APIs, Artifact Registry, GCS Bucket, IAM SA, Secret Manager.

.PARAMETER Teardown
    Delete all provisioned resources.

.PARAMETER DryRun
    Print commands without executing.

.EXAMPLE
    .\scripts\setup_gcp.ps1
    .\scripts\setup_gcp.ps1 -DryRun
    .\scripts\setup_gcp.ps1 -Teardown
#>

param(
    [switch]$Teardown,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Load .env first
$setupScript = Join-Path $PSScriptRoot "setup_env.ps1"
if (Test-Path $setupScript) { . $setupScript }

$ProjectId  = if ($env:CIV6AZ_VERTEX_PROJECT) { $env:CIV6AZ_VERTEX_PROJECT } else { (gcloud config get-value project 2>$null) }
$Region     = if ($env:CIV6AZ_VERTEX_REGION)  { $env:CIV6AZ_VERTEX_REGION }  else { "us-central1" }
$BucketName = if ($env:CIV6AZ_GCS_BUCKET)     { $env:CIV6AZ_GCS_BUCKET }     else { throw "CIV6AZ_GCS_BUCKET must be set (there is no default bucket)" }
$ArRepoName = "civ6az"
$SaName     = "civ6az-vertex"
$SaEmail    = "${SaName}@${ProjectId}.iam.gserviceaccount.com"

$RequiredApis = @(
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "containerregistry.googleapis.com"
)

$SaRoles = @(
    "roles/aiplatform.user",
    "roles/storage.objectAdmin",
    "roles/artifactregistry.reader",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
)

$SecretMap = @{
    "wandb-api-key"     = "WANDB_API_KEY"
    "openai-api-key"    = "OPENAI_API_KEY"
    "anthropic-api-key" = "ANTHROPIC_API_KEY"
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Invoke-Gcloud {
    param([string[]]$Arguments)
    if ($DryRun) {
        Write-Host "  [DRY-RUN] gcloud $($Arguments -join ' ')" -ForegroundColor DarkGray
        return
    }
    $output = & gcloud @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "gcloud command returned $LASTEXITCODE : $output"
    }
    return $output
}

function Write-Status {
    param([string]$Message, [string]$Level = "INFO")
    $colors = @{ "INFO" = "Cyan"; "OK" = "Green"; "WARN" = "Yellow"; "ERROR" = "Red" }
    $color = $colors[$Level]
    Write-Host "[$Level] " -ForegroundColor $color -NoNewline
    Write-Host $Message
}

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
if ($Teardown) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Red
    Write-Host " TEARDOWN - Deleting Civ6 AlphaZero GCP Resources" -ForegroundColor Red
    Write-Host ("=" * 60) -ForegroundColor Red
    Write-Host ""
    $confirm = Read-Host "Delete ALL resources for project $ProjectId? (y/N)"
    if ($confirm -notin @("y", "Y")) {
        Write-Status "Teardown cancelled." "INFO"
        exit 0
    }

    Write-Status "Deleting secrets..." "INFO"
    foreach ($secret in $SecretMap.Keys) {
        Invoke-Gcloud @("secrets", "delete", $secret, "--project=$ProjectId", "--quiet") 2>$null
    }

    Write-Status "Removing IAM bindings..." "INFO"
    foreach ($role in $SaRoles) {
        Invoke-Gcloud @("projects", "remove-iam-policy-binding", $ProjectId,
            "--member=serviceAccount:$SaEmail", "--role=$role", "--quiet") 2>$null
    }

    Write-Status "Deleting service account..." "INFO"
    Invoke-Gcloud @("iam", "service-accounts", "delete", $SaEmail,
        "--project=$ProjectId", "--quiet") 2>$null

    Write-Status "Deleting Artifact Registry repo..." "INFO"
    Invoke-Gcloud @("artifacts", "repositories", "delete", $ArRepoName,
        "--location=$Region", "--project=$ProjectId", "--quiet") 2>$null

    Write-Status "Deleting GCS bucket..." "INFO"
    Invoke-Gcloud @("storage", "rm", "-r", "gs://$BucketName") 2>$null

    Write-Status "Teardown complete." "OK"
    exit 0
}

# ---------------------------------------------------------------------------
# Provisioning
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host " Civ6 AlphaZero - GCP Resource Provisioning" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""
Write-Status "Project:  $ProjectId"
Write-Status "Region:   $Region"
Write-Status "Bucket:   $BucketName"
Write-Status "AR Repo:  $ArRepoName"
Write-Status "SA:       $SaEmail"
Write-Host ""

# Step 1: Set project
Invoke-Gcloud @("config", "set", "project", $ProjectId)

# Step 2: Enable APIs
Write-Status "=== Step 1: Enable GCP APIs ===" "INFO"
foreach ($api in $RequiredApis) {
    Write-Status "Enabling $api..."
    Invoke-Gcloud @("services", "enable", $api, "--project=$ProjectId")
}
Write-Status "All APIs enabled." "OK"

# Step 3: Artifact Registry
Write-Host ""
Write-Status "=== Step 2: Artifact Registry ===" "INFO"
$arCheck = & gcloud artifacts repositories describe $ArRepoName --location=$Region --project=$ProjectId 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Status "Artifact Registry repo '$ArRepoName' already exists." "OK"
} else {
    Write-Status "Creating Artifact Registry Docker repo..."
    Invoke-Gcloud @("artifacts", "repositories", "create", $ArRepoName,
        "--repository-format=docker",
        "--location=$Region",
        "--project=$ProjectId",
        "--description=Civ6 AlphaZero training container images")
    Write-Status "Artifact Registry repo created." "OK"
}

# Step 4: GCS Bucket
Write-Host ""
Write-Status "=== Step 3: GCS Bucket ===" "INFO"
$bucketCheck = & gcloud storage buckets describe "gs://$BucketName" --project=$ProjectId 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Status "GCS bucket '$BucketName' already exists." "OK"
} else {
    Write-Status "Creating GCS bucket..."
    Invoke-Gcloud @("storage", "buckets", "create", "gs://$BucketName",
        "--project=$ProjectId",
        "--location=$Region",
        "--default-storage-class=STANDARD",
        "--uniform-bucket-level-access")
    Write-Status "GCS bucket created." "OK"
}

# Step 5: Service Account
Write-Host ""
Write-Status "=== Step 4: IAM Service Account ===" "INFO"
$saCheck = & gcloud iam service-accounts describe $SaEmail --project=$ProjectId 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Status "Service account '$SaName' already exists." "OK"
} else {
    Write-Status "Creating service account..."
    Invoke-Gcloud @("iam", "service-accounts", "create", $SaName,
        "--project=$ProjectId",
        "--display-name=Civ6 AlphaZero Vertex AI Service Account",
        "--description=Least-privilege SA for Vertex AI training jobs")
    Write-Status "Service account created." "OK"
}

Write-Status "Binding IAM roles..."
foreach ($role in $SaRoles) {
    Invoke-Gcloud @("projects", "add-iam-policy-binding", $ProjectId,
        "--member=serviceAccount:$SaEmail",
        "--role=$role",
        "--quiet") 2>$null
}
Write-Status "IAM roles bound." "OK"

# Step 6: Secret Manager
Write-Host ""
Write-Status "=== Step 5: Secret Manager ===" "INFO"
foreach ($secret in $SecretMap.Keys) {
    $envVar = $SecretMap[$secret]
    $secretCheck = & gcloud secrets describe $secret --project=$ProjectId 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Status "Secret '$secret' already exists." "OK"
    } else {
        $value = [System.Environment]::GetEnvironmentVariable($envVar, "Process")
        if ([string]::IsNullOrEmpty($value)) {
            Write-Status "No value for $envVar. Skipping '$secret'." "WARN"
            continue
        }
        Write-Status "Creating secret '$secret'..."
        Invoke-Gcloud @("secrets", "create", $secret,
            "--project=$ProjectId",
            "--replication-policy=automatic")
        $value | & gcloud secrets versions add $secret --project=$ProjectId --data-file=- 2>&1
        Write-Status "Secret '$secret' created." "OK"
    }
}

# Step 7: Docker auth
Write-Host ""
Write-Status "=== Step 6: Docker Auth ===" "INFO"
Invoke-Gcloud @("auth", "configure-docker", "${Region}-docker.pkg.dev", "--quiet") 2>$null
Write-Status "Docker configured for ${Region}-docker.pkg.dev" "OK"

# Summary
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host " Provisioning Complete" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host ""
Write-Status "All resources provisioned successfully!" "OK"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Build container:  python scripts/build_and_push.py --tag v1.0.0"
Write-Host "  2. Dry-run test:     .\scripts\launch_muzero_training.ps1 -DryRun"
Write-Host "  3. Smoke test:       .\scripts\launch_muzero_training.ps1 -Iterations 1 -Workers 1 -GamesPerWorker 2"
Write-Host ""
