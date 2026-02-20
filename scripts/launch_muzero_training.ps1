<#
.SYNOPSIS
    Launch MuZero training on Vertex AI via the local orchestrator.

.DESCRIPTION
    Runs the Vertex AI orchestrator LOCALLY to dispatch MuZero self-play
    and training workers as Vertex AI Custom Jobs.

    The orchestrator runs on your machine. Workers run in containers on GCP.

    Pre-flight checks:
    - Validates gcloud authentication
    - Verifies container image exists in Artifact Registry
    - Displays cost estimate for live runs
    - Requires explicit confirmation for non-DryRun

.PARAMETER Iterations
    Number of self-play -> train iterations. Default: 3

.PARAMETER Workers
    Number of parallel self-play workers. Default: 2

.PARAMETER GamesPerWorker
    Self-play games per worker per iteration. Default: 50

.PARAMETER Epochs
    Training epochs per iteration. Default: 10

.PARAMETER DryRun
    If set, runs in dry-run mode (no cloud costs).

.PARAMETER AgentId
    Agent ID for GCS path scoping. Default: muzero-pbt

.PARAMETER SkipChecks
    Skip pre-flight checks (auth, image verify). Useful for CI.

.EXAMPLE
    # Dry run (free, no cloud jobs)
    .\scripts\launch_muzero_training.ps1 -DryRun

    # Minimal smoke test (1 iteration, 1 worker, 2 games)
    .\scripts\launch_muzero_training.ps1 -Iterations 1 -Workers 1 -GamesPerWorker 2

    # Full PBT run
    .\scripts\launch_muzero_training.ps1 -Iterations 10 -Workers 4 -GamesPerWorker 100
#>

param(
    [int]$Iterations = 3,
    [int]$Workers = 2,
    [int]$GamesPerWorker = 50,
    [int]$Epochs = 10,
    [int]$MctsSimulations = 64,
    [int]$NumResBlocks = 6,
    [int]$NumFilters = 64,
    [string]$AgentId = "muzero-pbt",
    [switch]$DryRun,
    [switch]$PostRunTournament,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"

# Load environment variables
$setupScript = Join-Path $PSScriptRoot "setup_env.ps1"
if (Test-Path $setupScript) {
    . $setupScript
}
else {
    Write-Error "Cannot find setup_env.ps1 at $setupScript"
    exit 1
}

# Validate critical env vars
$required = @("CIV6AZ_VERTEX_PROJECT", "CIV6AZ_GCS_BUCKET", "CIV6AZ_CONTAINER_IMAGE_URI")
foreach ($var in $required) {
    $val = [System.Environment]::GetEnvironmentVariable($var, "Process")
    if ([string]::IsNullOrEmpty($val)) {
        Write-Error "Missing required environment variable: $var"
        exit 1
    }
}

# =============================================================================
# Pre-flight Checks
# =============================================================================
if (-not $SkipChecks -and -not $DryRun) {
    Write-Host ""
    Write-Host "=== Pre-flight Checks ===" -ForegroundColor Cyan

    # Check 1: gcloud authentication
    Write-Host "  Checking gcloud authentication..." -NoNewline
    try {
        $identity = & gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
        if ([string]::IsNullOrEmpty($identity)) {
            Write-Host " FAIL" -ForegroundColor Red
            Write-Error "No active gcloud account. Run: gcloud auth login"
            exit 1
        }
        Write-Host " OK ($identity)" -ForegroundColor Green
    }
    catch {
        Write-Host " WARN (gcloud not found, skipping)" -ForegroundColor Yellow
    }

    # Check 2: Container image exists
    Write-Host "  Checking container image..." -NoNewline
    $imageUri = $env:CIV6AZ_CONTAINER_IMAGE_URI
    try {
        $describeResult = & gcloud artifacts docker images describe $imageUri 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
        }
        else {
            Write-Host " NOT FOUND" -ForegroundColor Yellow
            Write-Warning "Container image not found: $imageUri"
            Write-Warning "Build it first: python scripts/build_and_push.py --tag v1.0.0"
            $proceed = Read-Host "Continue anyway? (y/N)"
            if ($proceed -notin @("y", "Y")) { exit 1 }
        }
    }
    catch {
        Write-Host " SKIP (gcloud error)" -ForegroundColor Yellow
    }

    Write-Host ""
}

# =============================================================================
# Cost Estimation
# =============================================================================
# Approximate costs (T4 budget tier, us-central1)
# Self-play worker:  ~$0.35/hr (n1-standard-8 + T4 GPU)
# Training worker:   ~$0.35/hr (n1-standard-8 + T4 GPU)
# Estimate assumes ~30 min per self-play iteration and ~15 min training
$selfplayHoursPerIter = [math]::Max(0.5, $GamesPerWorker * 0.01)
$trainHoursPerIter = 0.25
$selfplayCost = $Iterations * $Workers * $selfplayHoursPerIter * 0.35
$trainCost = $Iterations * $trainHoursPerIter * 0.35
$totalEstimate = [math]::Round($selfplayCost + $trainCost, 2)

Write-Host "=== MuZero Training Launch ===" -ForegroundColor Cyan
Write-Host "Project:    $env:CIV6AZ_VERTEX_PROJECT"
Write-Host "Bucket:     $env:CIV6AZ_GCS_BUCKET"
Write-Host "Container:  $env:CIV6AZ_CONTAINER_IMAGE_URI"
Write-Host "Agent ID:   $AgentId"
Write-Host "Iterations: $Iterations"
Write-Host "Workers:    $Workers"
Write-Host "Games/Wkr:  $GamesPerWorker"
Write-Host "Epochs:     $Epochs"
Write-Host "MCTS Sims:  $MctsSimulations"
Write-Host "Arch:       ${NumResBlocks}L x ${NumFilters}D"

if (-not $DryRun) {
    Write-Host ""
    Write-Host "Estimated cost: " -NoNewline
    Write-Host "`$$totalEstimate USD" -ForegroundColor Yellow
    Write-Host "  Self-play: `$$([math]::Round($selfplayCost, 2)) ($Iterations iters x $Workers workers)"
    Write-Host "  Training:  `$$([math]::Round($trainCost, 2)) ($Iterations iters)"
}
Write-Host ""

# Build command
$cmd = @(
    "python", "-m", "alpha_zero.cloud.gcp.vertex_orchestrator",
    "--algorithm=muzero",
    "--iterations=$Iterations",
    "--workers=$Workers",
    "--games_per_worker=$GamesPerWorker",
    "--epochs=$Epochs",
    "--mcts_simulations=$MctsSimulations",
    "--num_res_blocks=$NumResBlocks",
    "--num_filters=$NumFilters",
    "--agent_id=$AgentId"
)

if ($DryRun) {
    $cmd += "--dry_run"
    Write-Host "MODE: DRY RUN (no cloud costs)" -ForegroundColor Yellow
}
else {
    Write-Host "MODE: LIVE (will incur Vertex AI costs)" -ForegroundColor Red
    # Require confirmation for live runs
    $confirm = Read-Host "Proceed with live training? (y/N)"
    if ($confirm -notin @("y", "Y")) {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
}

if ($PostRunTournament) {
    $cmd += "--post_run_tournament"
}

Write-Host ""
Write-Host "Command: $($cmd -join ' ')" -ForegroundColor DarkGray
Write-Host ""

# Execute
& $cmd[0] $cmd[1..($cmd.Length - 1)]
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "=== Training Complete ===" -ForegroundColor Green
}
else {
    Write-Host ""
    Write-Host "=== Training Failed (exit code: $exitCode) ===" -ForegroundColor Red
}

exit $exitCode

