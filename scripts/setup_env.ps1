<#
.SYNOPSIS
    Load .env file into the current PowerShell session as environment variables.

.DESCRIPTION
    Parses the project .env file and exports each KEY=VALUE pair as a
    process-scoped environment variable. Skips comments and blank lines.
    Designed for Windows PowerShell compatibility with gcloud CLI.

.PARAMETER EnvFile
    Path to the .env file. Defaults to the project root .env.

.EXAMPLE
    . .\scripts\setup_env.ps1
    # Now all .env values are available as $env:CIV6AZ_VERTEX_PROJECT etc.
#>

param(
    [string]$EnvFile = (Join-Path (Join-Path $PSScriptRoot "..") ".env")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $EnvFile)) {
    Write-Error "Environment file not found: $EnvFile"
    exit 1
}

$loaded = 0
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    # Skip empty lines and comments
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    # Parse KEY=VALUE (handle values with = signs)
    $eqIdx = $line.IndexOf("=")
    if ($eqIdx -gt 0) {
        $key = $line.Substring(0, $eqIdx).Trim()
        $value = $line.Substring($eqIdx + 1).Trim()
        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        $loaded++
    }
}

Write-Host "Loaded $loaded environment variables from $EnvFile" -ForegroundColor Green

# Verify critical GCP variables
$critical = @("CIV6AZ_VERTEX_PROJECT", "CIV6AZ_GCS_BUCKET", "CIV6AZ_CONTAINER_IMAGE_URI")
$missing = @()
foreach ($var in $critical) {
    $val = [System.Environment]::GetEnvironmentVariable($var, "Process")
    if ([string]::IsNullOrEmpty($val)) {
        $missing += $var
    }
}

if ($missing.Count -gt 0) {
    Write-Warning "Missing critical variables: $($missing -join ', ')"
}
else {
    Write-Host "All critical GCP variables set." -ForegroundColor Cyan
}
