<#
.SYNOPSIS
    GCP Authentication Setup Script for Vertex AI Training
    
.DESCRIPTION
    Configures Google Cloud Platform authentication and validates environment
    variables required for Civ6 AlphaZero Vertex AI training.
    
    This script:
    - Checks for gcloud CLI installation
    - Configures Application Default Credentials (ADC)
    - Validates required environment variables
    - Provides guidance for missing configuration
    
.PARAMETER SkipLogin
    Skip the interactive ADC login (useful for CI/CD environments with service accounts)
    
.PARAMETER ValidateOnly
    Only validate configuration without making changes
    
.EXAMPLE
    .\setup_gcp_auth.ps1
    
.EXAMPLE
    .\setup_gcp_auth.ps1 -ValidateOnly
    
.NOTES
    Author: Civ6 AlphaZero Team
    Version: 1.0.0
    Requires: Google Cloud SDK (gcloud)
#>

[CmdletBinding()]
param(
    [switch]$SkipLogin,
    [switch]$ValidateOnly
)

# Configuration - All values can be overridden via environment variables
$script:RequiredEnvVars = @(
    @{
        Name = "CIV6AZ_VERTEX_PROJECT"
        Description = "GCP Project ID for Vertex AI jobs"
        Example = "my-gcp-project-123"
    },
    @{
        Name = "CIV6AZ_GCS_BUCKET"
        Description = "GCS bucket for checkpoints and data"
        Example = "gs://my-civ6az-bucket"
    }
)

$script:OptionalEnvVars = @(
    @{
        Name = "CIV6AZ_CONTAINER_IMAGE_URI"
        Description = "Container image URI for training workers"
        Example = "gcr.io/my-project/civ6az:latest"
        Default = "Auto-generated from project and version"
    },
    @{
        Name = "CIV6AZ_VERTEX_REGION"
        Description = "GCP region for Vertex AI"
        Example = "us-central1"
        Default = "us-central1"
    },
    @{
        Name = "CIV6AZ_VERTEX_SERVICE_ACCOUNT"
        Description = "Service account email for Vertex AI jobs"
        Example = "vertex-sa@my-project.iam.gserviceaccount.com"
        Default = "Default compute service account"
    }
)

# Logging functions
function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    $colors = @{
        "INFO" = "Cyan"
        "OK" = "Green"
        "WARN" = "Yellow"
        "ERROR" = "Red"
    }
    $color = $colors[$Status]
    if (-not $color) { $color = "White" }
    Write-Host "[$Status] " -ForegroundColor $color -NoNewline
    Write-Host $Message
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Blue
    Write-Host " $Title" -ForegroundColor Blue
    Write-Host ("=" * 60) -ForegroundColor Blue
}

# Check gcloud installation
function Test-GcloudInstalled {
    try {
        $null = Get-Command gcloud -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Get gcloud configuration
function Get-GcloudConfig {
    $config = @{}
    try {
        $config.Account = (gcloud config get-value account 2>$null)
        $config.Project = (gcloud config get-value project 2>$null)
        $config.Region = (gcloud config get-value compute/region 2>$null)
        
        # Check ADC
        $adcPath = "$env:APPDATA\gcloud\application_default_credentials.json"
        $config.AdcExists = Test-Path $adcPath
        
        # Try to get active identity
        $identity = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
        $config.ActiveIdentity = $identity
    }
    catch {
        Write-Status "Failed to retrieve gcloud configuration: $_" "WARN"
    }
    return $config
}

# Validate environment variables
function Test-EnvironmentVariables {
    $allValid = $true
    $results = @()
    
    foreach ($var in $script:RequiredEnvVars) {
        $value = [Environment]::GetEnvironmentVariable($var.Name)
        $results += @{
            Name = $var.Name
            Value = $value
            Required = $true
            Valid = ($null -ne $value -and $value.Trim() -ne "")
            Description = $var.Description
            Example = $var.Example
        }
        if (-not $results[-1].Valid) {
            $allValid = $false
        }
    }
    
    foreach ($var in $script:OptionalEnvVars) {
        $value = [Environment]::GetEnvironmentVariable($var.Name)
        $results += @{
            Name = $var.Name
            Value = $value
            Required = $false
            Valid = $true  # Optional vars are always "valid"
            Description = $var.Description
            Example = $var.Example
            Default = $var.Default
        }
    }
    
    return @{
        AllValid = $allValid
        Variables = $results
    }
}

# Configure ADC
function Initialize-ApplicationDefaultCredentials {
    Write-Status "Initiating Application Default Credentials login..." "INFO"
    Write-Status "A browser window will open for authentication." "INFO"
    
    try {
        gcloud auth application-default login --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Application Default Credentials configured successfully!" "OK"
            return $true
        }
        else {
            Write-Status "ADC configuration failed with exit code: $LASTEXITCODE" "ERROR"
            return $false
        }
    }
    catch {
        Write-Status "ADC configuration error: $_" "ERROR"
        return $false
    }
}

# Display configuration guidance
function Show-ConfigurationGuide {
    param($EnvResults)
    
    Write-Section "Configuration Guide"
    
    $missingRequired = $EnvResults.Variables | Where-Object { $_.Required -and -not $_.Valid }
    
    if ($missingRequired.Count -gt 0) {
        Write-Status "The following required environment variables must be set:" "WARN"
        Write-Host ""
        
        Write-Host "# PowerShell (Session):" -ForegroundColor Yellow
        foreach ($var in $missingRequired) {
            Write-Host "`$env:$($var.Name) = `"$($var.Example)`"" -ForegroundColor Gray
        }
        
        Write-Host ""
        Write-Host "# PowerShell (Persistent - User Profile):" -ForegroundColor Yellow
        foreach ($var in $missingRequired) {
            Write-Host "[Environment]::SetEnvironmentVariable(`"$($var.Name)`", `"$($var.Example)`", `"User`")" -ForegroundColor Gray
        }
        
        Write-Host ""
        Write-Host "# Windows System Environment Variables (Admin):" -ForegroundColor Yellow
        Write-Host "# Use Settings > System > About > Advanced system settings > Environment Variables"
    }
    else {
        Write-Status "All required environment variables are configured!" "OK"
    }
    
    Write-Host ""
    Write-Host "Optional Variables:" -ForegroundColor Cyan
    $optionalVars = $EnvResults.Variables | Where-Object { -not $_.Required }
    foreach ($var in $optionalVars) {
        $status = if ($var.Value) { "[SET]" } else { "[DEFAULT: $($var.Default)]" }
        Write-Host "  $($var.Name): $status" -ForegroundColor Gray
    }
}

# Main execution
function Main {
    Write-Section "GCP Authentication Setup for Civ6 AlphaZero"
    Write-Status "Validating GCP configuration for Vertex AI training" "INFO"
    
    # Step 1: Check gcloud installation
    Write-Section "Step 1: Google Cloud SDK"
    if (Test-GcloudInstalled) {
        $version = gcloud version --format="value(Google Cloud SDK)" 2>$null
        Write-Status "gcloud CLI found (version: $version)" "OK"
    }
    else {
        Write-Status "gcloud CLI not found!" "ERROR"
        Write-Host ""
        Write-Host "Install Google Cloud SDK from:" -ForegroundColor Yellow
        Write-Host "  https://cloud.google.com/sdk/docs/install" -ForegroundColor Cyan
        return 1
    }
    
    # Step 2: Get current gcloud configuration
    Write-Section "Step 2: Current Configuration"
    $gcConfig = Get-GcloudConfig
    
    Write-Host "  Active Account:  " -NoNewline
    if ($gcConfig.ActiveIdentity) {
        Write-Host $gcConfig.ActiveIdentity -ForegroundColor Green
    }
    else {
        Write-Host "(none)" -ForegroundColor Yellow
    }
    
    Write-Host "  Default Project: " -NoNewline
    if ($gcConfig.Project) {
        Write-Host $gcConfig.Project -ForegroundColor Green
    }
    else {
        Write-Host "(none)" -ForegroundColor Yellow
    }
    
    Write-Host "  ADC Configured:  " -NoNewline
    if ($gcConfig.AdcExists) {
        Write-Host "Yes" -ForegroundColor Green
    }
    else {
        Write-Host "No" -ForegroundColor Yellow
    }
    
    # Step 3: Configure ADC if needed
    if (-not $ValidateOnly -and -not $SkipLogin) {
        Write-Section "Step 3: Application Default Credentials"
        
        if (-not $gcConfig.AdcExists) {
            Write-Status "ADC not configured. Setting up now..." "WARN"
            $adcResult = Initialize-ApplicationDefaultCredentials
            if (-not $adcResult) {
                Write-Status "Failed to configure ADC. Please run manually:" "ERROR"
                Write-Host "  gcloud auth application-default login" -ForegroundColor Cyan
            }
        }
        else {
            Write-Status "ADC already configured" "OK"
            Write-Host "  To refresh credentials, run:" -ForegroundColor Gray
            Write-Host "    gcloud auth application-default login" -ForegroundColor Cyan
        }
    }
    elseif ($SkipLogin) {
        Write-Section "Step 3: Application Default Credentials (Skipped)"
        Write-Status "ADC login skipped (SkipLogin flag set)" "INFO"
    }
    else {
        Write-Section "Step 3: Application Default Credentials"
        if ($gcConfig.AdcExists) {
            Write-Status "ADC is configured" "OK"
        }
        else {
            Write-Status "ADC not configured" "WARN"
        }
    }
    
    # Step 4: Validate environment variables
    Write-Section "Step 4: Environment Variables"
    $envResults = Test-EnvironmentVariables
    
    foreach ($var in $envResults.Variables) {
        $indicator = if ($var.Required) { "*" } else { " " }
        Write-Host "  $indicator $($var.Name): " -NoNewline
        
        if ($var.Value) {
            # Mask sensitive values
            $displayValue = if ($var.Value.Length -gt 20) {
                $var.Value.Substring(0, 10) + "..." + $var.Value.Substring($var.Value.Length - 5)
            }
            else {
                $var.Value
            }
            Write-Host $displayValue -ForegroundColor Green
        }
        elseif ($var.Required) {
            Write-Host "(NOT SET - REQUIRED)" -ForegroundColor Red
        }
        else {
            Write-Host "(not set, using default)" -ForegroundColor Gray
        }
    }
    Write-Host "  (* = required)" -ForegroundColor Gray
    
    # Step 5: Configuration guidance if needed
    if (-not $envResults.AllValid) {
        Show-ConfigurationGuide -EnvResults $envResults
        return 1
    }
    
    # Summary
    Write-Section "Summary"
    if ($envResults.AllValid -and ($gcConfig.AdcExists -or $SkipLogin)) {
        Write-Status "GCP configuration is complete!" "OK"
        Write-Host ""
        Write-Host "You can now run Vertex AI training jobs:" -ForegroundColor Cyan
        Write-Host "  python -m alpha_zero.cloud.gcp.cli train --config configs/vertex_training.yaml" -ForegroundColor Gray
        return 0
    }
    else {
        Write-Status "Configuration incomplete. Please address the issues above." "WARN"
        return 1
    }
}

# Run main function
exit (Main)
