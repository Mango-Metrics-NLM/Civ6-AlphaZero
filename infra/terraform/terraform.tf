# Terraform and provider configuration for Civ6 AlphaZero Gameplay Deployment.
#
# State is stored in a GCS bucket for team collaboration and locking.
# The google-beta provider is required for Cloud Run GPU support and
# Serverless VPC Access.

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.40"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.40"
    }
  }

  # Backend is configured per-environment in environments/*/backend.tf
  # to allow prod and staging to use separate state files.
  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}
