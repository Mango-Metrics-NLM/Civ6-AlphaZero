# Root-level input variables for the Civ6 AlphaZero Gameplay infrastructure.
#
# These are set via terraform.tfvars in each environment directory
# (environments/prod/terraform.tfvars, environments/staging/terraform.tfvars).

variable "project_id" {
  description = "GCP project ID. Must be set via .tfvars or -var flag."
  type        = string
}

variable "region" {
  description = "GCP region for all resources. Must be set via .tfvars or -var flag."
  type        = string
}

variable "zone" {
  description = "GCP zone for zonal resources (GCE instances)."
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Deployment environment name (prod, staging)."
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["prod", "staging"], var.environment)
    error_message = "Environment must be 'prod' or 'staging'."
  }
}

# ---------- Networking ----------

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.0.0.0/16"
}

# ---------- SpacetimeDB ----------

variable "spacetimedb_machine_type" {
  description = "Machine type for the SpacetimeDB GCE instance."
  type        = string
  default     = "e2-standard-4"
}

variable "spacetimedb_disk_size_gb" {
  description = "Boot disk size in GB for SpacetimeDB."
  type        = number
  default     = 50
}

# ---------- AI Inference ----------

variable "inference_container_image" {
  description = "Container image URI for the AI inference service."
  type        = string
  default     = "us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-inference:latest"
}

variable "inference_min_instances" {
  description = "Minimum Cloud Run instances for AI inference (0 = scale to zero)."
  type        = number
  default     = 0
}

variable "inference_max_instances" {
  description = "Maximum Cloud Run instances for AI inference."
  type        = number
  default     = 10
}

variable "inference_concurrency" {
  description = "Max concurrent requests per Cloud Run inference instance."
  type        = number
  default     = 5
}

variable "model_version" {
  description = "Current model version to serve (e.g. 'v268')."
  type        = string
  default     = "v268"
}

# ---------- Frontend ----------

variable "frontend_domain" {
  description = "Custom domain for the frontend (empty string = no custom domain)."
  type        = string
  default     = ""
}

# ---------- Auth ----------

variable "firebase_project_id" {
  description = "Firebase project ID (usually same as GCP project). Must be set via .tfvars."
  type        = string
}

# ---------- Monitoring ----------

variable "alert_email" {
  description = "Email address for monitoring alert notifications."
  type        = string
  default     = ""
}

variable "enable_cloud_armor" {
  description = "Whether to enable Cloud Armor WAF on the load balancer."
  type        = bool
  default     = true
}

# ---------- CI/CD ----------

variable "github_repo" {
  description = "GitHub repository in 'owner/repo' format for Cloud Build triggers."
  type        = string
  default     = ""
}

variable "github_branch" {
  description = "Branch pattern for Cloud Build trigger."
  type        = string
  default     = "^main$"
}

# ---------- GCS ----------

variable "training_bucket" {
  description = "Existing GCS bucket for training artifacts (models, trajectories)."
  type        = string
  default     = "civ6az-training"
}
