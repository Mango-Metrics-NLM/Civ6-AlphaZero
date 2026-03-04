variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "name_prefix" {
  type = string
}

variable "github_repo" {
  description = "GitHub repository in 'owner/repo' format (empty = no trigger)."
  type        = string
  default     = ""
}

variable "github_branch" {
  description = "Branch regex for the Cloud Build trigger."
  type        = string
  default     = "^main$"
}

variable "inference_service_name" {
  type = string
}

variable "auth_service_name" {
  type = string
}

variable "matchmaking_service_name" {
  type = string
}

variable "training_bucket" {
  description = "GCS bucket name for training artifacts and model uploads."
  type        = string
}

variable "model_prefix" {
  description = "GCS object prefix for model uploads (triggers Pub/Sub notification)."
  type        = string
  default     = "models/civ6az/"
}
