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
