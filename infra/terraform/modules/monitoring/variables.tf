variable "project_id" {
  type = string
}

variable "name_prefix" {
  type = string
}

variable "alert_email" {
  description = "Email for alert notifications (empty = no email alerts)."
  type        = string
  default     = ""
}

variable "inference_service_name" {
  description = "Cloud Run service name for AI inference."
  type        = string
}

variable "spacetimedb_instance" {
  description = "GCE instance name for SpacetimeDB."
  type        = string
}

variable "matchmaking_service_name" {
  description = "Cloud Run service name for matchmaking."
  type        = string
}

variable "auth_service_name" {
  description = "Cloud Run service name for auth."
  type        = string
}
