variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "name_prefix" {
  type = string
}

variable "labels" {
  type    = map(string)
  default = {}
}

variable "container_image" {
  description = "Full container image URI for the inference service."
  type        = string
}

variable "min_instances" {
  description = "Minimum Cloud Run instances (0 = scale to zero)."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum Cloud Run instances."
  type        = number
  default     = 10
}

variable "concurrency" {
  description = "Max concurrent requests per instance."
  type        = number
  default     = 5
}

variable "model_version" {
  description = "Model version string (e.g. 'v268')."
  type        = string
}

variable "training_bucket" {
  description = "GCS bucket containing model weights."
  type        = string
}

variable "vpc_connector_id" {
  description = "Serverless VPC Connector ID."
  type        = string
}

variable "service_account" {
  description = "Service account email."
  type        = string
}
