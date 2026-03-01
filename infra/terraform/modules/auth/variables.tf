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

variable "firebase_project_id" {
  description = "Firebase project ID for auth token validation."
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
