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

variable "vpc_connector_id" {
  description = "Serverless VPC Connector ID."
  type        = string
}

variable "service_account" {
  description = "Service account email."
  type        = string
}

variable "spacetimedb_internal_ip" {
  description = "Internal IP of the SpacetimeDB instance."
  type        = string
}
