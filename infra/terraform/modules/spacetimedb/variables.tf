variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "zone" {
  type = string
}

variable "name_prefix" {
  type = string
}

variable "labels" {
  type    = map(string)
  default = {}
}

variable "machine_type" {
  type    = string
  default = "e2-standard-4"
}

variable "disk_size_gb" {
  type    = number
  default = 50
}

variable "network_id" {
  description = "VPC network self_link."
  type        = string
}

variable "subnet_id" {
  description = "Subnet self_link for the instance."
  type        = string
}

variable "service_account" {
  description = "Service account email for the instance."
  type        = string
}

variable "spacetimedb_image" {
  description = "Pinned SpacetimeDB container image. Avoid :latest for supply chain safety."
  type        = string
  default     = "clockworklabs/spacetimedb:v1.1.1"
}
