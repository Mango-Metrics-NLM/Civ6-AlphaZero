variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "name_prefix" {
  type = string
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "labels" {
  type    = map(string)
  default = {}
}

variable "enable_cloud_armor" {
  type    = bool
  default = true
}
