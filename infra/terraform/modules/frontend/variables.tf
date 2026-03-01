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

variable "custom_domain" {
  description = "Custom domain for the frontend (empty = no custom domain)."
  type        = string
  default     = ""
}
