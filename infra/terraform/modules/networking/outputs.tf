output "vpc_id" {
  description = "VPC network self_link."
  value       = google_compute_network.main.self_link
}

output "vpc_name" {
  description = "VPC network name."
  value       = google_compute_network.main.name
}

output "services_subnet_id" {
  description = "Services subnet self_link."
  value       = google_compute_subnetwork.services.self_link
}

output "vpc_connector_id" {
  description = "Serverless VPC Connector ID for Cloud Run."
  value       = google_vpc_access_connector.main.id
}

output "lb_ip_address" {
  description = "Global load balancer external IP."
  value       = google_compute_global_address.lb.address
}

output "cloud_armor_policy_id" {
  description = "Cloud Armor security policy self_link (empty if disabled)."
  value       = var.enable_cloud_armor ? google_compute_security_policy.main[0].self_link : ""
}

output "inference_sa_email" {
  description = "Service account email for AI inference."
  value       = google_service_account.inference.email
}

output "spacetimedb_sa_email" {
  description = "Service account email for SpacetimeDB."
  value       = google_service_account.spacetimedb.email
}

output "auth_sa_email" {
  description = "Service account email for Auth service."
  value       = google_service_account.auth.email
}

output "matchmaker_sa_email" {
  description = "Service account email for Matchmaking service."
  value       = google_service_account.matchmaker.email
}
