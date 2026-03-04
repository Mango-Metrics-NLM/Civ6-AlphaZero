# Root-level outputs for the Civ6 AlphaZero Gameplay infrastructure.
#
# These provide the essential URLs and identifiers needed to connect
# the frontend, configure DNS, and operate the system.

output "inference_service_url" {
  description = "URL of the AI inference Cloud Run service."
  value       = module.ai_inference.service_url
}

output "auth_service_url" {
  description = "URL of the authentication Cloud Run service."
  value       = module.auth.service_url
}

output "matchmaking_service_url" {
  description = "URL of the matchmaking Cloud Run service."
  value       = module.matchmaking.service_url
}

output "spacetimedb_internal_ip" {
  description = "Internal IP of the SpacetimeDB instance (reachable from VPC)."
  value       = module.spacetimedb.internal_ip
}

output "spacetimedb_external_ip" {
  description = "External IP of the SpacetimeDB instance (via LB or direct)."
  value       = module.spacetimedb.external_ip
}

output "frontend_bucket_url" {
  description = "URL of the frontend GCS bucket (for deployment uploads)."
  value       = module.frontend.bucket_url
}

output "frontend_cdn_ip" {
  description = "CDN IP address for DNS configuration."
  value       = module.frontend.cdn_ip
}

output "replay_bucket_name" {
  description = "Name of the replay storage GCS bucket."
  value       = module.replay_storage.bucket_name
}

output "load_balancer_ip" {
  description = "Global load balancer external IP."
  value       = module.networking.lb_ip_address
}
