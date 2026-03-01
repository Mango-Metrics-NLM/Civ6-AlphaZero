output "instance_name" {
  description = "Name of the SpacetimeDB GCE instance."
  value       = google_compute_instance.spacetimedb.name
}

output "instance_id" {
  description = "Instance ID."
  value       = google_compute_instance.spacetimedb.instance_id
}

output "internal_ip" {
  description = "Internal IP address of the SpacetimeDB instance."
  value       = google_compute_instance.spacetimedb.network_interface[0].network_ip
}

output "external_ip" {
  description = "External IP (empty if none assigned)."
  value       = try(google_compute_instance.spacetimedb.network_interface[0].access_config[0].nat_ip, "")
}

output "self_link" {
  description = "Self link of the instance."
  value       = google_compute_instance.spacetimedb.self_link
}
