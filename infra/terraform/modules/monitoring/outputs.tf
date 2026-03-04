output "dashboard_id" {
  description = "Monitoring dashboard ID."
  value       = google_monitoring_dashboard.main.id
}

output "inference_latency_alert_id" {
  description = "Alert policy ID for inference latency."
  value       = google_monitoring_alert_policy.inference_latency.id
}
