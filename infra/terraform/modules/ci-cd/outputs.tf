output "trigger_id" {
  description = "Cloud Build trigger ID (empty if no GitHub repo configured)."
  value       = var.github_repo != "" ? google_cloudbuild_trigger.gameplay_services[0].id : ""
}

output "model_updates_topic" {
  description = "Pub/Sub topic for model update notifications."
  value       = google_pubsub_topic.model_updates.name
}
