output "bucket_name" {
  description = "Name of the replays GCS bucket."
  value       = google_storage_bucket.replays.name
}

output "bucket_url" {
  description = "URL of the replays GCS bucket."
  value       = google_storage_bucket.replays.url
}
