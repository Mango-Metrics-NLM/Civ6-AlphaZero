output "bucket_name" {
  description = "Name of the frontend GCS bucket."
  value       = google_storage_bucket.frontend.name
}

output "bucket_url" {
  description = "URL for uploading to the frontend bucket."
  value       = google_storage_bucket.frontend.url
}

output "cdn_ip" {
  description = "CDN external IP for DNS configuration."
  value       = google_compute_global_address.frontend_cdn.address
}

output "website_url" {
  description = "Direct GCS website URL."
  value       = "https://storage.googleapis.com/${google_storage_bucket.frontend.name}/index.html"
}
