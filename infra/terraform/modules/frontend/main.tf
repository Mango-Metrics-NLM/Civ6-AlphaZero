# Frontend module: GCS bucket for React SPA static assets with Cloud CDN.
#
# The React SPA communicates directly with:
# - SpacetimeDB via WebSocket (game state sync)
# - AI Inference via HTTPS (move requests)
# - Auth service via HTTPS (login/signup)
# - Matchmaking service via HTTPS (find game)
#
# All communication goes through the Global HTTPS Load Balancer.

# ---------- GCS Bucket for Static Assets ----------

resource "google_storage_bucket" "frontend" {
  name     = "${var.name_prefix}-frontend"
  project  = var.project_id
  location = var.region

  labels = var.labels

  # Static website configuration
  website {
    main_page_suffix = "index.html"
    not_found_page   = "index.html" # SPA fallback for client-side routing
  }

  # CORS for API calls from the frontend
  cors {
    origin          = ["*"] # Restrict to actual domain in production
    method          = ["GET", "HEAD"]
    response_header = ["Content-Type", "Cache-Control"]
    max_age_seconds = 3600
  }

  uniform_bucket_level_access = true

  # Versioning for rollback capability
  versioning {
    enabled = true
  }

  # Lifecycle: delete old versions after 30 days
  lifecycle_rule {
    condition {
      num_newer_versions = 3
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  force_destroy = false
}

# Public read access for the static website
resource "google_storage_bucket_iam_member" "frontend_public_read" {
  bucket = google_storage_bucket.frontend.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}

# ---------- Cloud CDN Backend Bucket ----------

resource "google_compute_backend_bucket" "frontend" {
  name        = "${var.name_prefix}-frontend-backend"
  project     = var.project_id
  bucket_name = google_storage_bucket.frontend.name
  enable_cdn  = true

  cdn_policy {
    cache_mode                   = "CACHE_ALL_STATIC"
    default_ttl                  = 3600 # 1 hour
    max_ttl                      = 86400 # 24 hours
    client_ttl                   = 3600
    negative_caching             = true
    serve_while_stale            = 86400
    signed_url_cache_max_age_sec = 0
  }
}

# ---------- URL Map (for standalone CDN IP, not the main LB) ----------

resource "google_compute_url_map" "frontend" {
  name            = "${var.name_prefix}-frontend-urlmap"
  project         = var.project_id
  default_service = google_compute_backend_bucket.frontend.id
}

# ---------- HTTPS Proxy + Forwarding Rule ----------

resource "google_compute_global_address" "frontend_cdn" {
  name    = "${var.name_prefix}-frontend-cdn-ip"
  project = var.project_id
}

# Managed SSL certificate (only if custom domain is provided)
resource "google_compute_managed_ssl_certificate" "frontend" {
  count = var.custom_domain != "" ? 1 : 0

  name    = "${var.name_prefix}-frontend-cert"
  project = var.project_id

  managed {
    domains = [var.custom_domain]
  }
}

resource "google_compute_target_https_proxy" "frontend" {
  count = var.custom_domain != "" ? 1 : 0

  name             = "${var.name_prefix}-frontend-https"
  project          = var.project_id
  url_map          = google_compute_url_map.frontend.id
  ssl_certificates = [google_compute_managed_ssl_certificate.frontend[0].id]
}

resource "google_compute_global_forwarding_rule" "frontend_https" {
  count = var.custom_domain != "" ? 1 : 0

  name        = "${var.name_prefix}-frontend-fwd"
  project     = var.project_id
  target      = google_compute_target_https_proxy.frontend[0].id
  ip_address  = google_compute_global_address.frontend_cdn.address
  port_range  = "443"
  ip_protocol = "TCP"
}

# HTTP proxy (always available, for direct bucket URL access)
resource "google_compute_target_http_proxy" "frontend" {
  name    = "${var.name_prefix}-frontend-http"
  project = var.project_id
  url_map = google_compute_url_map.frontend.id
}

resource "google_compute_global_forwarding_rule" "frontend_http" {
  name        = "${var.name_prefix}-frontend-fwd-http"
  project     = var.project_id
  target      = google_compute_target_http_proxy.frontend.id
  ip_address  = google_compute_global_address.frontend_cdn.address
  port_range  = "80"
  ip_protocol = "TCP"
}
