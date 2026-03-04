# Auth module: Cloud Run service that validates Firebase Auth JWTs and
# translates them to SpacetimeDB identity tokens.
#
# Firebase Auth handles the actual OAuth flows (Google, GitHub, email/password).
# This service is a thin middleware that:
# 1. Validates the Firebase JWT
# 2. Extracts user identity
# 3. Issues a SpacetimeDB connection token
# 4. Provides user profile API

resource "google_cloud_run_v2_service" "auth" {
  provider = google-beta

  name     = "${var.name_prefix}-auth"
  project  = var.project_id
  location = var.region

  labels = var.labels

  template {
    labels = var.labels

    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    max_instance_request_concurrency = 80

    timeout = "30s"

    service_account = var.service_account

    vpc_access {
      connector = var.vpc_connector_id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "256Mi"
        }
        cpu_idle = true # Allow CPU throttling when idle (saves cost)
      }

      env {
        name  = "FIREBASE_PROJECT_ID"
        value = var.firebase_project_id
      }

      env {
        name  = "PORT"
        value = "8080"
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 2
        period_seconds        = 3
        timeout_seconds       = 2
        failure_threshold     = 5
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        period_seconds    = 30
        timeout_seconds   = 3
        failure_threshold = 3
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
    ]
  }
}

# Public access (Firebase JWT validation happens in application code)
resource "google_cloud_run_v2_service_iam_member" "auth_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.auth.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
