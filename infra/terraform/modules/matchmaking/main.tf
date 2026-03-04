# Matchmaking module: Cloud Run service backed by Firestore for player
# matching and game room creation.
#
# The matchmaking flow:
# 1. Player submits matchmaking request with ELO rating
# 2. Service writes to Firestore queue collection
# 3. Periodic scan matches players within ELO range (expanding over time)
# 4. On match: creates SpacetimeDB game room, notifies both players
# 5. Players connect directly to SpacetimeDB via WebSocket

# ---------- Firestore Database ----------

resource "google_firestore_database" "matchmaking" {
  provider = google-beta

  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  # Lifecycle: prevent accidental deletion
  lifecycle {
    prevent_destroy = true
  }
}

# ---------- Firestore Indexes ----------

# Index for matchmaking queue: sort by ELO for efficient range queries
resource "google_firestore_index" "queue_by_elo" {
  provider = google-beta

  project    = var.project_id
  database   = google_firestore_database.matchmaking.name
  collection = "matchmaking_queue"

  fields {
    field_path = "status"
    order      = "ASCENDING"
  }

  fields {
    field_path = "elo_rating"
    order      = "ASCENDING"
  }

  fields {
    field_path = "queued_at"
    order      = "ASCENDING"
  }
}

# Index for active games: lookup by player
resource "google_firestore_index" "games_by_player" {
  provider = google-beta

  project    = var.project_id
  database   = google_firestore_database.matchmaking.name
  collection = "active_games"

  fields {
    field_path = "player_ids"
    array_config = "CONTAINS"
  }

  fields {
    field_path = "created_at"
    order      = "DESCENDING"
  }
}

# ---------- Cloud Run Service ----------

resource "google_cloud_run_v2_service" "matchmaking" {
  provider = google-beta

  name     = "${var.name_prefix}-matchmaker"
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
      image = "${var.name_prefix}-matchmaker:latest"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "256Mi"
        }
        cpu_idle = true
      }

      env {
        name  = "PORT"
        value = "8080"
      }

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }

      env {
        name  = "SPACETIMEDB_HOST"
        value = var.spacetimedb_internal_ip
      }

      env {
        name  = "SPACETIMEDB_PORT"
        value = "3000"
      }

      # Matchmaking config
      env {
        name  = "MATCH_ELO_RANGE_INITIAL"
        value = "100"
      }

      env {
        name  = "MATCH_ELO_RANGE_EXPAND_INTERVAL_SEC"
        value = "30"
      }

      env {
        name  = "MATCH_ELO_RANGE_EXPAND_AMOUNT"
        value = "50"
      }

      env {
        name  = "MATCH_ELO_RANGE_MAX"
        value = "500"
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

# Public access
resource "google_cloud_run_v2_service_iam_member" "matchmaking_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.matchmaking.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
