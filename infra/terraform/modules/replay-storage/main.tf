# Replay Storage module: GCS bucket with lifecycle policies for game replays.
#
# Replays are stored as JSON files containing the full game history
# (moves, game states, AI evaluations). Lifecycle policies automatically
# tier storage to reduce costs:
# - Standard (0-30 days): frequently accessed recent replays
# - Nearline (30-90 days): occasionally accessed
# - Coldline (90-365 days): rarely accessed
# - Delete after 365 days

resource "google_storage_bucket" "replays" {
  name     = "${var.name_prefix}-replays"
  project  = var.project_id
  location = var.region

  labels = var.labels

  storage_class                = "STANDARD"
  uniform_bucket_level_access  = true

  versioning {
    enabled = false # Replays are immutable, no versioning needed
  }

  # Transition to Nearline after 30 days
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  # Transition to Coldline after 90 days
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  # Delete after 365 days
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }

  # CORS for signed URL access from the frontend
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD"]
    response_header = ["Content-Type", "Content-Disposition"]
    max_age_seconds = 3600
  }
}

# No public access -- replays are accessed via signed URLs generated
# by the auth service. This prevents unauthorized scraping.
