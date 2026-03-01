# AI Inference module: Cloud Run service with NVIDIA L4 GPU for serving
# the Civ6TransformerNetwork + GumbelMCTS inference.
#
# Key design decisions:
# - Cloud Run GPU for scale-to-zero (no cost when idle)
# - L4 GPU is the cheapest available on Cloud Run
# - Concurrency=5: MCTS is CPU+GPU intensive, limit parallel searches
# - Model loaded from GCS on cold start, cached in memory
# - Health check endpoint validates model is loaded and GPU is available
#
# The container extends the existing Dockerfile.vertex with an HTTP/gRPC
# inference server that accepts game state and returns (action, policy, value).

locals {
  # Environment variables passed to the inference container
  env_vars = {
    CIV6AZ_MODEL_VERSION  = var.model_version
    CIV6AZ_GCS_BUCKET     = var.training_bucket
    CIV6AZ_MCTS_TYPE      = "gumbel"
    CIV6AZ_MCTS_SIMS      = "16"
    CIV6AZ_BOARD_SIZE      = "8"
    CIV6AZ_ACTION_SPACE    = "376"
    CIV6AZ_DEVICE          = "cuda"
    # Inference server config
    INFERENCE_PORT         = "8080"
    INFERENCE_WORKERS      = "1"
    # Disable W&B for inference
    WANDB_MODE             = "disabled"
  }
}

resource "google_cloud_run_v2_service" "inference" {
  provider = google-beta

  name     = "${var.name_prefix}-inference"
  project  = var.project_id
  location = var.region

  labels = var.labels

  template {
    labels = var.labels

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    max_instance_request_concurrency = var.concurrency

    timeout = "300s"

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
          cpu    = "8"
          memory = "32Gi"
          # Cloud Run GPU specification
          "nvidia.com/gpu" = "1"
        }

        cpu_idle          = false  # Keep CPU allocated while processing
        startup_cpu_boost = true   # Faster cold starts
      }

      # Startup probe: wait for model to load (up to 120s)
      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        timeout_seconds       = 3
        failure_threshold     = 24  # 10 + 24*5 = 130s max startup
      }

      # Liveness probe: ensure model is still responsive
      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
      }

      dynamic "env" {
        for_each = local.env_vars
        content {
          name  = env.key
          value = env.value
        }
      }
    }

    # GPU node selector for L4
    node_selector {
      accelerator = "nvidia-l4"
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  lifecycle {
    ignore_changes = [
      # Allow CI/CD to update the container image without Terraform drift
      template[0].containers[0].image,
    ]
  }
}

# Allow unauthenticated access (auth is handled at the application layer
# via Firebase JWT validation, not at the Cloud Run IAM layer).
resource "google_cloud_run_v2_service_iam_member" "inference_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.inference.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
