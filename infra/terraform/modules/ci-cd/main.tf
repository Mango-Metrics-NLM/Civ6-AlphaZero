# CI/CD module: Cloud Build triggers for the gameplay services.
#
# This extends the existing cloudbuild.yaml (training pipeline) with
# additional triggers for the gameplay services (inference, auth, matchmaker).
#
# Two trigger types:
# 1. Service deployment: builds + deploys Cloud Run services on push to main
# 2. Model update: deploys new model version when uploaded to GCS

# ---------- Cloud Build Service Account ----------

# The default Cloud Build SA needs permission to deploy Cloud Run services
resource "google_project_iam_member" "cloudbuild_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${data.google_project.main.number}@cloudbuild.gserviceaccount.com"
}

resource "google_project_iam_member" "cloudbuild_sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${data.google_project.main.number}@cloudbuild.gserviceaccount.com"
}

data "google_project" "main" {
  project_id = var.project_id
}

# ---------- Cloud Build Trigger: Service Deployment ----------

resource "google_cloudbuild_trigger" "gameplay_services" {
  count = var.github_repo != "" ? 1 : 0

  name        = "${var.name_prefix}-deploy-services"
  project     = var.project_id
  location    = var.region
  description = "Build and deploy gameplay services on push to main"

  github {
    owner = split("/", var.github_repo)[0]
    name  = split("/", var.github_repo)[1]

    push {
      branch = var.github_branch
    }
  }

  included_files = [
    "alpha_zero/**",
    "infra/**",
    "dockerfiles/**",
    "requirements-container.txt",
  ]

  filename = "infra/cloudbuild-gameplay.yaml"

  substitutions = {
    _REGION             = var.region
    _INFERENCE_SERVICE  = var.inference_service_name
    _AUTH_SERVICE       = var.auth_service_name
    _MATCHMAKER_SERVICE = var.matchmaking_service_name
  }
}

# ---------- Cloud Build Config for Gameplay Services ----------
# Note: the actual cloudbuild-gameplay.yaml is written as a separate file
# in the infra/ directory, not managed by Terraform. This trigger just
# points to it.

# ---------- Model Update Trigger ----------
# This is implemented as a Cloud Function triggered by GCS object finalize
# events on the model bucket. When a new model.pt is uploaded, it updates
# the MODEL_VERSION env var on the inference Cloud Run service.
#
# For simplicity at indie/startup scale, this can also be done manually:
#   gcloud run services update civ6az-prod-inference \
#     --update-env-vars=CIV6AZ_MODEL_VERSION=v269

# Pub/Sub topic for GCS notifications
resource "google_pubsub_topic" "model_updates" {
  name    = "${var.name_prefix}-model-updates"
  project = var.project_id
}

# GCS notification -> Pub/Sub when a new model is uploaded
resource "google_storage_notification" "model_upload" {
  count = var.github_repo != "" ? 1 : 0

  bucket         = "civ6az-training"
  payload_format = "JSON_API_V1"
  topic          = google_pubsub_topic.model_updates.id
  event_types    = ["OBJECT_FINALIZE"]

  object_name_prefix = "models/civ6az/"

  depends_on = [google_pubsub_topic.model_updates]
}
