# Root module: wires together all child modules for the Civ6 AlphaZero
# gameplay deployment on GCP.
#
# Usage:
#   cd infra/terraform
#   terraform init -backend-config=environments/prod/backend.tf
#   terraform plan -var-file=environments/prod/terraform.tfvars
#   terraform apply -var-file=environments/prod/terraform.tfvars

locals {
  name_prefix = "civ6az-${var.environment}"
  labels = {
    project     = "civ6az"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# ---------- Enable Required APIs ----------

resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "run.googleapis.com",
    "vpcaccess.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "firestore.googleapis.com",
    "identitytoolkit.googleapis.com", # Firebase Auth
    "storage.googleapis.com",
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# ---------- Networking ----------

module "networking" {
  source = "./modules/networking"

  project_id  = var.project_id
  region      = var.region
  name_prefix = local.name_prefix
  vpc_cidr    = var.vpc_cidr
  labels      = local.labels

  enable_cloud_armor = var.enable_cloud_armor

  depends_on = [google_project_service.apis]
}

# ---------- SpacetimeDB ----------

module "spacetimedb" {
  source = "./modules/spacetimedb"

  project_id   = var.project_id
  region       = var.region
  zone         = var.zone
  name_prefix  = local.name_prefix
  labels       = local.labels
  machine_type = var.spacetimedb_machine_type
  disk_size_gb = var.spacetimedb_disk_size_gb

  network_id       = module.networking.vpc_id
  subnet_id        = module.networking.services_subnet_id
  service_account  = module.networking.spacetimedb_sa_email

  depends_on = [module.networking]
}

# ---------- AI Inference (Cloud Run with GPU) ----------

module "ai_inference" {
  source = "./modules/ai-inference"

  project_id      = var.project_id
  region          = var.region
  name_prefix     = local.name_prefix
  labels          = local.labels
  container_image = var.inference_container_image
  min_instances   = var.inference_min_instances
  max_instances   = var.inference_max_instances
  concurrency     = var.inference_concurrency
  model_version   = var.model_version
  training_bucket = var.training_bucket

  vpc_connector_id = module.networking.vpc_connector_id
  service_account  = module.networking.inference_sa_email

  depends_on = [module.networking]
}

# ---------- Frontend (GCS + CDN) ----------

module "frontend" {
  source = "./modules/frontend"

  project_id    = var.project_id
  region        = var.region
  name_prefix   = local.name_prefix
  labels        = local.labels
  custom_domain = var.frontend_domain

  depends_on = [google_project_service.apis]
}

# ---------- Auth Service ----------

module "auth" {
  source = "./modules/auth"

  project_id          = var.project_id
  region              = var.region
  name_prefix         = local.name_prefix
  labels              = local.labels
  firebase_project_id = var.firebase_project_id

  vpc_connector_id = module.networking.vpc_connector_id
  service_account  = module.networking.auth_sa_email

  depends_on = [module.networking]
}

# ---------- Matchmaking Service ----------

module "matchmaking" {
  source = "./modules/matchmaking"

  project_id  = var.project_id
  region      = var.region
  name_prefix = local.name_prefix
  labels      = local.labels

  vpc_connector_id       = module.networking.vpc_connector_id
  service_account        = module.networking.matchmaker_sa_email
  spacetimedb_internal_ip = module.spacetimedb.internal_ip

  depends_on = [module.networking, module.spacetimedb]
}

# ---------- Replay Storage ----------

module "replay_storage" {
  source = "./modules/replay-storage"

  project_id  = var.project_id
  region      = var.region
  name_prefix = local.name_prefix
  labels      = local.labels

  depends_on = [google_project_service.apis]
}

# ---------- Monitoring ----------

module "monitoring" {
  source = "./modules/monitoring"

  project_id  = var.project_id
  name_prefix = local.name_prefix
  alert_email = var.alert_email

  inference_service_name   = module.ai_inference.service_name
  spacetimedb_instance     = module.spacetimedb.instance_name
  matchmaking_service_name = module.matchmaking.service_name
  auth_service_name        = module.auth.service_name

  depends_on = [
    module.ai_inference,
    module.spacetimedb,
    module.matchmaking,
    module.auth,
  ]
}

# ---------- CI/CD ----------

module "ci_cd" {
  source = "./modules/ci-cd"

  project_id    = var.project_id
  region        = var.region
  name_prefix   = local.name_prefix
  github_repo   = var.github_repo
  github_branch = var.github_branch

  inference_service_name   = module.ai_inference.service_name
  auth_service_name        = module.auth.service_name
  matchmaking_service_name = module.matchmaking.service_name

  depends_on = [google_project_service.apis]
}
