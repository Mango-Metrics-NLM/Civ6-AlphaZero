# GCS backend configuration for staging Terraform state.
#
# Pass this to terraform init:
#   terraform init -backend-config=environments/staging/backend.tf

bucket = "civ6az-training"
prefix = "terraform/gameplay/staging"
