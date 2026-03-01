# GCS backend configuration for production Terraform state.
#
# Pass this to terraform init:
#   terraform init -backend-config=environments/prod/backend.tf

bucket = "civ6az-training"
prefix = "terraform/gameplay/prod"
