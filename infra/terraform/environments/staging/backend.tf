# GCS backend configuration for staging Terraform state.
#
# Pass this to terraform init:
#   terraform init -backend-config=environments/staging/backend.tf

# Replace with a bucket dedicated to Terraform state. Do NOT point this at the
# training-artifact bucket: state holds resource ids and must be separately access-controlled.
bucket = "YOUR_TERRAFORM_STATE_BUCKET"
prefix = "terraform/gameplay/staging"
