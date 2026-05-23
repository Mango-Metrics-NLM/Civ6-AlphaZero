#!/usr/bin/env bash
# Setup GCP Monitoring for Civ6 AlphaZero Training Pipeline
#
# Creates alerting policies and notification channels for:
# - Training job failures
# - Cost ceiling breaches
# - Self-play stalls
#
# Prerequisites:
#   - gcloud CLI authenticated with project owner/editor
#   - GOOGLE_CLOUD_PROJECT env var set
#   - Notification channel (email/Slack) already created
#
# Usage:
#   export GOOGLE_CLOUD_PROJECT=your-project-id
#   export NOTIFICATION_CHANNEL=projects/your-project-id/notificationChannels/12345
#   export DAILY_BUDGET_USD=50.00
#   bash scripts/setup_monitoring.sh

set -euo pipefail

# --- Configuration (from env vars, no hardcoded values) ---
PROJECT="${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
CHANNEL="${NOTIFICATION_CHANNEL:?Set NOTIFICATION_CHANNEL}"
DAILY_BUDGET="${DAILY_BUDGET_USD:-50.00}"
REGION="${CIV6AZ_GCP_REGION:-us-central1}"

echo "=== Civ6 AlphaZero Monitoring Setup ==="
echo "Project: ${PROJECT}"
echo "Region:  ${REGION}"
echo "Budget:  \$${DAILY_BUDGET}/day"
echo ""

# --- 1. Training Job Failure Alert ---
echo "[1/4] Creating training job failure alert..."
gcloud alpha monitoring policies create \
  --project="${PROJECT}" \
  --display-name="AlphaZero Training Job Failure" \
  --notification-channels="${CHANNEL}" \
  --condition-display-name="Vertex AI Job Failed" \
  --condition-filter='resource.type="aiplatform.googleapis.com/CustomJob" AND metric.type="aiplatform.googleapis.com/custom_job/completed_count" AND metric.labels.state="JOB_STATE_FAILED"' \
  --condition-threshold-value=1 \
  --condition-threshold-duration=0s \
  --condition-comparison=COMPARISON_GT \
  --aggregation-alignment-period=300s \
  --aggregation-per-series-aligner=ALIGN_COUNT \
  --documentation-content="A Civ6 AlphaZero training/selfplay/eval job has failed. Check Vertex AI console." \
  2>/dev/null || echo "  (alert may already exist — skipping)"

# --- 2. Cost Ceiling Alert ---
echo "[2/4] Creating daily cost ceiling alert..."
gcloud billing budgets create \
  --billing-account="$(gcloud billing projects describe "${PROJECT}" --format='value(billingAccountName)' 2>/dev/null || echo 'UNKNOWN')" \
  --display-name="AlphaZero Daily Budget" \
  --budget-amount="${DAILY_BUDGET}" \
  --threshold-rules-from-file=<(cat <<EOF
[
  {"threshold_percent": 0.50, "spend_basis": "CURRENT_SPEND"},
  {"threshold_percent": 0.80, "spend_basis": "CURRENT_SPEND"},
  {"threshold_percent": 1.00, "spend_basis": "CURRENT_SPEND"}
]
EOF
) \
  --filter-projects="projects/${PROJECT}" \
  --all-updates-rule-monitoring-notification-channels="${CHANNEL}" \
  2>/dev/null || echo "  (budget may already exist — skipping)"

# --- 3. GPU Utilization Alert ---
echo "[3/4] Creating low GPU utilization alert..."
gcloud alpha monitoring policies create \
  --project="${PROJECT}" \
  --display-name="AlphaZero Low GPU Utilization" \
  --notification-channels="${CHANNEL}" \
  --condition-display-name="GPU Utilization < 20%" \
  --condition-filter='resource.type="gce_instance" AND metric.type="agent.googleapis.com/gpu/utilization"' \
  --condition-threshold-value=0.20 \
  --condition-threshold-duration=600s \
  --condition-comparison=COMPARISON_LT \
  --aggregation-alignment-period=300s \
  --aggregation-per-series-aligner=ALIGN_MEAN \
  --documentation-content="GPU utilization is below 20% for 10+ minutes. Check if self-play or training is stalled." \
  2>/dev/null || echo "  (alert may already exist — skipping)"

# --- 4. Log-based Alert for Circuit Breaker ---
echo "[4/4] Creating circuit breaker log alert..."
gcloud alpha monitoring policies create \
  --project="${PROJECT}" \
  --display-name="AlphaZero Circuit Breaker Triggered" \
  --notification-channels="${CHANNEL}" \
  --condition-display-name="Circuit Breaker Log Entry" \
  --condition-filter='resource.type="global" AND log_name="projects/'"${PROJECT}"'/logs/vertex_orchestrator" AND textPayload=~"CIRCUIT BREAKER"' \
  --condition-threshold-value=1 \
  --condition-threshold-duration=0s \
  --condition-comparison=COMPARISON_GT \
  --aggregation-alignment-period=60s \
  --aggregation-per-series-aligner=ALIGN_COUNT \
  --documentation-content="The orchestrator circuit breaker triggered due to consecutive failures." \
  2>/dev/null || echo "  (alert may already exist — skipping)"

echo ""
echo "=== Monitoring setup complete ==="
echo "View alerts: https://console.cloud.google.com/monitoring/alerting?project=${PROJECT}"
