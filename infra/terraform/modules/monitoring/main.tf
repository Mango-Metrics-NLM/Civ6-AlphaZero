# Monitoring module: Cloud Monitoring dashboards, alert policies, uptime
# checks, and log-based metrics for the Civ6 AlphaZero gameplay system.

# ---------- Notification Channel ----------

resource "google_monitoring_notification_channel" "email" {
  count = var.alert_email != "" ? 1 : 0

  display_name = "${var.name_prefix} Alert Email"
  project      = var.project_id
  type         = "email"

  labels = {
    email_address = var.alert_email
  }
}

locals {
  notification_channels = var.alert_email != "" ? [google_monitoring_notification_channel.email[0].id] : []
}

# ---------- Uptime Checks ----------

resource "google_monitoring_uptime_check_config" "inference" {
  display_name = "${var.name_prefix}-inference-uptime"
  project      = var.project_id
  timeout      = "10s"
  period       = "60s"

  http_check {
    path         = "/health"
    port         = 443
    use_ssl      = true
    validate_ssl = true
  }

  monitored_resource {
    type = "cloud_run_revision"
    labels = {
      project_id   = var.project_id
      service_name = var.inference_service_name
      location     = "us-central1"
    }
  }
}

# ---------- Alert: AI Inference Latency ----------

resource "google_monitoring_alert_policy" "inference_latency" {
  display_name = "${var.name_prefix}: AI Inference p99 Latency > 2s"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Cloud Run request latency p99"

    condition_threshold {
      filter          = <<-EOT
        resource.type = "cloud_run_revision"
        AND resource.labels.service_name = "${var.inference_service_name}"
        AND metric.type = "run.googleapis.com/request_latencies"
      EOT
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 2000 # milliseconds

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_PERCENTILE_99"
        cross_series_reducer = "REDUCE_MAX"
      }
    }
  }

  notification_channels = local.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content   = "AI inference p99 latency exceeds 2 seconds. Check GPU utilization, model loading times, and MCTS search depth. Consider scaling up instances or reducing MCTS simulations."
    mime_type = "text/markdown"
  }
}

# ---------- Alert: Inference Error Rate ----------

resource "google_monitoring_alert_policy" "inference_errors" {
  display_name = "${var.name_prefix}: AI Inference Error Rate > 1%"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Cloud Run 5xx error rate"

    condition_threshold {
      filter          = <<-EOT
        resource.type = "cloud_run_revision"
        AND resource.labels.service_name = "${var.inference_service_name}"
        AND metric.type = "run.googleapis.com/request_count"
        AND metric.labels.response_code_class = "5xx"
      EOT
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.01 # 1%

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = local.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content   = "AI inference service error rate exceeds 1%. Check logs for OOM errors, model loading failures, or GPU issues. Consider rolling back to previous model version."
    mime_type = "text/markdown"
  }
}

# ---------- Alert: SpacetimeDB Instance Down ----------

resource "google_monitoring_alert_policy" "spacetimedb_down" {
  display_name = "${var.name_prefix}: SpacetimeDB Instance Not Running"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "GCE instance uptime"

    condition_threshold {
      filter          = <<-EOT
        resource.type = "gce_instance"
        AND resource.labels.instance_id = "${var.spacetimedb_instance}"
        AND metric.type = "compute.googleapis.com/instance/uptime"
      EOT
      duration        = "120s"
      comparison      = "COMPARISON_LT"
      threshold_value = 1

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = local.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }

  documentation {
    content   = "SpacetimeDB GCE instance appears to be down. All active games will be affected. Check GCE console for instance status, disk errors, or preemption events."
    mime_type = "text/markdown"
  }
}

# ---------- Alert: High Cost ----------

resource "google_monitoring_alert_policy" "daily_cost" {
  display_name = "${var.name_prefix}: Daily Cost Exceeds $10"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Billing cost threshold"

    condition_threshold {
      filter          = <<-EOT
        resource.type = "global"
        AND metric.type = "billing/gcp/cost"
      EOT
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 10.0

      aggregations {
        alignment_period     = "86400s" # 1 day
        per_series_aligner   = "ALIGN_SUM"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = local.notification_channels

  alert_strategy {
    auto_close = "86400s"
  }

  documentation {
    content   = "Daily GCP cost exceeds $10 (projected $300/month). Review active resources, check for runaway GPU instances, and verify scale-to-zero is working."
    mime_type = "text/markdown"
  }
}

# ---------- Log-Based Metric: Auth Failures ----------

resource "google_logging_metric" "auth_failures" {
  name    = "${var.name_prefix}-auth-failures"
  project = var.project_id
  filter  = <<-EOT
    resource.type = "cloud_run_revision"
    AND resource.labels.service_name = "${var.auth_service_name}"
    AND textPayload =~ "authentication failed"
  EOT

  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"

    display_name = "Auth Failures"
  }
}

# ---------- Dashboard ----------

resource "google_monitoring_dashboard" "main" {
  project        = var.project_id
  dashboard_json = jsonencode({
    displayName = "${var.name_prefix} Gameplay Operations"
    gridLayout = {
      columns = 3
      widgets = [
        # Row 1: Request rates
        {
          title = "AI Inference: Requests/sec"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${var.inference_service_name}\" AND metric.type = \"run.googleapis.com/request_count\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                  }
                }
              }
            }]
          }
        },
        {
          title = "Matchmaking: Requests/sec"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${var.matchmaking_service_name}\" AND metric.type = \"run.googleapis.com/request_count\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                  }
                }
              }
            }]
          }
        },
        {
          title = "SpacetimeDB: CPU Utilization"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type = \"gce_instance\" AND metric.type = \"compute.googleapis.com/instance/cpu/utilization\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
            }]
          }
        },
        # Row 2: Latency
        {
          title = "AI Inference: Latency (p50/p95/p99)"
          xyChart = {
            dataSets = [
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${var.inference_service_name}\" AND metric.type = \"run.googleapis.com/request_latencies\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_50"
                    }
                  }
                }
                legendTemplate = "p50"
              },
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${var.inference_service_name}\" AND metric.type = \"run.googleapis.com/request_latencies\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_95"
                    }
                  }
                }
                legendTemplate = "p95"
              },
              {
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${var.inference_service_name}\" AND metric.type = \"run.googleapis.com/request_latencies\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_PERCENTILE_99"
                    }
                  }
                }
                legendTemplate = "p99"
              }
            ]
          }
        },
        {
          title = "Cloud Run: Instance Count"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type = \"cloud_run_revision\" AND metric.type = \"run.googleapis.com/container/instance_count\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_MAX"
                  }
                }
              }
            }]
          }
        },
        {
          title = "SpacetimeDB: Disk Utilization"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "resource.type = \"gce_instance\" AND metric.type = \"compute.googleapis.com/instance/disk/read_bytes_count\""
                  aggregation = {
                    alignmentPeriod  = "60s"
                    perSeriesAligner = "ALIGN_RATE"
                  }
                }
              }
            }]
          }
        },
      ]
    }
  })
}
