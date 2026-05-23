# SpacetimeDB module: GCE instance running SpacetimeDB with persistent disk,
# health checks, and scheduled disk snapshots for backup.
#
# SpacetimeDB is deployed on a dedicated GCE instance because it requires:
# - Persistent WebSocket connections (Cloud Run has 60-min timeout)
# - Local state storage (WAL, indexes)
# - Predictable latency (no cold starts)

# ---------- Instance ----------

resource "google_compute_instance" "spacetimedb" {
  name         = "${var.name_prefix}-spacetimedb"
  project      = var.project_id
  zone         = var.zone
  machine_type = var.machine_type

  tags = ["spacetimedb"]

  labels = var.labels

  boot_disk {
    initialize_params {
      image = "projects/cos-cloud/global/images/family/cos-stable"
      size  = var.disk_size_gb
      type  = "pd-ssd"
    }
  }

  network_interface {
    subnetwork = var.subnet_id

    # No external IP -- access via IAP tunnel or internal LB only
  }

  service_account {
    email  = var.service_account
    scopes = ["cloud-platform"]
  }

  # Startup script pulls and runs the SpacetimeDB container
  metadata = {
    gce-container-declaration = yamlencode({
      spec = {
        containers = [{
          image = var.spacetimedb_image
          name  = "spacetimedb"
          ports = [
            { containerPort = 3000 },  # HTTP/WebSocket
            { containerPort = 50000 }, # gRPC
          ]
          volumeMounts = [{
            name      = "stdb-data"
            mountPath = "/var/lib/spacetimedb"
          }]
          env = [
            { name = "SPACETIMEDB_LOG_LEVEL", value = "info" },
          ]
        }]
        volumes = [{
          name = "stdb-data"
          hostPath = {
            path = "/mnt/disks/stdb-data"
          }
        }]
        restartPolicy = "Always"
      }
    })

    # Enable container-optimized OS logging to Cloud Logging
    google-logging-enabled = "true"
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
  }

  # Allow Terraform to recreate if boot disk changes
  allow_stopping_for_update = true

  lifecycle {
    ignore_changes = [
      metadata["ssh-keys"],
    ]
  }
}

# ---------- Health Check ----------

resource "google_compute_health_check" "spacetimedb" {
  name    = "${var.name_prefix}-stdb-health"
  project = var.project_id

  check_interval_sec  = 10
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 3000
    request_path = "/health"
  }
}

# ---------- Scheduled Disk Snapshot (Backup) ----------

resource "google_compute_resource_policy" "snapshot_schedule" {
  name    = "${var.name_prefix}-stdb-snapshots"
  project = var.project_id
  region  = var.region

  snapshot_schedule_policy {
    schedule {
      hourly_schedule {
        hours_in_cycle = 1
        start_time     = "00:00"
      }
    }

    retention_policy {
      max_retention_days    = 7
      on_source_disk_delete = "KEEP_AUTO_SNAPSHOTS"
    }

    snapshot_properties {
      labels            = var.labels
      storage_locations = [var.region]
    }
  }
}

resource "google_compute_disk_resource_policy_attachment" "snapshot" {
  name    = google_compute_resource_policy.snapshot_schedule.name
  disk    = google_compute_instance.spacetimedb.boot_disk[0].source
  zone    = var.zone
  project = var.project_id
}
