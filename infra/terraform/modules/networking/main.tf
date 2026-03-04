# Networking module: VPC, subnets, firewall rules, Cloud NAT, Serverless VPC
# Connector, Global HTTPS Load Balancer, Cloud Armor, and service accounts.
#
# This module provides the network foundation for all other modules.
# Cloud Run services reach the VPC (and SpacetimeDB) via the Serverless
# VPC Connector. External traffic enters via the Global HTTPS LB.

# ---------- VPC ----------

resource "google_compute_network" "main" {
  name                    = "${var.name_prefix}-vpc"
  project                 = var.project_id
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

# ---------- Subnets ----------

resource "google_compute_subnetwork" "services" {
  name          = "${var.name_prefix}-services"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.main.id
  ip_cidr_range = "10.0.1.0/24"

  private_ip_google_access = true

  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling        = 0.5
    metadata             = "INCLUDE_ALL_METADATA"
  }
}

# Subnet for Serverless VPC Connector (must be /28)
resource "google_compute_subnetwork" "serverless" {
  name          = "${var.name_prefix}-serverless"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.main.id
  ip_cidr_range = "10.0.2.0/28"
}

# ---------- Cloud Router + NAT ----------

resource "google_compute_router" "main" {
  name    = "${var.name_prefix}-router"
  project = var.project_id
  region  = var.region
  network = google_compute_network.main.id
}

resource "google_compute_router_nat" "main" {
  name                               = "${var.name_prefix}-nat"
  project                            = var.project_id
  region                             = var.region
  router                             = google_compute_router.main.name
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# ---------- Serverless VPC Connector ----------

resource "google_vpc_access_connector" "main" {
  provider = google-beta

  name          = "${var.name_prefix}-connector"
  project       = var.project_id
  region        = var.region

  subnet {
    name = google_compute_subnetwork.serverless.name
  }

  machine_type  = "e2-micro"
  min_instances = 2
  max_instances = 3
}

# ---------- Firewall Rules ----------

# Allow health checks from Google LB ranges
resource "google_compute_firewall" "allow_lb_health" {
  name    = "${var.name_prefix}-allow-lb-health"
  project = var.project_id
  network = google_compute_network.main.id

  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
    ports    = ["3000", "50000", "8080"]
  }

  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]
  target_tags   = ["spacetimedb"]
}

# Allow Cloud Run (via VPC Connector) to reach SpacetimeDB
resource "google_compute_firewall" "allow_serverless" {
  name    = "${var.name_prefix}-allow-serverless"
  project = var.project_id
  network = google_compute_network.main.id

  direction = "INGRESS"
  priority  = 1100

  allow {
    protocol = "tcp"
    ports    = ["3000", "50000"]
  }

  source_ranges = [google_compute_subnetwork.serverless.ip_cidr_range]
  target_tags   = ["spacetimedb"]
}

# Allow IAP-tunneled SSH for management
resource "google_compute_firewall" "allow_iap_ssh" {
  name    = "${var.name_prefix}-allow-iap-ssh"
  project = var.project_id
  network = google_compute_network.main.id

  direction = "INGRESS"
  priority  = 1200

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"] # IAP range
  target_tags   = ["spacetimedb"]
}

# Default deny all ingress
resource "google_compute_firewall" "deny_all_ingress" {
  name    = "${var.name_prefix}-deny-all-ingress"
  project = var.project_id
  network = google_compute_network.main.id

  direction = "INGRESS"
  priority  = 65534

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]
}

# ---------- Global External IP for Load Balancer ----------

resource "google_compute_global_address" "lb" {
  name    = "${var.name_prefix}-lb-ip"
  project = var.project_id
}

# ---------- Cloud Armor Security Policy ----------

resource "google_compute_security_policy" "main" {
  count = var.enable_cloud_armor ? 1 : 0

  name    = "${var.name_prefix}-waf"
  project = var.project_id

  # Default rule: allow
  rule {
    action   = "allow"
    priority = 2147483647

    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }

    description = "Default allow rule"
  }

  # Block known bad IPs (rate limiting)
  rule {
    action   = "rate_based_ban"
    priority = 1000

    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }

    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"

      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }

      ban_duration_sec = 300
    }

    description = "Rate limit: 100 req/min per IP, ban for 5 min on exceed"
  }
}

# ---------- Service Accounts ----------

resource "google_service_account" "inference" {
  account_id   = "${var.name_prefix}-inference"
  display_name = "Civ6AZ AI Inference Service"
  project      = var.project_id
}

resource "google_service_account" "spacetimedb" {
  account_id   = "${var.name_prefix}-stdb"
  display_name = "Civ6AZ SpacetimeDB Service"
  project      = var.project_id
}

resource "google_service_account" "auth" {
  account_id   = "${var.name_prefix}-auth"
  display_name = "Civ6AZ Auth Service"
  project      = var.project_id
}

resource "google_service_account" "matchmaker" {
  account_id   = "${var.name_prefix}-match"
  display_name = "Civ6AZ Matchmaking Service"
  project      = var.project_id
}

# IAM bindings for service accounts

resource "google_project_iam_member" "inference_storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.inference.email}"
}

resource "google_project_iam_member" "spacetimedb_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.spacetimedb.email}"
}

resource "google_project_iam_member" "auth_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.auth.email}"
}

resource "google_project_iam_member" "matchmaker_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.matchmaker.email}"
}
