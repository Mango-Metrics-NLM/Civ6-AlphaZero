#!/usr/bin/env python3
"""TLS certificate inspection for GCP endpoints.

Detects corporate MITM / TLS-inspection proxies by checking who issued
the certificate for each Google Cloud endpoint.  A trusted Google cert
will show issuer org = 'Google Trust Services'.  A corporate proxy will
show a different org (e.g., 'Zscaler', 'Palo Alto Networks', etc.).

Usage:
    python scripts/run_cert_check.py
"""
from __future__ import annotations

import json
import socket
import ssl
import sys

ENDPOINTS = [
    "oauth2.googleapis.com",
    "accounts.google.com",
    "storage.googleapis.com",
    "us-central1-aiplatform.googleapis.com",
    "cloudresourcemanager.googleapis.com",
]


def check_cert(hostname: str, port: int = 443) -> dict:
    ctx = ssl.create_default_context()
    try:
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as sock:
            sock.settimeout(8)
            sock.connect((hostname, port))
            cert = sock.getpeercert()

        issuer = dict(x[0] for x in cert.get("issuer", []))
        subject = dict(x[0] for x in cert.get("subject", []))
        sans = [v for _, v in cert.get("subjectAltName", [])[:4]]
        return {
            "hostname": hostname,
            "status": "ok",
            "subject_CN": subject.get("commonName", ""),
            "issuer_O": issuer.get("organizationName", ""),
            "issuer_CN": issuer.get("commonName", ""),
            "not_after": cert.get("notAfter", ""),
            "sans": sans,
        }
    except ssl.SSLCertVerificationError as exc:
        return {"hostname": hostname, "status": "ssl_error", "error": str(exc)}
    except Exception as exc:
        return {"hostname": hostname, "status": "error", "error": str(exc)}


def main() -> int:
    results = [check_cert(ep) for ep in ENDPOINTS]

    mitm_detected = False
    print("\n=== GCP TLS Certificate Issuer Report ===\n")
    for r in results:
        if r["status"] != "ok":
            print(f"  [ERROR] {r['hostname']}: {r['error']}")
            continue

        is_google = "google" in r["issuer_O"].lower()
        flag = "" if is_google else "  <-- POSSIBLE MITM/PROXY"
        if not is_google:
            mitm_detected = True
        print(f"  {'[OK]   ' if is_google else '[WARN] '} {r['hostname']}")
        print(f"           issuer:   {r['issuer_O']} / {r['issuer_CN']}{flag}")
        print(f"           subject:  {r['subject_CN']}")
        print(f"           expires:  {r['not_after']}")

    print()
    if mitm_detected:
        print("RESULT: Corporate TLS inspection (MITM proxy) detected.")
        print("        Google's cert is being replaced by a corporate CA.")
        print()
        print("  Options to unblock GCP OAuth:")
        print("  A) Configure gcloud to trust your corporate root CA:")
        print("       gcloud config set core/custom_ca_certs_file <path-to-corp-root.pem>")
        print("       Set REQUESTS_CA_BUNDLE=<path-to-corp-root.pem> (for Python SDK)")
        print()
        print("  B) Export the intercepting cert from your browser and set:")
        print("       $env:REQUESTS_CA_BUNDLE = 'C:\\path\\to\\corp-root.pem'")
        print("       $env:GOOGLE_API_USE_CLIENT_CERTIFICATE = 'false'")
        print()
        print("  C) Switch to a non-inspected network (hotspot, home VPN)")
        print("       or ask IT to whitelist *.googleapis.com from TLS inspection.")
    else:
        print("RESULT: All certs issued by Google Trust Services — no MITM proxy detected.")
        print("        TLS inspection is NOT the root cause of your OAuth issue.")

    print()
    print("Raw JSON:")
    print(json.dumps(results, indent=2))
    return 1 if mitm_detected else 0


if __name__ == "__main__":
    sys.exit(main())
