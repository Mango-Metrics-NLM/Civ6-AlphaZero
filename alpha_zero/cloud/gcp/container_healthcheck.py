"""Container health check — validates all critical imports inside the Docker container.

Usage:
    python -m alpha_zero.cloud.gcp.container_healthcheck

Exit codes:
    0 — all modules imported successfully
    1 — one or more modules failed to import
"""

from __future__ import annotations

import importlib
import sys
import time

# Modules to validate, organized by phase
PHASE_MODULES: dict[str, list[str]] = {
    "Phase 1-3 (Core)": [
        "alpha_zero.core.civ6_network",
        "alpha_zero.core.civ6_mcts",
        "alpha_zero.core.mcts_factory",
        "alpha_zero.envs.microciv",
    ],
    "Phase 4 (Training)": [
        "alpha_zero.training.convergence_monitor",
        "alpha_zero.training.muzero_trainer",
    ],
    "Phase 5 (RAG/Utils)": [
        "alpha_zero.utils.logging_config",
        "alpha_zero.utils.config",
    ],
    "Phase 7 (GCP Pipeline)": [
        "alpha_zero.cloud.gcp.entrypoint",
        "alpha_zero.cloud.gcp.selfplay_worker",
        "alpha_zero.cloud.gcp.training_worker",
        "alpha_zero.cloud.gcp.evaluation_worker",
        "alpha_zero.cloud.gcp.vertex_jobs",
        "alpha_zero.cloud.gcp.vertex_orchestrator",
        "alpha_zero.cloud.gcp.storage",
        "alpha_zero.cloud.gcp.config",
    ],
    "Phase 8 (Safeguards)": [
        "alpha_zero.training.training_safeguards",
        "alpha_zero.training.training_report",
    ],
    "Phase 9 (MuZero)": [
        "alpha_zero.training.muzero_integration",
        "alpha_zero.pbt.muzero_hyperparams",
    ],
    "Phase 10 (Tournament)": [
        "alpha_zero.tournament.orchestrator",
        "alpha_zero.tournament.analysis",
        "alpha_zero.tournament.wandb_dashboards",
    ],
    "PBT": [
        "alpha_zero.pbt.config",
        "alpha_zero.pbt.coordinator",
        "alpha_zero.pbt.evaluator",
        "alpha_zero.cloud.gcp.pbt_orchestrator",
    ],
}


def main() -> int:
    """Run container health check."""
    start = time.monotonic()
    errors: list[str] = []
    total = 0

    for phase, modules in PHASE_MODULES.items():
        phase_ok = True
        for mod_name in modules:
            total += 1
            try:
                importlib.import_module(mod_name)
            except Exception as exc:
                errors.append(f"  {mod_name}: {type(exc).__name__}: {exc}")
                phase_ok = False

        status = "OK" if phase_ok else "FAIL"
        print(f"[{status}] {phase} ({len(modules)} modules)")

    elapsed = time.monotonic() - start

    if errors:
        print(f"\n{len(errors)} import failures:", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    print(f"\nAll {total} modules imported successfully in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
