# Project Context

> OpenSpec project metadata for **Civ6-AlphaZero**. This file is read by the
> spec workflow (and by any OpenSpec-aware agent) to understand scope and
> conventions before proposing a change. Keep it short and current.

## Purpose

Civ6-AlphaZero is an AlphaZero/MuZero-style reinforcement-learning system for
Civilization VI. It learns through self-play, using Monte Carlo Tree Search
guided by deep neural networks, and scales from a simplified training
environment (MicroCiv) to the live game via a headless FireTuner socket bridge.
The authoritative version is the ``version`` field in ``pyproject.toml`` — do
not hardcode it elsewhere.

## Tech stack

- **Language / runtime:** Python 3.11 and 3.12 (type hints throughout).
- **Core:** PyTorch, NumPy, Gymnasium.
- **Experiment tracking:** MLflow is the sole tracker (Weights & Biases was
  retired on 2026-07-13). ``alpha_zero/monitoring/`` is the MLflow package.
- **Hyperparameter search:** Optuna (``run_sweep.py``, ``configs/sweep_muzero.yaml``).
- **Cloud:** Google Cloud Vertex AI, GCS, Cloud Build (see ``infra/``, ``dockerfiles/``).
- **Quality tooling:** pytest, ruff 0.8.0 (100-char lines), mypy (advisory), bandit.

## Entry points

Invocation is via ``python -m`` — there are **no** installed console scripts.

- ``python -m alpha_zero.orchestrator`` — unified training orchestrator.
- ``python -m alpha_zero.web`` — Human-vs-AI web interface.
- ``python -m alpha_zero.cloud.gcp`` — Vertex AI job launcher.
- Top-level runners: ``run_sweep.py``, ``run_civ6_ai.py``, ``run_debug_selfplay.py``,
  ``run_enhanced_training.py``, ``run_muzero_training.py``, ``inspect_db.py``.

## Conventions

- **Coverage gate:** 85% on ``alpha_zero/`` (``pyproject.toml`` ``fail_under`` and
  CI ``--cov-fail-under=85`` must agree). Validator / operational code lives in
  ``scripts/`` with tests in ``tests/`` and is outside the gate.
- **Change budget:** at most 20 files modified per pass; deletions over 100 lines
  need human confirmation — prefer moving stale docs to ``docs/archive/``.
- **Governance validators** (``scripts/validate_openspec.py``, ``validate_kb.py``,
  ``validate_skills.py``, ``check_doc_sync.py``) are stdlib-only and enforced by
  Claude Code hooks, ``.pre-commit-config.yaml``, and the CI ``governance`` job.

## Governance pointers

- ``PROJECT_CHARTER.md`` — scope, non-goals, invariants, amendment protocol.
- ``CLAUDE.md`` — agent constraint spec and build/test command reference.
- ``docs/MODULE_REFERENCE.md`` — single source of truth for the module inventory.
- ``kb/`` — the Planning → DEV-SQE → Pre-PR handoff protocol.

## Spec workflow

A substantive change is proposed under ``openspec/changes/<change-id>/`` (a
``proposal.md``, ``tasks.md``, optional ``design.md``, and spec deltas) **before**
implementation. The binding structural gate is:

```bash
python scripts/validate_openspec.py          # whole tree
python scripts/validate_openspec.py --change <change-id>
```

The upstream npm ``openspec`` CLI is an optional convenience; the vendored Python
validator is authoritative for this repository.
