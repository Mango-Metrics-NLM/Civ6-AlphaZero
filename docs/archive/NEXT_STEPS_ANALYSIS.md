# Next-Steps Analysis — Civ6-AlphaZero (2026-08-11)

_A codebase + documentation scan to determine the best logical development steps forward._
_Generated against the Phase 21 integration tip (v1.8.0). CI on `main` is enforcing 85% test coverage and 100% mypy strict checks._

---

## TL;DR

The build is heavily fortified (green CI, 100% mypy --strict compliance, 85%+ test coverage) and fully integrated with external orchestration ecosystems (Hugging Face Hub, SpaceTimeDB, and MLflow). The primary focus has shifted from stabilization and integration to **scaling the training loop** and **expanding inference serving**.

**Recommended order:**

1. **SpaceTimeDB Sync Stabilization:** Ensure the SpaceTimeDB module gracefully handles large scale concurrent connections from evaluation workers without hitting limits.
2. **Expand the ELO Tournament Dataset:** Continuously run `tournament_worker.py` and push exports via `export_dataset.py` to the Hugging Face Hub.
3. **Enhance MicroCiv Explorer UI:** Now that the Space is live, add more visual replay features to the Gradio interface for analyzing MuZero agent paths.
4. **Agent Skill Refinement:** Continue curating `.agents/skills` based on recurring architecture patterns (like `type-fortification` and `hub-publish-validator`).

---

## 1. Documentation & Quality Gates (Fortified)

The project has achieved remarkable code hygiene and stability:
- **Test Coverage:** Enforced at **85%** in `Makefile` and `ci.yml`.
- **Type Safety:** The entire `alpha_zero/` module is **100% mypy --strict** compliant, guided by the `type-fortification-skill`.
- **Governance:** `check_doc_sync.py` now enforces parity between SpaceTimeDB models, Hub model cards, and documentation.

## 2. Infrastructure Integrations

- **Hugging Face Hub & Spaces:** `make publish-spaces` automatically validates and pushes Hugging Face spaces (`microciv-explorer`, `elo-dashboard`). Checkpoint pipelines extract and upload encoder submodules.
- **SpaceTimeDB:** Successfully integrated for multi-player tournament synchronization, tracking player ELO ratings directly via Rust modules.
- **GCP/Vertex AI:** Preemption handling and spot VM orchestration remain stable.

## 3. The Chess / LM-Studio Direction

The chess environment (`envs/chess_env.py`) and Qwen advisor are functional and part of the codebase. The previous packaging bugs and sequential call bottlenecks have been largely addressed.
- **Action:** Continue monitoring the batching logic for Qwen API limits if moving to a hosted inference provider instead of LM-Studio.

---

## Recommended path forward

### Track A — Hugging Face Hub Operations
1. Maintain the flow of trained models to the Hub using `hub-release` and `hub-publish-validator` skills.
2. Validate that the models deployed on the `microciv-explorer` Space behave accurately under continuous heavy load.

### Track B — SpaceTimeDB
3. Profile the database sync process (`alpha_zero/spacetimedb/`) under load (e.g. 100+ concurrent evaluation workers).

### Track C — MuZero Scaling
4. Run long-horizon MuZero training on GCP and publish intermediate evaluations to the ELO dashboard.

### Decision needed from maintainer
- **Strategic Planning SDK (#49/#32):** Is it time to integrate these changes fully into the core, or keep them as distinct experimental offshoots?

---

## Single best next step

If one thing: **Run a long-horizon (10k+ iteration) MuZero training sweep on Vertex AI.** Watch the GCP billing thresholds via `CostMonitor`, let the preemption handler do its job, and observe the continuous integration of ELO metrics into the SpaceTimeDB and Hugging Face dashboard.
