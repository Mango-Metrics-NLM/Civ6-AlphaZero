# DEV-SQE → Pre-PR Handoff

**Date:** 2026-11-10
**Spec:** kb/specs/20261110-public-release-readiness.md (OpenSpec change: openspec/changes/public-release-readiness)
**PR:** #132 (draft) on `claude/cool-lovelace-o02vn9`, 15 commits by reviewability class (see the review §8 commit map)
**Files Changed:** see the ledger event of this date; headline: deletion-only sweep (91 files), README v2 and docs index, `docs/EVALUATION.md`, evidence contract, community and metadata files, charter corrections (separate commit, maintainer ratifies), 2.0.0 roll, CI/supply-chain workflows, runner moves and identifier scrub, public-readiness gate, evaluation plumbing (benchmark protocol, additive draw-aware metrics, promotion feasibility, `--mcts`, learning oracle)
**Tests Added:** 13 new test modules (listed in the ledger event); every new gate was shown failing on a planted defect before it was kept
**Test Results:** `make verify` exit 0 — global coverage 87% (gate 85%), `alpha_zero/cloud` 60.0% (floor 58%); regression + cloud unit + governance 1,241 passed / 2 skipped; unit 9,714 passed / 58 skipped
**Lint Status:** clean (`ruff check .`, `ruff format --check .` 896 files)
**Type Check Status:** clean (`python -m mypy alpha_zero/`, 331 files)
**Smoke Tests:** `training_civ6 --dry_run` exit 0 (7.5 s); `orchestrator --preset fast --dry_run` exit 0; `validate_kb --git` OK on the full history; `validate_openspec --change public-release-readiness` OK; `check_doc_sync` OK
**Known Issues:** no post-fix evidence of learning exists yet (the README says so); the promotion default is unchanged and cannot promote at the measured draw rate (feasibility is now logged); GHAS-dependent scanners skip while the repository is private; the maintainer checklist in the review §9 gates the public flip (key rotation, IAM, default-branch rename after P5, protection with zero required approvals, `eval` extra reachability, charter ratification, tag v2.0.0, PR triage); CONFIRM-FIRST hunks for `distributed/coordinator.py` and `cloud/gcp/vertex_orchestrator.py` are recorded, not applied.
