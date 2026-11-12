# Tasks

> Boxes ticked at archive (2026-11-11, task T13) from a spot-check of every
> deliverable; the work itself landed in PR #132 (`8ea973e`, `b408083`).

## 0. Review and governance (P0)
- [x] 0.1 Publish `docs/audits/2026-11-10-public-release-review.md` with the Sept-audit status table and the maintainer checklist
- [x] 0.2 Scaffold this change and keep `python scripts/validate_openspec.py --change public-release-readiness` green

## 1. Deletion-only slice (P1)
- [x] 1.1 Remove `replays/`, `demos/`, root `interfaces/`, `templates/{src,tests,pyproject.toml}`, `mcp_config.json`, `configs/wandb_dashboard.json`, root `Dockerfile`, `GeneratedPaths.lua` and their tests
- [x] 1.2 Update coupling config (pyproject excludes, `.dockerignore`, dependabot, suite-wiring parent list)
- [x] 1.3 Re-measure `docs/evidence/tech_debt_baseline/baseline.json` in the same change

## 2. Moves, scrub and readiness gate (P2)
- [x] 2.1 Move `run_*.py` to `scripts/`; move Rust stubs under `experimental/`; move `templates/*.md` to `docs/templates/`
- [x] 2.2 Remove the personal email, project number and service-account addresses; make bucket defaults fail closed; `$PROJECT_ID` in Cloud Build; `*.example.yaml` + gitignored `configs/local/`
- [x] 2.3 Add `tests/regression/test_public_readiness_regression.py` with its manifest; show it red before the scrub and green after
- [x] 2.4 Close the ignore-file gaps (`*.safetensors`, `*.ckpt`, `*.npz`, `*.onnx`, `*.bin`)

## 3. Public surface (P3)
- [x] 3.1 README v2 (status table, one Hub link, trademark line, citation)
- [x] 3.2 `docs/README.md` index with Diátaxis labels; stubs removed; `NEXT_STEPS.md` → `docs/LIVE_CIV6.md`
- [x] 3.3 `docs/EVALUATION.md` with the power table and a test that recomputes it
- [x] 3.4 Evidence contract (`docs/evidence/SCHEMA.md`, index, backfill, pre-fix banners, A/B paragraph, model-card corrections)
- [x] 3.5 `CITATION.cff`, `GOVERNANCE.md`, `.editorconfig`, `.python-version`, `py.typed`, pyproject metadata
- [x] 3.6 ROADMAP validation column and Now/Next/Later; SUPPORT/SECURITY/CoC/CONTRIBUTING updates; `kb/README.md` and `.claude/README.md` framing
- [x] 3.7 `check_doc_sync.py` D4b, D6′, D8, D9, CFF version check; CLAUDE.md citation-resolution test

## 4. Charter corrections (P4, maintainer ratifies)
- [x] 4.1 Fix charter links, M1 wording, speed-up claims and the Python floor in a charter-only commit (done in the charter-only commit of PR #132; ratification stays a maintainer item, gap LF-G4)

## 5. CI and supply chain (P5)
- [x] 5.1 `ci.yml`: branch-agnostic push trigger, cache keys, `fetch-depth: 0`, ratchet base fix, bandit baseline step, 3.13 experimental, contributor smoke
- [x] 5.2 `security-scan.yml`, `codeql.yml`, `scorecard.yml` (non-blocking)
- [x] 5.3 `release.yml`: version/tag assertion, build, checksums, notes from CHANGELOG, provenance step gated CONFIRM-FIRST (the provenance step ships disabled, CONFIRM-FIRST, as specified)
- [x] 5.4 `deploy_spaces.yml`: no push trigger, pinned installs, environment gate, unified names
- [x] 5.5 `validate_kb.py --git` enforced; pre-commit bumps and gitleaks hook; Makefile dead targets; pinning test extensions

## 6. Release (P6)
- [x] 6.1 Roll `[Unreleased]` into `[2.0.0]` (summary + archived detail); bump versions; fix changelog ordering; extend D7

## 7. Evaluation plumbing (P7, milestone v2.1)
- [x] 7.1 `run_arena` search seed and paired seats; `--mcts` flag; evaluation-worker seed
- [x] 7.2 `BenchmarkProtocol` over `run_arena`; non-vacuous smoke test
- [x] 7.3 Additive draw-aware metrics and the AST/pairing regression gate
- [x] 7.4 Promotion feasibility report and corrected runbook; Elo transform
- [x] 7.5 Gumbel target-entropy measurement and documented deviations; closed-loop learning oracle
