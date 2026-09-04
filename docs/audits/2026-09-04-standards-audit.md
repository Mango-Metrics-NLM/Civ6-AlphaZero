# 2026 Coding-Standards Audit — Civ6-AlphaZero

**Date:** 2026-09-04 · **Commit audited:** `81c3dad` (post-PR #117) · **Branch:** `claude/2026-standards-audit-7tgk7e`
**Method:** six parallel read-only audit passes (CI/supply chain, application security, code
quality/architecture, test suite, docs/governance/DX, and an external 2026-practice rubric built
from primary sources), with every gate-integrity and high-severity claim reproduced by hand before
it was accepted. Nothing in the tree was modified except the addition of this file.

Confidence tags: **[Certain]** = reproduced by command in this session · **[Likely]** = agent
evidence with file:line, not independently re-run · **[Guessing]** = inference.

---

## 1. Verdict

**No.** The project meets the 2026 bar on *process discipline* — deterministic governance gates,
a real negative-control culture, seeded reproducibility, ratchets on tech debt — and it is above
average on that axis. It does **not** meet the bar on the things a 2026 reviewer checks first:
supply-chain reproducibility, dependency vulnerability posture, release engineering, and the
security of the one deployed public surface. Several of its own gates are also fail-open in ways
that make the green CI badge overstate what is actually protected.

Summary score (0–5 per area, 5 = meets or exceeds current best practice):

| Area | Score | One-line reason |
|---|---|---|
| Lint / format / type gates | **4** | ruff + format + mypy all clean [Certain]; 100% PEP 604; but `ANN401` globally off (1,814 `Any`), 10/29 packages strict, no `py.typed` |
| Governance / meta-gates | **4** | Validators, suite-wiring gate and hook tests are real and have negative controls [Certain] |
| Testing volume & structure | **4** | ~12k tests, 1.25 test:source line ratio, 11 CI-wired suites [Likely] |
| Testing quality | **2** | 166 zero-assert tests, 57 files unseeded, no `pytest-randomly`, 47% of source lines outside the coverage gate [Likely] |
| Code quality / architecture | **2** | 24 files >1k lines, 13 mutual subpackage cycles, 24,451 dead lines (16.7%), 192 `*Config` classes with 24 duplicate names [Likely] |
| Dependency management | **1** | `uv.lock` unused and stale (fails `uv lock --check`), three drifting unpinned requirements files, 10 Dependabot PRs open / 0 ever merged [Certain] |
| Vulnerability posture | **1** | No `pip-audit`/OSV in CI; `mlflow<3.0` pin holds 31 open advisories in the deployed image; bandit gate sees 0 of 76 findings [Likely] |
| Application security | **2** | Public unauthenticated GPU inference endpoint; wildcard CORS with credentials; 7 unsafe `torch.load` + 5 `pickle.loads` on storage-sourced bytes [Certain for the first two] |
| CI/CD & release | **1** | 14/15 action refs on mutable tags; zero git tags ever, so `release.yml` has never run; no SBOM/provenance/scan [Certain] |
| Docs / onboarding | **2** | Documented `pip install -e '.[dev]'` fails on `chess` [Certain]; CLAUDE.md is 98 KB (~17–25k tokens) vs Anthropic's <200-line guidance and contradicts the tree in 8 places [Certain] |

**Overall: 2.3 / 5.** A 2026-standard codebase would score ≥4 on every row.

---

## 2. What is genuinely above the bar

Credit where it is measured, because the remediation list below should not erase it.

- **Gates that prove they gate.** `tests/regression/test_suite_wiring_regression.py` parses CI
  with `yaml`, follows pytest's collection rule, and carries a negative-control class. The
  governance suite runs the real PreToolUse hook via subprocess and asserts the JSON decision.
  Coverage omit lists and debt metrics ratchet only one way. This is rarer than it should be.
- **Reproducibility is a tested contract** (`--seed` on training and eval, ~7 s determinism
  regression test), and the evidence directory records how it was found: by diffing runs, not
  by auditing call sites.
- **Lint/type hygiene is real, not aspirational.** `ruff check .`, `ruff format --check .`
  (882 files) and `mypy alpha_zero/` (328 files) are all clean in this session [Certain].
  Zero `typing.Optional`/`Union`; `from __future__ import annotations` in 300/328 modules.
- **Least-privilege workflows** (`permissions: contents: read` at workflow level, job-level
  escalation only in release), concurrency groups and timeouts on every job.
- **Secret hygiene on the tracked tree**: gitleaks pinned once, 0 findings on HEAD's lineage;
  `.gitignore`/`.dockerignore` cover `.env*`, keys, checkpoints.
- **Safe-load discipline** was recently centralised (`utils/safe_load.py`) with an AST gate that
  blocks *growth* of unsafe `torch.load` sites.

---

## 3. Findings that decide the verdict

Ordered by how much they change the risk picture, not by count.

### 3.1 Security (application + supply chain)

| # | Finding | Sev | Evidence | Gated today? |
|---|---|---|---|---|
| S1 | **Inference endpoint is public and unauthenticated.** `serving/inference_server.py` has no auth dependency; Terraform binds `roles/run.invoker` to `allUsers` with a comment promising Firebase JWT "at the application layer" that does not exist; `cloud-run-inference.yaml` sets `ingress: all` on an L4 GPU autoscaling to 10. Cost-DoS and free model extraction. [Certain] | High | `infra/terraform/modules/ai-inference/main.tf:129-137`, `cloud-run-inference.yaml:11` | No |
| S2 | **Unrotated W&B API key reachable in history** (commit `87cd68a2`, 6 occurrences, on a remote branch). The runbook still says UNREMEDIATED; the CI gate scopes to HEAD's lineage by design, and `make security-secrets-audit` is `\|\| true`. [Likely] | High | `docs/runbooks/secret-scanning.md:28`, `Makefile:262-264` | No |
| S3 | **RCE from storage-sourced bytes.** 7 unconditional `weights_only=False` and 5 `pickle.loads` on checkpoints/trajectories read from GCS, local storage and SpacetimeDB. Bucket-write = code execution on every trainer. [Likely] | High | `storage/gcs_backend.py:469,558`, `cloud/gcp/storage.py:765`, `spacetimedb/serialization.py:182`, `orchestrator/checkpointer.py:256`, `distributed/coordinator.py:397` | Growth only |
| S4 | **Web UI: wildcard CORS + `allow_credentials=True` + bind-all default, no auth on 22 routes or the websocket.** `DELETE /api/replay/{id}` reachable cross-origin. [Certain] | Med | `web/config.py:69,102`, `web/app.py:91-92` | No (`noqa: S104`) |
| S5 | **31 open advisories in the deployed image** via `mlflow>=2.12,<3.0`; every fix is 3.x. Plus pyarrow and chromadb advisories. No `pip-audit`/OSV job exists to see them. [Likely] | Med | `pyproject.toml:70`, `requirements-container.txt:15` | No |
| S6 | **Bandit gate is vacuous.** `--severity-level high --confidence-level high` on `alpha_zero/` yields 0 findings; the unfiltered scan yields 76 (2 HIGH, 28 MEDIUM), and `scripts/`/`infra/` are never scanned. [Likely] | Med | `ci.yml:205` | Self-defeating |
| S7 | **PreToolUse push guard is bypassed by any wrapper.** `bash -c "git push --force origin main"`, `sh -c`, `eval`, `xargs`, and `gh pr merge` all pass; a pre-approved `scripts/vm_ssh.py` executes remote shell via `gcloud`, bypassing the `gcloud` deny. [Certain for `bash -c`] | Med | `.claude/hooks/pretooluse_guard.sh`, `.claude/settings.json` | No |
| S8 | Windows path traversal in replay `game_id` (no validation; single-segment route param blocks it on POSIX only). [Likely] | Low | `human_interface/replay.py:541-544,631` | No |

### 3.2 Gates that are green but fail open

These matter more than any single bug because they are what the project *relies on* to say it is
healthy.

| # | Gate | What actually happens | Evidence |
|---|---|---|---|
| G1 | **Dead-code ratchet** (`test_debt_ratchet.py`) | `audit_import_graph.py` skips `.git`, `__pycache__`, venvs and `egg-info` but **not `build/`, `dist/`, `.tox/`**. A `build/lib/alpha_zero` copy (what `python -m build` or an editable install leaves) makes a second `__init__.py` per package look like an external re-exporter and revives 23 dead modules: **53 → 30 modules, 24,451 → 15,120 lines** on an identical tree. Reproduced on a scratch clone. [Certain] | `scripts/audit_import_graph.py:67` |
| G2 | **Dev install** | `pip install -e '.[dev]'` — the command in README, CONTRIBUTING and CLAUDE.md §6.1 — aborts on the `chess` sdist (`AttributeError: install_layout`) on a Debian/Ubuntu Python. CLAUDE.md §15.3 documents the failure rather than removing `chess` from `dev`. When it fails, **nothing** in the extra installs. [Certain] | `pyproject.toml:35,84` |
| G3 | **Unit suite collection** | 7 unit and 2 integration modules bare-`import chess` instead of `importorskip`; without it `pytest tests/unit` exits 2 with 0 tests run. The wiring gate's optional-import check scans one file for two names. [Likely] | `tests/unit/test_chess_actions.py:5`, `test_suite_wiring_regression.py:369` |
| G4 | **Coverage headline** | "85%" is 85% of the 53% that is measured: **157/328 files, 68,616/146,496 lines (46.8%) are omitted**, including `cloud/*` (the deployed path, 15.9k lines) which is held only to a 58% floor. Two omit-rationale headings have no entry beneath them, which the "every entry has a reason" test cannot see. [Likely] | `pyproject.toml:342-404,367,402` |
| G5 | **PR-event CI** | `pull_request: branches: [main]`, but the repository's default branch is `claude/civ6-alphazero-implementation-vMnun`. PRs into the real default branch never get a `pull_request` run; they are covered only by `push` on `claude/**`. Anything `pull_request`-only (dependency-review, fork PRs, path filters) will silently never fire. [Certain] | `.github/workflows/ci.yml:3-7` |
| G6 | **Release workflow** | Triggers on `v*.*.*` tags. **Zero tags exist, locally or on the remote.** It has never run; `1.8.0` in pyproject and the CHANGELOG has no artifact. [Certain] | `release.yml:3-6` |
| G7 | **KB immutability check** (`validate_kb --git`) | Prints "shallow clone — skipping K7" and exits 0. `make validate` therefore passes with the rule unenforced on any shallow checkout. [Likely] | `scripts/validate_kb.py` |
| G8 | **Coordinator smoke test** (`§1.2` success criterion) | The dry-run branch caps games/actors but not `board_size`/`max_turns`/`num_simulations` — the same defect `DryRunProfile` fixed for `training_civ6` on 2026-08-28. On a 2-thread sandbox it did not finish in 300 s and its SIGTERM teardown touches the multiprocessing manager after it has exited. [Likely; a concurrent run in this session completed in ~2 min] | `distributed/coordinator.py:752-761,408` |
| G9 | **PLR0915 burndown** | Documented as three `main` functions; implemented as *whole-file* `per-file-ignores`, so it exempts six functions (248/168/82/68/64/62 statements). [Likely] | `pyproject.toml` per-file-ignores |
| G10 | **The package deprecates itself.** `utils/__init__.py:30` imports the deprecated `utils/logging` shim, so `python -W error::DeprecationWarning -c "import alpha_zero.core"` fails, as do 18 of 29 subpackages. The same chain imports `torch.utils.tensorboard` at module scope, making torch a 2.3 s hard transitive dependency of `rag/`, `bridge/`, `tournament/`. [Certain] | `alpha_zero/utils/__init__.py:30`, `utils/logging_config.py:340` |

### 3.3 Supply chain and release engineering

- **`uv.lock` is dead weight** [Certain]: last touched 2026-05-31, records `alpha-zero 1.6.0`,
  `mypy 1.13`, `ruff 0.8` against pinned `1.19.1`/`0.15.8`; excluded from the Docker context;
  consumed by nothing; `uv lock --check` fails. Every CI job and both deployed Dockerfiles install
  from unpinned lower bounds, so **no build is reproducible**.
- **Three requirements files drift from `pyproject.toml`** [Likely]: `torch>=1.13` vs `>=2.0`;
  five packages in `requirements.txt` that no extra declares (`python-dotenv` is imported by
  `utils/llm_client.py` yet is not a dependency of the wheel).
- **Actions on mutable tags** [Certain]: `checkout@v4`, `setup-python@v5`, `cache@v4`,
  `upload-artifact@v4` — 14 of 15 references. Only `action-gh-release` is SHA-pinned.
- **Dependabot generates work nobody does** [Certain]: 10 PRs open since 2026-07-25, 0 merged
  ever; two already obsolete. No `docker`/`uv` ecosystem, no grouping, no cooldown.
- **Absent entirely**: OpenSSF Scorecard, CodeQL/SARIF, `zizmor`, `pip-audit`/`osv-scanner`,
  SBOM, `attest-build-provenance`, cosign, Trivy/Grype, digest-pinned base images, PyPI/OIDC
  trusted publishing, `py.typed`, PEP 639 `license =`, classifiers, `[project.urls]`.
- **Python matrix** is 3.11 + 3.12 only; 3.13 and 3.14 (released 2025-10) untested; 3.15 ships
  2026-10-01. `Dockerfile.vertex` claims Python 3.12 for an NGC 24.09 base that ships 3.10 and
  never enforces `requires-python` because it uses `PYTHONPATH` rather than `pip install`
  [Guessing on the base-image Python; unverified without a `docker run`].
- **Dev install pulls CUDA wheels on CPU-only hosts** (CI runners included): torch `2.14.0+cu130`
  landed here with the full `nvidia-*` set — 7 installs per workflow run, ~800 MB each,
  cached only as pip wheels. [Certain for the local install]

### 3.4 Code quality and architecture

- **Size**: 328 files / 146,496 lines; 24 files >1,000 lines; largest function 248 statements.
- **Dead code**: 53 modules / 24,451 lines (16.7%) with no live importer, plus the ratchet defect
  in G1 [Certain]. `integration/game_connector.py` (1,358 lines) and the four `core/jax_*`
  modules (1,131 lines, 63+28 of pyright's 161 possibly-unbound errors) head the list.
- **Layering**: `core/ → distributed/`, `envs/ → core/`, `cloud/ → orchestrator/`, and 13
  mutual subpackage cycles held apart by 379 function-local imports (`PLC0415`). No
  `import-linter` contract.
- **Duplication**: 3 replay buffers, 6 MCTS node classes, 29 `save/load_checkpoint`
  implementations in 18 files, 10 `setup_logging` functions, 192 `*Config` classes with
  `TrainingConfig` ×4, `MCTSConfig` ×4, `NetworkConfig` ×3.
- **Configuration**: 404 direct `os.environ`/`os.getenv` reads; 334 distinct `CIV6AZ_*` names
  plus 7 other prefixes; 146 `int()/float()` env parses with no bounds.
- **Async**: 11 unguarded `asyncio.run()` calls in library code (`utils/llm_client.py:265,650,718`
  will raise `RuntimeError` from the FastAPI layer); 3 deprecated `get_event_loop()` in
  constructors (error on 3.14).
- **Error handling**: 222 blind `except Exception`; 97 of them log with `logger.error` not
  `.exception`, dropping tracebacks in worker logs. 27 legacy global `np.random.*` draws remain
  in search code — the exact class the 2026-08-30 seeding fix was about — with no `NPY002` rule.
- **Rule-set gap** (`--select ALL`, minus docstring/comment families): 6,064 violations across
  103 codes. Real signal worth gating at near-zero cost: `BLE001`, `TRY400`, `RUF100` (67 stale
  `noqa`), `PGH003` (5 bare `type: ignore`), `NPY002`, `ASYNC`, `PT` on tests (339
  unittest-style asserts in a pytest suite). Noise not worth enabling: `COM812`, `EM`, `TRY003`,
  `G004`.
- **Typing**: 0 errors under project config is real; `--strict` shows 164 in 56 files;
  `[tool.pyright]` exists but nothing runs it (350 errors). `warn_unused_ignores=false` hides 15
  stale ignores.

### 3.5 Test suite

- **Assertion quality** [Likely]: 166 tests with no assertion, 164 whose only assertion is
  `is not None`, 5 `assert True`.
- **Determinism** [Likely]: 1,110 unseeded random draws across 97 files (57 with no seed call at
  all); no `pytest-randomly`; one flaky test permanently skipped rather than quarantined.
- **Unit purity**: 82/305 unit files import torch, 48 build real `nn.Module`s; the "unit" run is
  6 minutes at 2 threads. xdist is installed and never used; `-x` in three CI steps hides every
  failure after the first.
- **Skips**: 126 sites / 69 files (CLAUDE.md says 124/68); 16 of the 50 CI-run skips are debt,
  including 4 tests that can never run on ubuntu CI (`test_bridge_config.py:12`) and 3 tests
  for a method that does not exist.
- **Taxonomy on paper only**: `hypothesis` in `dev`, used once; `benchmark` marker registered,
  used 0×; 0 tests reference `schema_version` despite the versioned `EvaluationResult`; no
  mutation testing.
- **Mock fragility**: 1,276 patches over 178 string targets; `sys.argv` is the #1 target (30×).

### 3.6 Documentation and governance

- **CLAUDE.md**: 1,434 lines / 12,527 words / 98 KB [Certain]. Anthropic's current guidance is
  under 200 lines. 55% of it is §5 narrative (83 decision rows, 22 known-issue rows, 131 dates).
  Contradictions with the tree: "Python 3.10+" (§2.1, §6.1) vs `>=3.11`; §7 file counts wrong for
  `training`, `evaluation`, `optimizations`, `opponents`, `human_interface` and missing
  `hexcrystal/`, `hub/`; §10 example builds a 356-wide head against a 376 action space; "Last
  updated 2026-07-25" above 20+ rows dated 08-25 to 09-02; §9.9 says W&B coexists, §5.2 says
  retired; hook count 5 vs 7 registered; skill count 9 vs 10.
- **CHANGELOG** [Likely]: two `## [Unreleased]` headings (lines 8 and 663), 683 lines unreleased
  since 07-12 including BREAKING changes, duplicate `[1.3.0]`, `[0.9.0]` dated after `[0.10.0]`.
  `check_doc_sync` D7 checks only the first release heading.
- **README** [Likely]: one documented command exits 2 (`selfplay_worker --dry_run` without its
  required flags); duplicate "Key Features" sections.
- **Onboarding**: no `.editorconfig`, `.python-version`, devcontainer, docs site build; hooks are
  mode 644; `templates/` (16 files, own pyproject) is excluded from every gate.
- **Commit hygiene**: 78% Conventional Commits, 89% with bodies, but 39% of subjects exceed 72
  chars, 33% are merge commits, CODEOWNERS is a single person (bus factor 1).
- **Stale docstrings**: `core/civ6_network.py:207,274` and `core/transformer_network.py:982,1030`
  direct readers to the deleted `core/network_heads`.
- **The same question was asked six months ago**: PR #69 ("Organize project structure to meet
  modern Python standards", Copilot, 2026-03-05) is still open.

---

## 4. Remediation plan

Effort: S ≤ ½ day, M ≤ 3 days, L > 3 days. Ordered by risk reduction per hour.

### Now (this week; all S)

1. **Rotate the leaked W&B key and record the date** in `docs/runbooks/secret-scanning.md`; then
   delete the branch and purge. Rotation is the remediation; purge is cleanup.
2. **Close the public inference endpoint**: remove `allUsers` from
   `infra/terraform/modules/ai-inference/main.tf` (or front with IAP/API Gateway) until the
   promised JWT dependency exists; add request-size limits before `np.array` in
   `serving/inference_server.py`; stop returning `str(e)`.
3. **Fix the CI PR trigger** in `.github/workflows/ci.yml:3-7` to the real default branch.
4. **Add `build`, `dist`, `.tox`, `.eggs` to `SKIP_PARTS`** in `scripts/audit_import_graph.py`
   and a negative-control test that plants `build/lib/alpha_zero/__init__.py`.
5. **Make the dev install succeed**: move `chess` out of `dev` (keep the `[chess]` extra) and
   convert the 9 bare `import chess` to `pytest.importorskip`; widen the wiring gate's
   optional-import check to every test module.
6. **Add `pip-audit` (or OSV) as a blocking CI job** on `requirements-container.txt` and `.[dev]`.
7. **Pin every action to a full SHA** (14 sites) and triage the 10 Dependabot PRs; add `groups:`,
   `cooldown:`, and the `docker` ecosystem.
8. **Fix `web/config.py` defaults**: `cors_origins=[]`, `host="127.0.0.1"`; drop
   `allow_credentials=True` or require an origin check on state-changing routes.
9. **Harden the push guard**: unwrap `bash|sh -c`, `eval`, `xargs`; deny `gh pr merge`; replace
   the blanket `Bash(python scripts/:*)` allow with an explicit list.
10. **Un-deprecate the package import**: point `utils/__init__.py` at `logging_config`; make the
    tensorboard import lazy; add a `-W error::DeprecationWarning` import test over all 29
    subpackages.

### Next (this month; M)

11. **Choose a lock strategy and enforce it.** Either `uv lock` + `uv sync --locked` in CI and
    Docker with a `uv lock --check` job, or delete `uv.lock`. Generate hashed
    `requirements*.txt` from the lock (`uv export --generate-hashes`) and install with
    `--require-hashes` in `Dockerfile.vertex`/`Dockerfile.inference`. Lift `mlflow<3.0` or
    document a compensating control.
12. **Route all 22 `torch.load` and 5 `pickle.loads` sites through `utils/safe_load`**; move
    optimizer/config state to `weights_only`-compatible dicts; replace pickled trajectories with
    `safetensors`/`npz`. Lower the unsafe-load ratchet to 1.
13. **Cut `v1.8.0` and `v1.9.0` tags** so `release.yml` runs once; merge the two `[Unreleased]`
    headings; extend `check_doc_sync` D7 to assert one `[Unreleased]`, descending versions, and a
    tag matching `pyproject` at release. Add `python -m build` + `attest-build-provenance` and
    attach wheel/sdist.
14. **Widen the bandit gate** to `alpha_zero/ scripts/ infra/` at medium/medium against a
    committed baseline so only new findings fail. Add `zizmor` and an OpenSSF Scorecard workflow.
15. **Enable the cheap, high-signal ruff rules** (`BLE001`, `TRY400`, `RUF100`, `PGH003`,
    `NPY002`, `ASYNC`, `PT`) with per-file burndown lists; replace the whole-file `PLR0915`
    ignores with `# noqa` on the two `main`s.
16. **Test determinism**: add `pytest-randomly`, an autouse seeding fixture in `tests/conftest.py`,
    and an AST ratchet on zero-assert tests (baseline 166); remove `-x` from CI; run the
    torch-touching unit subset with `-n auto`.
17. **Split CLAUDE.md** to <200 lines of instruction: §5.2/§5.3/§5.4 history to
    `docs/architecture/adr/` (the MADR system already exists with 3 entries), §7 to a pointer at
    `MODULE_REFERENCE.md`, §9–13 to `docs/`; path-scoped `.claude/rules/*.md` for the
    per-package gotchas. Fix the 3.10/3.11, 356/376, W&B, hook-count and skill-count lines.
18. **Package metadata**: `license = "MIT"`, classifiers, `[project.urls]`, `py.typed`
    (+ package-data), `python-dotenv` declared or the import removed; PEP 735
    `[dependency-groups]` for dev/test. Add 3.13 and 3.14 to the matrix; drop the CUDA wheel set
    from CI via a CPU torch index.

### Later (quarter; L)

19. **Delete the 53 dead modules (24,451 lines)** per `kb/specs/20260902-tech-debt-hardening.md`,
    lowering the ratchet baseline in the same PR, then bring `cloud/*` into the global coverage
    gate.
20. **One config surface**: single `TrainingConfig`/`MCTSConfig`/`NetworkConfig`; one env loader
    with bounded `env_int/env_float` replacing 404 direct reads; `import-linter` contracts to
    break `core→distributed`, `envs→core` and the 13 cycles.
21. **One checkpoint API and one logging setup**; structured JSON logs in containers; lazy `%s`
    args on hot-path log lines.
22. **Test taxonomy that exists**: hypothesis properties for action-mask/executor agreement (the
    known WS-D2 bug is a property violation), a `schema_version` round-trip contract, nightly
    non-blocking `mutmut` on `core/mcts`, `envs/microciv.py`, `evaluation/scoring.py`.
23. **Web/serving auth** proper: the Firebase-JWT dependency the Terraform comment promises, or
    remove the surface.

---

## 5. What was verified in this session (reproduction log)

| Check | Result |
|---|---|
| `ruff check .` | clean, 0.48 s |
| `ruff format --check .` | 882 files already formatted |
| `python -m mypy alpha_zero/` | Success: no issues in 328 source files (58.7 s) |
| `pytest tests/regression/` | 126 passed in 77.9 s |
| `python -m alpha_zero.training_civ6 --dry_run` | exit 0, 11.3 s |
| `python -m alpha_zero.orchestrator --preset fast --dry_run` | exit 0 |
| `python -m alpha_zero.distributed.coordinator --dry_run` | completed once in ~2 min; timed out at 300 s in a concurrent agent run (see G8) |
| `pip install -e '.[dev]'` on Python 3.11.15 | **fails** on `chess` sdist (G2) |
| `audit_import_graph.py` clean vs with `build/lib/alpha_zero` | 53/24,451 → 30/15,120 (G1) |
| `python -W error::DeprecationWarning -c "import alpha_zero.core"` | **fails** (G10) |
| PreToolUse guard, `git push --force origin main` | deny |
| PreToolUse guard, `bash -c "git push --force origin main"` | **allowed** (S7) |
| `git tag`, `git ls-remote --tags origin` | 0 / 0 (G6) |
| `git log --author=dependabot` | 0 commits |
| Latest CI on default branch (run 441, 2026-09-02) | success |

Agent-run measurements not re-run by hand are tagged [Likely] above; their scratch artifacts
(bandit JSON, ruff `--select ALL` statistics, mypy `--strict`, pyright JSON, pip-audit output,
AST zero-assert list, unseeded-file list, debt JSON) were produced in the session scratchpad and
are reproducible from the commands quoted in each section.

## 6. Sources for the 2026 bar

The rubric behind the scores was built from primary sources dated 2025–2026: uv `sync --locked`
and `export` docs; PEP 639/735/751; Python 3.14 release and 3.10 EOL (2026-10); ruff 0.16 release
notes (defaults changed 59→413 rules); mypy 1.20, Pyrefly 1.0 and ty beta status; pytest 9
`strict = true`; coverage `fail_under` in config; OpenSSF Scorecard checks; GitHub SHA-pinning
policy (2025-08) and Dependabot cooldown default (2026-07); `actions/attest-build-provenance`;
Sigstore keyless signing; PyTorch 2.6 `weights_only=True` default and CVE-2025-32434; Docker
digest pinning; MLflow 3 `LoggedModel`; PyTorch randomness notes; Anthropic's CLAUDE.md guidance
(<200 lines). Items the research could not confirm with a 2025–2026 primary source are marked
in the rubric and were not used to lower any score.
