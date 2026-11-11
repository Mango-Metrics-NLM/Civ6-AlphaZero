# Makefile for Civ6 AlphaZero Vertex AI Deployment
#
# Usage:
#   make help              - Show available targets
#   make verify            - Reproduce every CI gate locally
#   make verify-fast       - The cheap gates only (lint, types, governance, regression)
#   make build-all         - Build the deployed image (dockerfiles/Dockerfile.vertex)
#   make push              - Push it to Artifact Registry
#   make local-smoke       - Dry-run the deployed self-play worker

# ============================================================================
# Configuration
# ============================================================================

# Torch intra-op threads for every test target below.
#
# Without this cap, two tests wedge in `torch.conv2d` on the OpenMP spin-wait and
# hit the 120 s pytest timeout: CLAUDE.md's build table has documented that for
# months, but nothing actually set it, so the one command the handoff skills
# mandate (`make verify`) was documented as hanging. `?=` so a caller can raise
# it on a bigger machine (`OMP_NUM_THREADS=8 make test`) and `export` so every
# recipe and every child process inherits it.
OMP_NUM_THREADS ?= 2
export OMP_NUM_THREADS

# GCP Configuration (override with environment variables)
PROJECT_ID ?= $(shell gcloud config get-value project 2>/dev/null)
REGION ?= us-central1
BUCKET ?= $(PROJECT_ID)-civ6-alphazero
ARTIFACT_REGISTRY ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/civ6-alphazero

# Image tags
VERSION ?= latest
BASE_IMAGE ?= $(ARTIFACT_REGISTRY)/civ6-base:$(VERSION)

# Local image names (for development)
LOCAL_BASE_IMAGE ?= civ6-alphazero-base:$(VERSION)

# Python
PYTHON ?= python3
PIP ?= pip3

# Hugging Face namespace for `publish-spaces`. The maintainer's account unless
# overridden; the Space names below are the ones .github/workflows/deploy_spaces.yml
# publishes, so the two paths cannot drift onto differently-named Spaces.
HF_ORG ?= ianshank

# Secret scanning. Pinned so a local run and CI detect the same things; override
# with `make security-secrets GITLEAKS_VERSION=x.y.z`. GITLEAKS_BIN lets a machine
# with gitleaks already on PATH skip the download entirely.
GITLEAKS_VERSION ?= 8.30.0
# Default to the managed, version-pinned binary. Defaulting to whatever `gitleaks`
# happens to be on PATH would silently run a different ruleset from CI and defeat
# the pin two lines above. Override explicitly to opt out:
#   make security-secrets GITLEAKS_BIN=/usr/local/bin/gitleaks
#
# The version is IN THE PATH, not just in the download URL. The installer only
# fetches when the target is missing, so a bare `.tools/gitleaks` left over from
# an earlier pin would be reused for ever -- the same defeat-the-pin bug as the
# PATH default, one step further along. Stamping the name makes a pin change a
# cache miss, which is the only way that check can be got wrong.
GITLEAKS_BIN ?= $(CURDIR)/.tools/gitleaks-$(GITLEAKS_VERSION)
GITLEAKS_CONFIG ?= .gitleaks.toml

# SHA-256 of each release archive, copied from
# https://github.com/gitleaks/gitleaks/releases/download/v$(GITLEAKS_VERSION)/gitleaks_$(GITLEAKS_VERSION)_checksums.txt
# A version-pinned URL alone pins nothing: the asset behind it can be replaced.
# When GITLEAKS_VERSION moves, every line below must move with it (the installer
# refuses to run a platform that has no pinned sum, so a half-updated table fails
# loudly rather than installing an unverified binary).
GITLEAKS_SHA256_linux_x64     ?= 79a3ab579b53f71efd634f3aaf7e04a0fa0cf206b7ed434638d1547a2470a66e
GITLEAKS_SHA256_linux_arm64   ?= b4cbbb6ddf7d1b2a603088cd03a4e3f7ce48ee7fd449b51f7de6ee2906f5fa2f
GITLEAKS_SHA256_darwin_x64    ?= ca221d012d247080c2f6f61f4b7a83bffa2453806b0c195c795bbe9a8c775ed5
GITLEAKS_SHA256_darwin_arm64  ?= b251ab2bcd4cd8ba9e56ff37698c033ebf38582b477d21ebd86586d927cf87e7
GITLEAKS_SHA256_windows_x64   ?= 54fe94f644b832dd08e8c3a5915efb3bfa862386d59fb27ca0792cb687a83573

# ============================================================================
# Help
# ============================================================================

.PHONY: help
help:
	@echo "Civ6 AlphaZero - Vertex AI Deployment"
	@echo ""
	@echo "Development:"
	@echo "  make install           Install the project with the dev extra (pip install -e '.[dev]')"
	@echo "  make install-dev       Alias of install"
	@echo "  make lint              Run linting (ruff)"
	@echo "  make typecheck         Run type checking (mypy)"
	@echo "  make test-unit         Run unit tests"
	@echo "  make test-integration  Run integration tests"
	@echo "  make test-regression   Run the repository-hygiene regression suite"
	@echo "  make verify            Reproduce EVERY CI gate locally (run before a PR)"
	@echo "  make verify-fast       Lint, types, governance + regression only (minutes, not an hour)"
	@echo "  make test-all          Run all tests with coverage (global gate: 85%)"
	@echo "  make test-cloud-coverage  Measure alpha_zero/cloud against coverage-floors.json"
	@echo "  make validate          Run governance validators (openspec, kb, skills, agents, doc-sync)"
	@echo "  make hooks-install     Install pre-commit + git hooks (self-installing)"
	@echo "  make eval-llm          Run LLM eval-harness gate (StrategyAdvisor analyze + report, offline)"
	@echo ""
	@echo "Docker:"
	@echo "  make build-all         Build the deployed image (dockerfiles/Dockerfile.vertex)"
	@echo "  make push              Push it to Artifact Registry"
	@echo ""
	@echo "Local Testing:"
	@echo "  make local-smoke       Dry-run the deployed self-play worker"
	@echo ""
	@echo "Vertex AI:"
	@echo "  python -m alpha_zero.cloud.gcp.vertex_orchestrator --help   (the deployed orchestrator)"
	@echo "  python -m infra.vertex_launcher --help                       (LEGACY launcher, dry-run only)"
	@echo ""
	@echo "Hugging Face:"
	@echo "  make publish-spaces    Upload both Gradio Spaces to $(HF_ORG)/ (override with HF_ORG=...)"
	@echo ""
	@echo "Configuration:"
	@echo "  PROJECT_ID=$(PROJECT_ID)"
	@echo "  REGION=$(REGION)"
	@echo "  BUCKET=$(BUCKET)"

# ============================================================================
# Development
# ============================================================================

# One install path, the same one README, CONTRIBUTING, CLAUDE.md and every CI
# job use. `install` used to run `pip install -r requirements.txt` (a file that
# drifts from pyproject.toml) and `install-dev` layered unpinned tool versions on
# top, so a developer's ruff/mypy could disagree with the pinned ones in [dev].
.PHONY: install
install:
	$(PIP) install -e '.[dev]'

.PHONY: install-dev
install-dev: install

.PHONY: install-cloud
install-cloud: install
	$(PIP) install google-cloud-storage google-cloud-aiplatform gcsfs kfp

.PHONY: lint
lint:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format .

.PHONY: lint-check
lint-check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

# Matches the CI `lint` job exactly. Use `typecheck-strict` for the harsher
# local pass -- CI does NOT run --strict, so a --strict failure is not a gate.
# Invoked as `python -m mypy` because a mypy resolved from an isolated tool venv
# cannot see this project's installed type stubs and reports phantom errors.
.PHONY: typecheck
typecheck:
	$(PYTHON) -m mypy alpha_zero/ --show-error-codes

.PHONY: typecheck-strict
typecheck-strict:
	$(PYTHON) -m mypy alpha_zero/ --strict

.PHONY: test-unit
test-unit:
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

.PHONY: test-integration
test-integration:
	$(PYTHON) -m pytest tests/integration/ -v --tb=short

# Repository-hygiene invariants and known-bug reproductions (~10 s). The same
# invocation CI's `test` job and the pre-push hook use.
.PHONY: test-regression
test-regression:
	$(PYTHON) -m pytest tests/regression/ -v --tb=short

# Mirrors the CI `coverage` job: same suites, same marker filter, same gate.
# It previously ran the whole tests/ tree at --cov-fail-under=80 while CI used
# 85 over a narrower set, so passing locally proved nothing about CI.
.PHONY: test-all
test-all:
	$(PYTHON) -m pytest tests/unit/ tests/sanity/ tests/integration/ tests/cloud/unit/ tests/regression/ \
		-m "not slow and not requires_env" \
		--cov=alpha_zero --cov-report=term-missing --cov-report=html \
		--cov-fail-under=85

.PHONY: test-storage
test-storage:
	$(PYTHON) -m pytest tests/unit/test_storage_backend.py tests/unit/test_replay_buffer.py -v

.PHONY: test-sanity
test-sanity:
	$(PYTHON) -m pytest tests/sanity/ -v --tb=short

.PHONY: test-cloud
test-cloud:
	$(PYTHON) -m pytest tests/cloud/unit/ -v --tb=short

# Mirrors CI's `cloud-tests` job, which since 2026-09-02 also MEASURES the
# deployed path. alpha_zero/cloud/* is omitted from the repo-wide gate, so until
# this floor existed no job measured it at all. A separate coverage config is
# required, not cosmetic: the repo-wide one omits the very package under test,
# and `coverage json` without --rcfile reads pyproject.toml and reports nothing.
.PHONY: test-cloud-coverage
test-cloud-coverage:
	$(PYTHON) -m pytest tests/cloud/unit/ -v --tb=short -x \
		--cov=alpha_zero.cloud --cov-config=.coveragerc-cloud --cov-report=
	$(PYTHON) -m pytest tests/cloud/smoke/ -v --tb=short \
		--cov=alpha_zero.cloud --cov-config=.coveragerc-cloud --cov-append --cov-report=
	$(PYTHON) -m coverage json --rcfile=.coveragerc-cloud -o coverage-cloud.json
	$(PYTHON) scripts/check_coverage_floors.py --report coverage-cloud.json --package alpha_zero/cloud

# Mirrors CI's `security` job, which runs BOTH the pytest suite and a bandit
# scan. The bandit half was missing entirely, so `make verify` could pass while
# CI's security job failed.
.PHONY: test-security
test-security:
	$(PYTHON) -m pytest tests/security/ -v --tb=short
	$(PYTHON) -m bandit -r alpha_zero/ -s B101 --severity-level high --confidence-level high

# CI's advisory medium/medium bandit pass, gated on the accepted-findings
# baseline. Regenerate the baseline (and review the diff) with:
#   bandit -r alpha_zero scripts infra -s B101 -ll -ii -f json -o .security/bandit-baseline.json
.PHONY: test-security-medium
test-security-medium:
	$(PYTHON) -m bandit -r alpha_zero scripts infra -s B101 -ll -ii -b .security/bandit-baseline.json

# Print the pinned gitleaks version so CI reads it from here rather than
# restating it. One definition, two consumers.
.PHONY: print-gitleaks-version
print-gitleaks-version:
	@echo $(GITLEAKS_VERSION)

# Fetch the pinned gitleaks binary only when it is not already available.
# Resolves the release artifact for THIS platform: CLAUDE.md 2.1 requires
# cross-platform support, and always fetching linux_x64 installed an unusable
# binary on macOS and ARM -- silently, since `make verify` would then fail on a
# confusing exec error rather than on the missing tool.
# Artifact names verified against the v8.30.0 release.
#
# The archive is downloaded to disk and its SHA-256 compared with the pinned sum
# BEFORE anything is extracted. Piping `curl | tar` executed whatever the URL
# served; a mismatch now deletes the download and fails the target.
.PHONY: gitleaks-install
gitleaks-install:
	@if [ ! -x "$(GITLEAKS_BIN)" ]; then \
		os=$$(uname -s); arch=$$(uname -m); \
		case "$$os" in \
			Linux)   plat=linux ;; \
			Darwin)  plat=darwin ;; \
			MINGW*|MSYS*|CYGWIN*) plat=windows ;; \
			*) echo "gitleaks: unsupported OS '$$os'. Install it yourself and set GITLEAKS_BIN." >&2; exit 1 ;; \
		esac; \
		case "$$arch" in \
			x86_64|amd64)  cpu=x64 ;; \
			arm64|aarch64) cpu=arm64 ;; \
			*) echo "gitleaks: unsupported arch '$$arch'. Install it yourself and set GITLEAKS_BIN." >&2; exit 1 ;; \
		esac; \
		case "$${plat}_$${cpu}" in \
			linux_x64)     want="$(GITLEAKS_SHA256_linux_x64)" ;; \
			linux_arm64)   want="$(GITLEAKS_SHA256_linux_arm64)" ;; \
			darwin_x64)    want="$(GITLEAKS_SHA256_darwin_x64)" ;; \
			darwin_arm64)  want="$(GITLEAKS_SHA256_darwin_arm64)" ;; \
			windows_x64)   want="$(GITLEAKS_SHA256_windows_x64)" ;; \
			*) echo "gitleaks: no pinned SHA-256 for $${plat}_$${cpu}; add GITLEAKS_SHA256_$${plat}_$${cpu} from the release checksums file." >&2; exit 1 ;; \
		esac; \
		if [ "$$plat" = "windows" ]; then ext=zip; bin=gitleaks.exe; else ext=tar.gz; bin=gitleaks; fi; \
		url="https://github.com/gitleaks/gitleaks/releases/download/v$(GITLEAKS_VERSION)/gitleaks_$(GITLEAKS_VERSION)_$${plat}_$${cpu}.$${ext}"; \
		archive="$(CURDIR)/.tools/gitleaks-$(GITLEAKS_VERSION)-$${plat}_$${cpu}.$${ext}"; \
		echo "Installing gitleaks $(GITLEAKS_VERSION) ($${plat}_$${cpu}) -> $(CURDIR)/.tools"; \
		mkdir -p $(CURDIR)/.tools; \
		curl -sSfL "$$url" -o "$$archive" || { rm -f "$$archive"; exit 1; }; \
		got=$$($(PYTHON) -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$$archive"); \
		if [ "$$got" != "$$want" ]; then \
			echo "gitleaks: SHA-256 mismatch for $$url" >&2; \
			echo "  expected $$want" >&2; \
			echo "  got      $$got" >&2; \
			rm -f "$$archive"; exit 1; \
		fi; \
		if [ "$$ext" = "zip" ]; then \
			unzip -oq "$$archive" "$$bin" -d $(CURDIR)/.tools; \
		else \
			tar -xzf "$$archive" -C $(CURDIR)/.tools "$$bin"; \
		fi; \
		rm -f "$$archive"; \
		mv -f "$(CURDIR)/.tools/$$bin" $(GITLEAKS_BIN); \
	fi

# Secret scan: the tracked tree AND the built branch's own history. History
# matters because a secret removed in a later commit is still published in the
# one that added it.
#
# Scoped to `HEAD` deliberately. Without --log-opts, gitleaks walks *every* ref
# the clone has, so on a CI checkout with fetch-depth: 0 the gate reports findings
# from unmerged branches that the PR does not contain and cannot fix. A PR gate
# should assert what this branch contributes; auditing every stale ref is a
# separate activity -- see `security-secrets-audit`.
#
# Still a real gate: it covers this branch's entire HEAD lineage, 0 findings.
#
# The tree half scans the *committable* file set -- tracked plus untracked-and-
# not-ignored -- rather than the working directory. `gitleaks dir .` reads
# gitignored files too, and this project requires a gitignored `.env` holding
# real API keys (CLAUDE.md 2.1, 3.1), so a correctly set-up developer would have
# had `make verify` fail on a file that cannot be committed. Verified: gitleaks
# does flag a secret in a gitignored file, and has no flag to stop it.
#
# `git ls-files -co --exclude-standard` is the right set precisely because an
# accidentally *tracked* `.env` is in it, and still fails. Allowlisting `.env`
# would have hidden exactly the case worth catching.
#
# Findings are reported against the staging path; the sed rewrites them back to
# repo-relative so a failure names a file you can actually open.
.PHONY: security-secrets
security-secrets: gitleaks-install
	@stage=$$(mktemp -d); out=$$(mktemp); \
	git ls-files -z -co --exclude-standard | tar --null -T - -cf - | tar -x -C "$$stage"; \
	$(GITLEAKS_BIN) dir "$$stage" --config $(GITLEAKS_CONFIG) --redact --no-banner \
		> "$$out" 2>&1; \
	status=$$?; \
	sed "s#$$stage/##g" "$$out"; \
	rm -rf "$$stage" "$$out"; \
	exit $$status
	$(GITLEAKS_BIN) git . --config $(GITLEAKS_CONFIG) --redact --no-banner --log-opts="HEAD"

# Repo-wide audit across every ref, including unmerged and stale branches. NOT a
# PR gate -- it reports history that no single PR can remediate, because purging
# a secret from a published branch requires rotation plus coordinated history
# surgery. Run it deliberately; see docs/runbooks/secret-scanning.md.
.PHONY: security-secrets-audit
security-secrets-audit: gitleaks-install
	@echo "Repo-wide audit across all refs. Findings here are incidents, not PR blockers."
	$(GITLEAKS_BIN) git . --config $(GITLEAKS_CONFIG) --redact --no-banner || true

.PHONY: test-governance
test-governance:
	$(PYTHON) -m pytest -c tests/governance/pytest.ini --confcutdir=tests/governance tests/governance

# The suites CI runs, in CI's order and with CI's marker filters.
.PHONY: test
test:
	$(PYTHON) -m pytest tests/unit/ -v --tb=short -x -m "not slow"
	$(PYTHON) -m pytest tests/sanity/ -v --tb=short
	$(PYTHON) -m pytest tests/regression/ -v --tb=short
	$(PYTHON) -m pytest tests/integration/ -v --tb=short -m "not slow and not requires_env" -x
	$(PYTHON) -m pytest tests/smoke/ tests/inference/ -v --tb=short
	$(PYTHON) -m pytest tests/scripts/ -v --tb=short -m "not requires_env"
	$(PYTHON) -m pytest tests/e2e/ -v --tb=short -m "not slow and not requires_env"
	$(PYTHON) -m pytest tests/cloud/unit/ -v --tb=short -x
	$(PYTHON) -m pytest tests/cloud/smoke/ -v --tb=short

# `validate_kb.py --git` makes the handoff-immutability check (K7) mandatory: on
# a shallow clone it FAILS rather than printing a note, because `make validate`
# passing with the rule unenforced is what the 2026-09-04 audit found (G7).
# `git fetch --unshallow` once, or run `scripts/validate_kb.py` without the flag
# for the K1-K6 subset.
.PHONY: validate
validate:
	$(PYTHON) scripts/validate_openspec.py
	$(PYTHON) scripts/validate_kb.py --git
	$(PYTHON) scripts/validate_skills.py
	$(PYTHON) scripts/validate_agents.py
	$(PYTHON) scripts/check_doc_sync.py

# Self-installing: pre-commit is not a declared dependency, so the bare
# `pre-commit install` this used to run failed on a clean checkout and left
# .pre-commit-config.yaml (including the doc-sync and KB gates) entirely inert.
.PHONY: hooks-install
hooks-install:
	$(PYTHON) -m pip install --quiet pre-commit
	$(PYTHON) -m pre_commit install --install-hooks
	$(PYTHON) -m pre_commit install --hook-type pre-push

# One command that reproduces every CI gate, in CI's order. The handoff skills
# and any pre-PR check should call this rather than restating the sequence --
# six hand-assembled invocations is how the local and CI gates drifted apart.
# Every gate CI enforces, in CI's order. Keep this in step with
# .github/workflows/ci.yml -- when they drift, passing locally stops meaning
# anything, which is the failure this target exists to prevent. It previously
# omitted the coverage job, the bandit scan and the eval gate while claiming to
# reproduce everything.
.PHONY: verify
verify: lint-check typecheck test test-security security-secrets test-governance validate test-all test-cloud-coverage eval-llm
	@echo "verify: reproduced CI's lint, type, test, security, governance, coverage (global + cloud floor) and eval gates"

# The gates that finish in minutes: everything in `verify` except the test
# suites that need torch time (unit/integration/e2e/cloud/coverage/eval) and the
# gitleaks download. Not a substitute for `verify` before a PR -- a fast green
# here says the change is lint-, type- and governance-clean and broke no
# repository-hygiene invariant, nothing about behaviour.
.PHONY: verify-fast
verify-fast: lint-check typecheck test-governance test-regression validate
	@echo "verify-fast: lint, types, governance tests, regression suite and validators are green (run 'make verify' before a PR)"

.PHONY: eval-llm
eval-llm:
	$(PYTHON) scripts/run_llm_eval.py

.PHONY: test-hub
test-hub:
	$(PYTHON) -m pytest tests/unit/test_hub_*.py tests/unit/test_transformer_hub_mixin.py tests/unit/test_muzero_hub_mixin.py tests/integration/test_hub_publish_pipeline.py -v

.PHONY: upload-hub
upload-hub:
	$(PYTHON) scripts/upload_to_hub.py --model-type $(MODEL_TYPE) $(UPLOAD_ARGS)

.PHONY: export-datasets
export-datasets:
	$(PYTHON) scripts/export_dataset.py --type $(DATASET_TYPE) --source-path $(SOURCE_PATH) $(EXPORT_ARGS)

# Same Space ids as .github/workflows/deploy_spaces.yml (`<HF_ORG>/microciv-explorer`
# and `<HF_ORG>/elo-dashboard`). The two used to name different Spaces in
# different namespaces, so a manual upload and a workflow run published to
# different places.
.PHONY: publish-spaces
publish-spaces:
	huggingface-cli upload $(HF_ORG)/microciv-explorer ./spaces/microciv-explorer --repo-type space
	huggingface-cli upload $(HF_ORG)/elo-dashboard ./spaces/elo-dashboard --repo-type space

# ============================================================================
# Docker Build
# ============================================================================
# The deployed image is dockerfiles/Dockerfile.vertex, built by cloudbuild.yaml
# (`gcloud builds submit --config cloudbuild.yaml .`). The per-role images
# (base/selfplay/trainer/evaluator) and the LEGACY
# alpha_zero.cloud.{selfplay_worker,trainer,evaluator} modules they entrypointed
# were removed on 2026-09-02 as dead code; nothing built them.

.PHONY: build-all
build-all:
	docker build -f dockerfiles/Dockerfile.vertex -t $(LOCAL_BASE_IMAGE) .
	@echo "Built $(LOCAL_BASE_IMAGE) from dockerfiles/Dockerfile.vertex"

# ============================================================================
# Docker Push
# ============================================================================

.PHONY: setup-artifact-registry
setup-artifact-registry:
	@echo "Creating Artifact Registry repository..."
	gcloud artifacts repositories create civ6-alphazero \
		--repository-format=docker \
		--location=$(REGION) \
		--description="Civ6 AlphaZero container images" \
		|| true
	gcloud auth configure-docker $(REGION)-docker.pkg.dev

.PHONY: push
push: build-all
	docker tag $(LOCAL_BASE_IMAGE) $(BASE_IMAGE)
	docker push $(BASE_IMAGE)
	@echo "Pushed $(BASE_IMAGE)"

# ============================================================================
# Local Testing
# ============================================================================
# `make local-test` / `make local-down` (docker-compose over the per-role images)
# were removed with those images on 2026-09-02. `local-smoke` survives, but it no
# longer goes through docker at all: it is the deployed worker's own dry run.

.PHONY: local-smoke
local-smoke:
	@echo "Running local smoke test (deployed self-play worker, dry run)..."
	CIV6_STORAGE_BACKEND=local CIV6_STORAGE_BASE_PATH=./test_storage \
	$(PYTHON) -m alpha_zero.cloud.gcp.selfplay_worker \
		--worker_id 0 --model_version 0 --iteration 0 --games 1 --dry_run
	@echo "Smoke test passed"

.PHONY: clean-test-storage
clean-test-storage:
	rm -rf ./test_storage

# ============================================================================
# GCS Setup
# ============================================================================

# Bucket creation only. This target also ran `gsutil lifecycle set
# infra/gcs_lifecycle.json`, a file that has never existed in the repository, so
# it failed on every invocation after creating the bucket. Lifecycle rules are
# set in the console or with a policy file you supply:
#   gsutil lifecycle set <policy.json> gs://$(BUCKET)
.PHONY: setup-bucket
setup-bucket:
	@echo "Creating GCS bucket gs://$(BUCKET)..."
	gsutil mb -p $(PROJECT_ID) -l $(REGION) gs://$(BUCKET) || true

.PHONY: setup-secrets
setup-secrets:
	@echo "Setting up secrets in Secret Manager..."
	@if [ -z "$(WANDB_API_KEY)" ]; then \
		echo "WANDB_API_KEY not set, skipping..."; \
	else \
		echo "$(WANDB_API_KEY)" | gcloud secrets create wandb-api-key \
			--data-file=- \
			--replication-policy=automatic \
			|| gcloud secrets versions add wandb-api-key --data-file=-; \
	fi

# ============================================================================
# Vertex AI
# ============================================================================
# The six targets that lived here (`deploy-pipeline`, `estimate-cost`,
# `run-iteration`, `run-selfplay`, `run-trainer`, `run-evaluator`) were deleted
# on 2026-11-10. Two had been rewritten to `exit 1` with an apology because the
# code they called never existed; the four `run-*` wrappers drove
# `infra/vertex_launcher.py`, whose real-submit path imports a function that
# does not exist, so only their `--dry-run` form ever worked. A target that
# cannot run is documentation pretending to be automation.
#
# The deployed path:  python -m alpha_zero.cloud.gcp.vertex_orchestrator --help
# The legacy launcher (dry-run only):  python -m infra.vertex_launcher --help

# ============================================================================
# Utilities
# ============================================================================

.PHONY: check-config
check-config:
	@echo "Checking configuration..."
	@echo "PROJECT_ID: $(PROJECT_ID)"
	@echo "REGION: $(REGION)"
	@echo "BUCKET: $(BUCKET)"
	@echo "ARTIFACT_REGISTRY: $(ARTIFACT_REGISTRY)"
	@test -n "$(PROJECT_ID)" || (echo "ERROR: PROJECT_ID not set" && exit 1)
	@gcloud projects describe $(PROJECT_ID) > /dev/null 2>&1 || \
		(echo "ERROR: Cannot access project $(PROJECT_ID)" && exit 1)
	@echo "Configuration OK"

.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# `docker-compose down -v` was removed on 2026-09-02 with docker-compose.yml and
# the per-role images it composed; there is no local stack left to tear down.
.PHONY: clean-all
clean-all: clean clean-test-storage
	rm -f coverage-cloud.json
	docker rmi $(LOCAL_BASE_IMAGE) 2>/dev/null || true

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help
