# Makefile for Civ6 AlphaZero Vertex AI Deployment
#
# Usage:
#   make help              - Show available targets
#   make build-all         - Build all container images
#   make push              - Push images to Artifact Registry
#   make local-test        - Run local integration test
#   make deploy-pipeline   - Deploy Vertex AI pipeline

# ============================================================================
# Configuration
# ============================================================================

# GCP Configuration (override with environment variables)
PROJECT_ID ?= $(shell gcloud config get-value project 2>/dev/null)
REGION ?= us-central1
BUCKET ?= $(PROJECT_ID)-civ6-alphazero
ARTIFACT_REGISTRY ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/civ6-alphazero

# Image tags
VERSION ?= latest
BASE_IMAGE ?= $(ARTIFACT_REGISTRY)/civ6-base:$(VERSION)
SELFPLAY_IMAGE ?= $(ARTIFACT_REGISTRY)/civ6-selfplay:$(VERSION)
TRAINER_IMAGE ?= $(ARTIFACT_REGISTRY)/civ6-trainer:$(VERSION)
EVALUATOR_IMAGE ?= $(ARTIFACT_REGISTRY)/civ6-evaluator:$(VERSION)

# Local image names (for development)
LOCAL_BASE_IMAGE ?= civ6-alphazero-base:$(VERSION)
LOCAL_SELFPLAY_IMAGE ?= civ6-selfplay:$(VERSION)
LOCAL_TRAINER_IMAGE ?= civ6-trainer:$(VERSION)
LOCAL_EVALUATOR_IMAGE ?= civ6-evaluator:$(VERSION)

# Python
PYTHON ?= python3
PIP ?= pip3

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

# ============================================================================
# Help
# ============================================================================

.PHONY: help
help:
	@echo "Civ6 AlphaZero - Vertex AI Deployment"
	@echo ""
	@echo "Development:"
	@echo "  make install           Install Python dependencies"
	@echo "  make install-dev       Install development dependencies"
	@echo "  make lint              Run linting (ruff)"
	@echo "  make typecheck         Run type checking (mypy)"
	@echo "  make test-unit         Run unit tests"
	@echo "  make test-integration  Run integration tests"
	@echo "  make verify            Reproduce EVERY CI gate locally (run before a PR)"
	@echo "  make test-all          Run all tests with coverage"
	@echo "  make validate          Run governance validators (openspec, kb, skills, agents, doc-sync)"
	@echo "  make hooks-install     Install pre-commit + git hooks (self-installing)"
	@echo "  make eval-llm          Run LLM eval-harness gate (StrategyAdvisor analyze + report, offline)"
	@echo ""
	@echo "Docker:"
	@echo "  make build-base        Build base image"
	@echo "  make build-selfplay    Build self-play worker image"
	@echo "  make build-trainer     Build trainer image"
	@echo "  make build-evaluator   Build evaluator image"
	@echo "  make build-all         Build all images"
	@echo "  make push              Push all images to Artifact Registry"
	@echo ""
	@echo "Local Testing:"
	@echo "  make local-test        Run local docker-compose test"
	@echo "  make local-smoke       Run smoke test (1 game + 1 step)"
	@echo "  make local-down        Stop local containers"
	@echo ""
	@echo "Vertex AI:"
	@echo "  make deploy-pipeline   Deploy training pipeline"
	@echo "  make run-iteration     Run single training iteration"
	@echo "  make estimate-cost     Estimate costs for configuration"
	@echo ""
	@echo "Configuration:"
	@echo "  PROJECT_ID=$(PROJECT_ID)"
	@echo "  REGION=$(REGION)"
	@echo "  BUCKET=$(BUCKET)"

# ============================================================================
# Development
# ============================================================================

.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-dev
install-dev: install
	$(PIP) install pytest pytest-cov pytest-asyncio mypy ruff

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

# Mirrors the CI `coverage` job: same suites, same marker filter, same gate.
# It previously ran the whole tests/ tree at --cov-fail-under=80 while CI used
# 85 over a narrower set, so passing locally proved nothing about CI.
.PHONY: test-all
test-all:
	$(PYTHON) -m pytest tests/unit/ tests/sanity/ tests/integration/ tests/cloud/unit/ tests/regression/ \
		-m "not slow and not requires_env" \
		--ignore=tests/unit/test_distributed \
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

# Mirrors CI's `security` job, which runs BOTH the pytest suite and a bandit
# scan. The bandit half was missing entirely, so `make verify` could pass while
# CI's security job failed.
.PHONY: test-security
test-security:
	$(PYTHON) -m pytest tests/security/ -v --tb=short
	$(PYTHON) -m bandit -r alpha_zero/ -s B101 --severity-level high --confidence-level high

# Fetch the pinned gitleaks binary only when it is not already available.
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
		if [ "$$plat" = "windows" ]; then ext=zip; else ext=tar.gz; fi; \
		url="https://github.com/gitleaks/gitleaks/releases/download/v$(GITLEAKS_VERSION)/gitleaks_$(GITLEAKS_VERSION)_$${plat}_$${cpu}.$${ext}"; \
		echo "Installing gitleaks $(GITLEAKS_VERSION) ($${plat}_$${cpu}) -> $(CURDIR)/.tools"; \
		mkdir -p $(CURDIR)/.tools; \
		if [ "$$ext" = "zip" ]; then \
			curl -sSfL "$$url" -o $(CURDIR)/.tools/gitleaks.zip && \
			unzip -oq $(CURDIR)/.tools/gitleaks.zip -d $(CURDIR)/.tools && \
			rm -f $(CURDIR)/.tools/gitleaks.zip; \
		else \
			curl -sSfL "$$url" | tar -xz -C $(CURDIR)/.tools gitleaks; \
		fi; \
		mv -f $(CURDIR)/.tools/gitleaks $(GITLEAKS_BIN); \
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
verify: lint-check typecheck test test-security security-secrets test-governance validate test-all eval-llm
	@echo "verify: reproduced CI's lint, type, test, security, governance, coverage and eval gates"

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

.PHONY: publish-spaces
publish-spaces:
	huggingface-cli upload Mango-Metrics-NLM/Civ6-AlphaZero-Microciv-Explorer ./spaces/microciv-explorer --repo-type space
	huggingface-cli upload Mango-Metrics-NLM/Civ6-AlphaZero-Elo-Dashboard ./spaces/elo-dashboard --repo-type space

# ============================================================================
# Docker Build
# ============================================================================

.PHONY: build-base
build-base:
	docker build \
		-f dockerfiles/Dockerfile.base \
		-t $(LOCAL_BASE_IMAGE) \
		.

.PHONY: build-selfplay
build-selfplay: build-base
	docker build \
		-f dockerfiles/Dockerfile.selfplay \
		--build-arg BASE_IMAGE=$(LOCAL_BASE_IMAGE) \
		-t $(LOCAL_SELFPLAY_IMAGE) \
		.

.PHONY: build-trainer
build-trainer: build-base
	docker build \
		-f dockerfiles/Dockerfile.trainer \
		--build-arg BASE_IMAGE=$(LOCAL_BASE_IMAGE) \
		-t $(LOCAL_TRAINER_IMAGE) \
		.

.PHONY: build-evaluator
build-evaluator: build-base
	docker build \
		-f dockerfiles/Dockerfile.evaluator \
		--build-arg BASE_IMAGE=$(LOCAL_BASE_IMAGE) \
		-t $(LOCAL_EVALUATOR_IMAGE) \
		.

.PHONY: build-all
build-all: build-base build-selfplay build-trainer build-evaluator
	@echo "All images built successfully"

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

.PHONY: push-base
push-base:
	docker tag $(LOCAL_BASE_IMAGE) $(BASE_IMAGE)
	docker push $(BASE_IMAGE)

.PHONY: push-selfplay
push-selfplay:
	docker tag $(LOCAL_SELFPLAY_IMAGE) $(SELFPLAY_IMAGE)
	docker push $(SELFPLAY_IMAGE)

.PHONY: push-trainer
push-trainer:
	docker tag $(LOCAL_TRAINER_IMAGE) $(TRAINER_IMAGE)
	docker push $(TRAINER_IMAGE)

.PHONY: push-evaluator
push-evaluator:
	docker tag $(LOCAL_EVALUATOR_IMAGE) $(EVALUATOR_IMAGE)
	docker push $(EVALUATOR_IMAGE)

.PHONY: push
push: push-base push-selfplay push-trainer push-evaluator
	@echo "All images pushed to $(ARTIFACT_REGISTRY)"

# ============================================================================
# Local Testing
# ============================================================================

.PHONY: local-test
local-test:
	docker-compose up -d
	@echo "Waiting for services to start..."
	sleep 10
	docker-compose logs -f trainer

.PHONY: local-smoke
local-smoke:
	@echo "Running local smoke test..."
	CIV6_STORAGE_BACKEND=local \
	CIV6_STORAGE_BASE_PATH=./test_storage \
	$(PYTHON) -m alpha_zero.cloud.selfplay_worker \
		--num-games=1 \
		--iteration=0 \
		--worker-id=0 \
		--num-simulations=10 \
		--device=cpu \
		--dry-run
	@echo "Smoke test passed"

.PHONY: local-down
local-down:
	docker-compose down -v

.PHONY: clean-test-storage
clean-test-storage:
	rm -rf ./test_storage

# ============================================================================
# GCS Setup
# ============================================================================

.PHONY: setup-bucket
setup-bucket:
	@echo "Creating GCS bucket gs://$(BUCKET)..."
	gsutil mb -p $(PROJECT_ID) -l $(REGION) gs://$(BUCKET) || true
	gsutil lifecycle set infra/gcs_lifecycle.json gs://$(BUCKET)

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

# BROKEN: `pipelines/` does not exist in this repository and never has, so this
# target fails immediately. Kept with the failure made explicit rather than
# silently advertising a deployment path that cannot work. The live orchestrator
# is `python -m alpha_zero.cloud.gcp.vertex_orchestrator`.
.PHONY: deploy-pipeline
deploy-pipeline:
	@echo "deploy-pipeline is not implemented: the 'pipelines' package does not exist."
	@echo "Use: $(PYTHON) -m alpha_zero.cloud.gcp.vertex_orchestrator --help"
	@exit 1

.PHONY: run-iteration
run-iteration:
	@echo "Running training iteration $(ITERATION)..."
	$(PYTHON) -m infra.vertex_launcher \
		--mode=iteration \
		--iteration=$(ITERATION) \
		--selfplay-profile=selfplay_t4_spot \
		--train-profile=train_l4_single \
		--eval-profile=eval_l4_spot

.PHONY: run-selfplay
run-selfplay:
	$(PYTHON) -m infra.vertex_launcher \
		--mode=selfplay \
		--iteration=$(ITERATION) \
		--num-workers=$(NUM_WORKERS) \
		--games-per-worker=$(GAMES_PER_WORKER)

.PHONY: run-trainer
run-trainer:
	$(PYTHON) -m infra.vertex_launcher \
		--mode=train \
		--iteration=$(ITERATION) \
		--training-steps=$(TRAINING_STEPS)

.PHONY: run-evaluator
run-evaluator:
	$(PYTHON) -m infra.vertex_launcher \
		--mode=eval \
		--iteration=$(ITERATION) \
		--num-games=$(EVAL_GAMES)

.PHONY: estimate-cost
estimate-cost:
	$(PYTHON) -c "from alpha_zero.cloud.config import CloudConfig; \
		c = CloudConfig.from_env(); \
		print('Estimated cost per iteration:', c.estimate_iteration_cost())"

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

.PHONY: clean-all
clean-all: clean clean-test-storage
	docker-compose down -v || true
	docker rmi $(LOCAL_BASE_IMAGE) $(LOCAL_SELFPLAY_IMAGE) \
		$(LOCAL_TRAINER_IMAGE) $(LOCAL_EVALUATOR_IMAGE) 2>/dev/null || true

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help
