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
	@echo "  make test-all          Run all tests with coverage"
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
	ruff check alpha_zero/ tests/ --fix
	ruff format alpha_zero/ tests/

.PHONY: lint-check
lint-check:
	ruff check alpha_zero/ tests/
	ruff format --check alpha_zero/ tests/

.PHONY: typecheck
typecheck:
	mypy alpha_zero/ --ignore-missing-imports

.PHONY: test-unit
test-unit:
	pytest tests/unit/ -v --tb=short

.PHONY: test-integration
test-integration:
	pytest tests/integration/ -v --tb=short

.PHONY: test-all
test-all:
	pytest tests/ -v --cov=alpha_zero --cov-report=term-missing --cov-report=html

.PHONY: test-storage
test-storage:
	pytest tests/unit/test_storage_backend.py tests/unit/test_replay_buffer.py -v

.PHONY: eval-llm
eval-llm:
	$(PYTHON) scripts/run_llm_eval.py

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

.PHONY: deploy-pipeline
deploy-pipeline:
	$(PYTHON) -m pipelines.civ6_pipeline --submit

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
