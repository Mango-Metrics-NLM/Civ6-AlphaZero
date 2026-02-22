# PRD: Container Rebuild & Training Resumption

## User Story

**As a** ML engineer running the AlphaZero training pipeline  
**I want** a new container image built with all current code fixes  
**So that** MuZero selfplay games succeed and the training pipeline can resume

## Background

Container `v14-muzero` (built ~Feb 18) has stale code where `MuZeroMCTS.search()` has a different calling convention than what the selfplay worker expects. This causes 100% game failures, producing 0 trajectories, which cascades to training failures across all iterations 028-033.

The local codebase has the correct `MuZeroMCTS.search(observation, action_mask, add_noise=True)` signature that matches the selfplay worker's call `mcts.search(obs, action_mask=obs["action_mask"])`.

## Acceptance Criteria

### AC1: Container Build Succeeds

**Given** `cloudbuild.yaml` with `_IMAGE_TAG=v1.1.0`  
**When** Cloud Build is submitted  
**Then** the build completes with all smoke tests passing (lint, unit tests, dry-run selfplay, dry-run training, dry-run muzero-selfplay, orchestrator guard)

### AC2: Selfplay Produces Trajectories

**Given** the new container `v1.1.0` is deployed  
**When** a selfplay job runs 10 games with MuZero MCTS  
**Then** at least 1 game succeeds and trajectories are saved (manifest shows `total_examples > 0`)

### AC3: Training Consumes Trajectories

**Given** selfplay has generated trajectories  
**When** a training job is launched  
**Then** the training worker loads trajectories, creates a model, and completes at least 1 epoch

### AC4: .env Updated

**Given** the new image is pushed  
**When** `.env` is checked  
**Then** `CIV6AZ_CONTAINER_IMAGE_URI` points to the new Artifact Registry URI

### AC5: W&B Key Injected

**Given** `WANDB_API_KEY` is set in the host environment  
**When** a selfplay or training job is launched  
**Then** the job logs show "W&B logging enabled" (not "No API key configured")

## Out of Scope

- Multi-node DDP training
- Model architecture changes
- Cleaning up stale GCS data (separate task)

## Success Metrics

- Container builds in < 30 minutes
- Selfplay success rate ≥ 80% (was 0%)
- Training completes 1 full iteration without crashes
- W&B dashboard shows metrics from the new run
