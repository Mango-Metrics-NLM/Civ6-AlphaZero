# Epic 3: Batch Self-Play Cloud Wiring — PRD

## User Story

**As a** training infrastructure engineer  
**I want** validated batch self-play using `BatchGumbelMCTS` + `VectorizedMicroCivEnv`  
**So that** I can deploy 5-10x throughput self-play workers on Vertex AI GPU instances

## Background

All batch self-play infrastructure already exists:

- `run_batch_selfplay()` in `selfplay_worker.py` (L859-1074)
- `BatchGumbelMCTS` in `batch_mcts.py` (654 lines)
- `VectorizedMicroCivEnv` in `vectorized_env.py` (339 lines)
- CLI args: `--num_parallel_games`, `--max_batch_size`, `--mcts_type=batch_gumbel`
- `main()` branching in `selfplay_worker.py` (L1175-1238)
- `vertex_orchestrator.py` passes `--num_parallel_games` to container args

**The gap is test coverage and validation.**

## Acceptance Criteria

### AC1: Batch Self-Play Dry Run

**Given** a selfplay_worker with `--mcts_type=batch_gumbel --num_parallel_games=4 --dry_run`  
**When** the worker runs  
**Then** it exits 0 and produces trajectory output

### AC2: Integration Test — run_batch_selfplay

**Given** a small network and 4 parallel envs  
**When** `run_batch_selfplay()` runs 2 games  
**Then** it returns valid trajectory dicts with correct keys and shapes

### AC3: Orchestrator Parameter Passthrough

**Given** orchestrator args with `--num_parallel_games=8 --max_batch_size=32`  
**When** the orchestrator builds container args  
**Then** `--num_parallel_games=8` appears in the forwarded CLI args

### AC4: Trajectory Format Consistency

**Given** trajectories from batch and sequential self-play  
**When** comparing trajectory dict schemas  
**Then** both produce identical key sets (`spatial`, `vector`, `policies`, `actions`, `rewards`)

## Out of Scope

- GPU performance benchmarking (requires cloud hardware)
- Actual Vertex AI deployment (requires GCP credentials)
- Modifying core `BatchGumbelMCTS` or `VectorizedMicroCivEnv` logic

## Success Metrics

- All new tests pass locally
- Batch dry-run exits 0
- ≥80% coverage on `run_batch_selfplay()` path in `selfplay_worker.py`
