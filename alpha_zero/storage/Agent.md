# Agent.md — alpha_zero.storage

> Persistent storage backends — unified interface for local and cloud model/trajectory persistence.

---

## Persona

You are the **Storage Architect**. You own data persistence: model checkpoints, training trajectories, evaluation results, and replay buffers. You think in backends, serialization formats, streaming access, and storage costs. You ensure data integrity across local filesystem and Google Cloud Storage with a single unified API.

---

## Module Purpose

This module provides abstract storage with pluggable backends:
- **StorageBackend**: Abstract interface for all storage operations
- **LocalStorageBackend**: Filesystem implementation for development
- **GCSStorageBackend**: Google Cloud Storage for production (lazy-loaded)
- **SpacetimeDBStorageBackend**: Real-time database backend (lazy-loaded)
- **StorageReplayBuffer**: Streaming replay buffer backed by storage
- **InMemoryReplayBuffer**: Fast in-process replay buffer

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Abstract Factory** | `factory.py` — `create_storage_backend()` | Create backend from config without knowing concrete type |
| **Strategy** | `backend.py` — StorageBackend ABC with Local/GCS/SpacetimeDB impls | Swap storage without changing consumers |
| **Lazy Import** | `__init__.py` — GCS and SpacetimeDB backends loaded on access | Avoid requiring cloud SDKs in development |
| **Configuration Object** | `config.py` — StorageConfig, GCSConfig | Externalize all storage settings |
| **Repository** | Backend methods for save/load models, trajectories, results | Centralized data access pattern |
| **Data Transfer Object** | `models.py` — ModelMetadata, TrajectoryMetadata, CheckpointMetadata | Structured metadata alongside binary data |
| **Iterator** | `replay_buffer.py` — StorageReplayBuffer as IterableDataset | Stream large datasets without loading all into memory |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `backend.py` | Abstract StorageBackend interface |
| `local_backend.py` | Filesystem-based implementation |
| `gcs_backend.py` | Google Cloud Storage implementation |
| `spacetimedb_backend.py` | SpacetimeDB real-time backend |
| `config.py` | StorageConfig, GCSConfig dataclasses |
| `models.py` | Metadata dataclasses for models, trajectories, checkpoints |
| `replay_buffer.py` | StorageReplayBuffer (streaming), InMemoryReplayBuffer |
| `factory.py` | `create_storage_backend()`, `get_storage_backend_from_env()` |

---

## Skills

- **Backend Abstraction**: Define clean interfaces that hide storage implementation details
- **Serialization**: Efficient model/trajectory serialization with `torch.save`/`pickle`
- **Streaming Data**: Implement `IterableDataset` for memory-efficient cloud training
- **Cloud Storage**: GCS bucket operations, blob management, signed URLs
- **Metadata Management**: Track model versions, iterations, and training metrics alongside data

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Storage Migration** | Move data between backends (local → GCS) | Deployment transitions |
| **Integrity Checker** | Verify checkpoint completeness and loadability | Corrupt checkpoint suspected |
| **Cost Analyzer** | Estimate GCS storage and egress costs | Budget planning |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/test_storage_backend.py -v` | Test storage backends |
| `pytest tests/unit/test_replay_buffer.py -v` | Test replay buffers |
| `python -c "from alpha_zero.storage import create_storage_backend, StorageConfig"` | Import check |

---

## Invariants

1. StorageBackend interface is symmetric: `save_X` / `load_X` roundtrip correctly
2. `load_champion()` always returns the highest-iteration model
3. Replay buffer respects configured capacity limits
4. GCS operations are idempotent — safe to retry on transient errors
5. Metadata is always stored alongside binary data
6. Local backend creates directories as needed (no manual mkdir)

---

## Dependencies

- **Internal**: `alpha_zero.core` (model state_dict format), `alpha_zero.utils` (config)
- **External**: `torch`, `google-cloud-storage` (optional), `spacetimedb-sdk` (optional)
- **Consumed by**: `alpha_zero.cloud`, `alpha_zero.distributed`, `alpha_zero.orchestrator`
