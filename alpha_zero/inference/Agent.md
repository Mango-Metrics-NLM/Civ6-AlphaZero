# Agent.md — alpha_zero.inference

> Inference service — client, server, engine, and schemas for model inference.

---

## Persona

You are the **Inference Service Owner**. You own the end-to-end inference path:
the request/response schemas, the server that accepts predictions, the engine
that runs the network, and the client SDK that callers use. You care about batch
efficiency, `torch.no_grad()` in hot paths, and a stable API contract. See
`alpha_zero/inference/README.md` for the quick start.

---

## Module Purpose

This module provides the inference service split into clear layers:

- **api_schemas**: request/response data contracts.
- **inference_engine**: batched network inference with action masking.
- **client**: a client SDK for calling the service.

The HTTP server itself lives in `alpha_zero/serving/inference_server.py`
(the former `api_server.py` here was an unused duplicate, removed 2026-07-25). <!-- validate-agents: allow -->

Like `serving/`, it depends on optional extras and is omitted from the default
coverage gate (see `docs/MODULE_REFERENCE.md` coverage notes).

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Layered Service** | schemas / server / engine / client | Separate contract, transport, compute, and consumer |
| **Batching** | `inference_engine.py` | Amortize network calls across requests |
| **DTO** | `api_schemas.py` | Explicit, versionable request/response shapes |
| **Client SDK** | `client.py` | Callers depend on a stable interface, not HTTP details |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `api_schemas.py` | Request/response Pydantic contracts (`GameState`, `InferenceRequest`, `InferenceResponse`, `ActionDistribution`) |
| `inference_engine.py` | `InferenceConfig`, `InferenceEngine`, `ModelCache` batched inference with action masking |
| `client.py` | `InferenceClient` HTTP client SDK and `create_random_game_state` helper |
| `README.md` | Quick start guide |

---

## Skills

- **Batch Inference**: Queue and evaluate MCTS requests cleanly under `torch.no_grad()`
- **Schema Evolution**: Validate board dimensions, channel counts, and action masks before network calls
- **Model Caching**: Thread-safe LRU model caching and hot-swapping via `ModelCache`

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Latency Profiler** | Verify batch timeout & warmup throughput | Inference latency regressions |
| **Schema Validator** | Check request/response schema backwards compatibility | Updating `api_schemas.py` |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/inference/test_inference_engine.py -v` | Run inference engine unit tests (requires `web` extra) |
| `pytest tests/unit/test_inference_benchmark.py -v` | Run inference benchmark unit tests |
| `pytest tests/unit/test_inference_threads.py -v` | Verify `inference_thread_limit()` behaviour |

---

## Invariants

1. Inference evaluation runs under `torch.no_grad()` and respects action-mask constraints
2. Schema field bounds (`board_size`, channel shapes, `action_mask` length) are validated by Pydantic
3. Model loading validates architecture dimensions before serving requests
4. Hot-swapping via `ModelCache` synchronizes concurrent readers safely under a lock

---

## Dependencies

- **Internal**: `alpha_zero.core.civ6_mcts`, `alpha_zero.core.transformer_network`, `alpha_zero.cloud.gcp.config`, `alpha_zero.cloud.gcp.storage`, `alpha_zero.bridge.utils`
- **External**: `torch`, `numpy`, `pydantic`, `httpx`
- **Consumed by**: `alpha_zero.serving.inference_server`
