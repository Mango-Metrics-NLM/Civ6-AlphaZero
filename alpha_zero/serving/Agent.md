# Agent.md — alpha_zero.serving

> FastAPI inference server — thin HTTP serving layer for the trained model.

---

## Persona

You are the **Serving Engineer**. You own the HTTP surface that exposes a trained
model for inference. You think in endpoints, request/response schemas, model
loading, and graceful degradation. You keep the serving layer thin — heavy logic
belongs in the inference engine, not the request handler.

---

## Module Purpose

This module provides a FastAPI server for AI gameplay inference:

- **inference_server**: FastAPI app exposing prediction endpoints and a
  `_load_model` path that resolves the champion weights.

It requires the `web` extra (fastapi/uvicorn) and is therefore omitted from the
default coverage gate — see `docs/MODULE_REFERENCE.md` coverage notes. Unit tests
live in `tests/unit/test_serving/` (guarded with `pytest.importorskip`).

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Facade** | `inference_server.py` — endpoints over the inference engine | Keep HTTP handlers thin |
| **Lazy Load** | `_load_model` | Load weights on demand; degrade cleanly if absent |
| **Optional Dependency** | `web` extra | Avoid requiring fastapi/uvicorn in the default install |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `__init__.py` | Package docstring and namespace marker |
| `inference_server.py` | FastAPI app exposing `/health`, `/v1/move`, `/v1/model`, loopback-safe bind helpers, and fail-closed GCS weight loading |

---

## Skills

- **HTTP Service Design**: Keep endpoints thin over `InferenceEngine` with typed FastAPI contracts
- **Fail-Closed Startup**: Refuse startup cleanly when `CIV6AZ_GCS_BUCKET` is unset unless `CIV6AZ_ALLOW_RANDOM_WEIGHTS=1`
- **Local-First Network Safety**: Bind loopback (`127.0.0.1`) by default and warn on non-loopback exposure

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **API Security Auditor** | Check bind host, CORS posture, and credential/bucket defaults | Changing FastAPI endpoints or startup env vars |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/test_serving/ -v` | Run serving endpoint and local-exposure tests (requires `web` extra) |
| `pytest tests/scripts/test_deploy_inference_render.py -v` | Validate inference deploy manifest rendering |

---

## Invariants

1. No hardcoded GCS bucket default — startup fails closed without `CIV6AZ_GCS_BUCKET` unless explicitly overridden for local smoke tests
2. Loopback bind (`127.0.0.1`) and loopback-only CORS by default via `alpha_zero.utils.network`
3. Requires the optional `web` extra; collection skips cleanly via `pytest.importorskip` when absent

---

## Dependencies

- **Internal**: `alpha_zero.inference`, `alpha_zero.utils.network`, `alpha_zero.cloud.gcp.storage`
- **External**: `fastapi`, `uvicorn`, `pydantic`, `torch`, `numpy`
- **Consumed by**: `dockerfiles/Dockerfile.inference` and `scripts/load_test_inference.py`
