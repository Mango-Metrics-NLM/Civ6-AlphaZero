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
(the former `api_server.py` here was an unused duplicate, removed 2026-07-25).

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

- `api_schemas.py` — request/response schemas.
- `inference_engine.py` — batched inference with masking.
- `client.py` — client SDK.
- `README.md` — quick start.
