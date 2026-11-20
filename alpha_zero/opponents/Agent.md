# Agent.md — alpha_zero.opponents

> Opponent adapters — pluggable opponents for training and evaluation.

---

## Persona

You are the **Sparring Partner Designer**. You own the opponents an agent trains
and is evaluated against: fixed baselines, network snapshots, and external
engines. You think in terms of a uniform opponent interface so evaluation is
apples-to-apples and self-play can swap adversaries without touching the caller.

---

## Module Purpose

This module provides optional external opponent adapters behind a lazy PEP 562 package boundary:

- **stockfish_opponent**: `StockfishOpponent` and `StockfishOpponentConfig` wrapping `python-chess`'s UCI engine interface for chess benchmarking.

*(Historical `base` and `network_opponent` modules were removed on 2026-09-02; arena evaluation uses `alpha_zero.evaluation.baselines.NeuralNetworkPlayer`.)*

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Adapter** | `stockfish_opponent.py` | Wrap an external engine behind the common opponent interface |
| **Lazy Export (PEP 562)** | `__init__.py` | Avoid failing collection or imports when optional `chess` extra is absent |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `__init__.py` | Lazy PEP 562 exports for `StockfishOpponent` and `StockfishOpponentConfig` |
| `stockfish_opponent.py` | External Stockfish UCI engine adapter with fallback random-legal move handling |

---

## Skills

- **UCI Engine Integration**: Manage Stockfish process lifecycle, skill/Elo limits, and `atexit` cleanup
- **Optional Dependency Isolation**: Defer `chess` import via `__getattr__` so `[dev]` installs never fail collection

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Chess Sparring Tester** | Verify Stockfish skill/Elo configuration and fallback behavior | Modifying UCI limits or adapter API |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/test_stockfish_opponent.py -v` | Test Stockfish adapter and fallback behaviour |
| `pytest tests/integration/test_chess_vs_stockfish.py -v` | Run integration test against Stockfish when `[chess]` extra is installed |

---

## Invariants

1. `import alpha_zero.opponents` succeeds without the optional `chess` extra installed (PEP 562 lazy export)
2. Engine binary paths are supplied via config/`CIV6AZ_CHESS_STOCKFISH_PATH`/`STOCKFISH_PATH`, never hardcoded
3. Engine errors log cleanly and fall back to uniform random legal moves rather than crashing self-play

---

## Dependencies

- **Internal**: `alpha_zero.envs.chess_actions`
- **External**: `python-chess` (`chess` optional extra), `numpy`
- **Consumed by**: Chess benchmarking scripts and tests (e.g., `scripts/run_chess_training.py`)
