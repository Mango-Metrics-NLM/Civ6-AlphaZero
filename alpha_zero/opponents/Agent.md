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

This module provides a common opponent contract and concrete implementations:

- **base**: the abstract `Opponent` interface (given a state, return an action).
- **network_opponent**: an opponent driven by a policy/value network snapshot.
- **stockfish_opponent**: an external-engine opponent adapter (chess benchmarking).

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | `base.py` — `Opponent` ABC with concrete impls | Swap opponents without changing evaluation/self-play code |
| **Adapter** | `stockfish_opponent.py` | Wrap an external engine behind the common interface |
| **Snapshot** | `network_opponent.py` | Freeze a network version as a fixed adversary |

---

## Key Files

- `base.py` — `Opponent` interface.
- `network_opponent.py` — network-backed opponent.
- `stockfish_opponent.py` — external chess-engine opponent.
