# Planning → DEV-SQE Handoff

**Date:** 2026-04-02
**Spec:** kb/specs/20260402-full-civ6-local-gameplay.md
**Priority:** High

## Acceptance Criteria
1. Training convergence validated (loss ↓ across iterations, ≥1 promotion in 5 iters)
2. MicroCiv `num_players` configurable 2–8, dynamic encoder channels
3. EncoderRegistry factory (13-ch, 84-ch, hex-crystalline) — no hardcoded channel counts
4. Bridge mock tests pass; state/action converters updated for 84-ch
5. Hierarchical training xfail fixed and all tests passing
6. `Civ6GameAdapter` implemented, web UI serves Civ6 sessions
7. Fog of war determinization wired to multiplayer MCTS
8. DDP xfails = 0
9. `pytest tests/ -v --cov=alpha_zero --cov-fail-under=85` passes
10. ruff clean, mypy clean, no hardcoded values

## Constraints
- Python 3.10+, 100-char line limit
- `CIV6AZ_` prefix for all env var overrides
- Dataclasses with `from_yaml()`, `from_env()`, `from_preset()` on all configs
- Factory/registry patterns throughout
- Backwards compatible — existing MicroCiv 2-player tests must keep passing

## Implementation Order
Phase 0 → Phase 1 → Phase 2 → Phase 3 → (Phase 4 + 5 + 6 in parallel) → Phase 7 (parallel) → Phase 8

## Blocked On
- Phase 1–8 all blocked on Phase 0 (training convergence validation)
- Phase 3–8 blocked on Civ6 being installed locally for final E2E (bridge mock tests do NOT require Civ6)
