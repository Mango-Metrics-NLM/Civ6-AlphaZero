# Spec: Full Civ6 Local Gameplay

**Date:** 2026-04-02
**Planner Session:** 169d78de-01a4-4b35-9699-12ae5a11aaf3
**Target Version:** v2.0.0

## Objective
Enable full Civilization VI gameplay on a local computer through the AlphaZero AI,
connecting the existing bridge/Lua mod, trained neural networks, and web UI into a
cohesive local play experience.

## Acceptance Criteria
- [ ] Training convergence validated: loss decreases across iterations, ≥1 promotion
- [ ] MicroCiv supports 2–8 players via `num_players` config (no hardcoding)
- [ ] Encoder registry selects 13-ch vs 84-ch dynamically from config/env
- [ ] Bridge integration tests pass with mock Civ6 (no real game required for CI)
- [ ] Hierarchical training integration xfail fixed and passing
- [ ] Web UI serves real Civ6 sessions via `Civ6GameAdapter`
- [ ] Fog of war determinization integrated with multiplayer MCTS
- [ ] DDP xfails reduced from 12 to 0
- [ ] Full test suite: `pytest tests/ -v --cov=alpha_zero --cov-fail-under=85`
- [ ] `python -m alpha_zero.civ6_local --dry-run` completes
- [ ] ruff + mypy clean

## Phases
1. **Phase 0**: Validate training convergence (BLOCKING)
2. **Phase 1**: N-Player MicroCiv (G7-G9)
3. **Phase 2**: 84-channel encoder registry
4. **Phase 3**: Bridge validation & hardening
5. **Phase 4**: Hierarchical action space
6. **Phase 5**: Web UI for real Civ6
7. **Phase 6**: Fog of war / partial observability
8. **Phase 7**: DDP test fixes (parallel)
9. **Phase 8**: Full integration entry point

## Constraints
- Python 3.10+, ruff+mypy clean, 100-char line limit
- No hardcoded values — dataclass configs, `CIV6AZ_` env overrides
- Backwards compatible — MicroCiv 2-player pipeline unchanged
- Factory/registry patterns for encoders, MCTS variants, agents
- 85% coverage on new/modified modules

## Key Files
- `alpha_zero/envs/microciv.py` — Phase 1 target
- `alpha_zero/envs/civ6_encoder.py` — Phase 2 target
- `alpha_zero/bridge/` — Phase 3 target
- `alpha_zero/hierarchical/` — Phase 4 target
- `alpha_zero/web/` — Phase 5 target
- `alpha_zero/multiplayer/fog_of_war.py` — Phase 6 target
- `alpha_zero/distributed/` — Phase 7 target
