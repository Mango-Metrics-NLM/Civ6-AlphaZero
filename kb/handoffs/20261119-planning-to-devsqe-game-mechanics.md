# Planning → DEV-SQE Handoff

**Date:** 2026-11-19
**Spec:** kb/specs/20261119-game-mechanics-plan.md (program plan; approved by the maintainer in the planning session of 2026-11-19; role reviews in its section 12)
**Register:** kb/specs/20261119-game-mechanics-gap-register.md (`GM-` rows)
**OpenSpec:** openspec/changes/add-game-rules-kernel, openspec/changes/add-bridge-research-parity
**Probe protocol:** docs/GAME_RULES.md
**Session:** session_01CXusP9Ct9R7eg25YLcY7R7
**Branch:** `claude/focused-lovelace-ri074n`

**Why this plan exists:**
- The maintainer asked for a plan to close the game-mechanics gaps (full gameplay, science, technology).
- No tech tree, research or science yield exists anywhere in Python, although three documents claimed one.
- The richer rules that exist (`alpha_zero/gameplay/`) are reachable only through a defective env the local training path never builds.
- The real-game bridge exports no research state, never offers research, and its search mode cannot work as wired.

**Acceptance Criteria:** Phase 1 only (spec section 2 and Appendix A). Each slice names its command or test; negative controls go in the PR body.
- **GM-S0a:** a golden fingerprint of the default env (integer game state; numpy version recorded) that cannot change without an `ENV_VERSION` bump. It gets its own OpenSpec change (one requirement on `training-reproducibility`).
- **GM-K1, GM-K2:** leaf package `alpha_zero/rules/` with its CLI entry point in GM-K1; one failing fixture per validator rule; hash stable under CRLF and key order; debt ratchet flat; importer refuses to write inside the repository.
- **GM-B0..GM-B5:** generated Lua bundle byte-identical on first generation; Lua 5.1 harness that fails, not skips, in its CI job; exporter parity tested on the harness and confirmed live by the maintainer; additive wire fields; snapshots only under `CIV6AZ_DATA_ROOT`; `--mcts` refuses with its gap id after an un-mocked test records the failure.
- **Every Phase 1 PR:** the play-path diff of spec section 8 is empty.

**Constraints:**
- Nothing matched by `scripts/release_play_path.txt` changes until v2.1.0 ships. That includes `alpha_zero/envs/Agent.md` and `alpha_zero/gameplay/Agent.md`.
- The `default` preset stays byte-identical. MicroCiv stays the sole training environment (NG-1); the mod stays read-only on official surfaces (NG-3).
- Confirm-first items covered by the plan approval are listed in spec section 7. Two are not covered and need the maintainer at their PR: END_TURN reporting a blocked turn (ships with its Python handling), and GM-B3 spanning four bridge modules.
- Every simulator rung (GM-S0 onward) needs its own OpenSpec change and task plan, after v2.1.0. Their release slots are proposals; the register records the scrum-master's alternative.
- No real GameInfo tables or name lists are committed; fixtures are synthetic.
- Jobs stay under 1,000 games. No training card for a new rule set before A1 and A1-R are recorded.

**Priority:** medium. Phase 1 yields to the v2.1 tasks when review time is short; at most two Phase 1 PRs open, inside the cap of three open agent PRs.

**Blocked On:**
- Maintainer: merging each PR; the live checklist after GM-B2 and GM-B4; the v2.1 runs (T0b, A1, A1-R) before any simulator rung.
- Order: GM-K1 before GM-K2 and GM-B3; GM-B0, GM-B1, GM-B2, GM-B3 in sequence; GM-B2 before GM-B4.
- Not blocked: authoring GM-S0a, GM-K1, GM-K2 and GM-B0..GM-B5.
