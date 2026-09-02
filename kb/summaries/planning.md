# Planning Log

## 2026-09-02 - Tech-debt reduction and hardening plan
- **Session:** session_01EcSsbbTYiSkmZsMVyipm7G
- **Focus:** Self-reflection audit before the next feature work: coverage reach, dead code, hardcoded values, god files, hardening gaps, CI state. Every figure measured on `062339e` with its rule stated.
- **Decisions:** CI is already green, so the plan widens what CI measures instead of "fixing" it (48% of `alpha_zero/` lines are coverage-omitted, overlapping the unsafe `torch.load`/`pickle` sites almost exactly). Delete dead code before decomposing; decompose only the seven live god files; one definition per config value with a regression test each; ratchets only tighten. `action_space_size` (356/376/381) is treated as a correctness bug, not style. Key rotation (W&B leak) is WS0 and a human action.
- **Output:** `kb/specs/20260902-tech-debt-hardening.md` (8 workstreams, 6–7 weeks, 7 new gates), handoff `kb/handoffs/20260902-planning-to-devsqe-tech-debt-hardening.md`.
- **Next:** Maintainer answers spec §8 (deletion list authorization incl. separate yes/no on `mechanics/`; architectural authorization for WS3.2/WS5/WS7.2). DEV-SQE starts WS1.1 (red draft coverage PR) and WS2.1 regardless.

## Phase 6: PBT (Started 2026-02-08)

- Created spec: `kb/specs/20260208-phase-6-pbt.md`
- Core Modules: `pbt.population`, `pbt.mutation`, `cloud.gcp.pbt_orchestrator`
