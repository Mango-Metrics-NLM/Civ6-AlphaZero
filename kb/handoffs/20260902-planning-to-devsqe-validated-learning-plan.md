# Planning → DEV-SQE Handoff

**Date:** 2026-09-02
**Spec:** kb/specs/20260902-validated-learning-plan.md (v2, after adversarial review)
**Session:** session_01AQAVGXog3wGfPaqu4XHfXX
**Branch:** `claude/dev-roadmap-next-steps-fk867n`, PR #117

**Why this plan exists:** the project has never shown its training loop learns. The only
committed run reports "no learning shown"; ~70% of games at the shipped arena
configuration end in a 1–1 city tie, which compresses every statistic toward 0.5 and makes
the promotion default arithmetically unreachable. Two code readings disagreed about the
lever (game length vs geometry), so it was measured: at board 8, random-play draws are
0.68 / 0.47 / 0.37 at 30 / 60 / 100 turns, and board size at 30 turns makes it worse.
Independent re-run reproduced the numbers within noise.

**Acceptance Criteria:** spec §3, each with its gate and its vacuity guard:
- L1: a seeded ≥20-iteration local run at board 8 / 60 turns with `--save_interval 1`,
  committed with a `VALIDATION-RUN` event and `check_learning_trend.py`'s recorded verdict
  (exit 2 below 3 evaluation points).
- D0: the draw-rate table under L1's checkpoints and under 3-player random play.
- D1: one `max_turns` default on the deployed path, set from D0, six test assertions updated.
- D2: zero mask/executor disagreements on moves under seeded random play.
- E1: every draw-blind site in §2.2 emits score rate + CI; the regex gate over
  `wins\s*/\s*(max\(|num_|total|len\()` has a negative control.
- C1: one from-scratch, games-capped, NG-7-signed cloud run with shadow `score_rate_lower_bound`.
- N1: seeded `NPlayerMCTS` (no `np.random.` left in `multiplayer/mcts.py`) selected by `num_players`.
- N2 only after D0's 3-player row ≤ 40% (or D3); F1/F2/F3, B1/B2, R1 per §3.

**Constraints:**
- WS-L starts on day 1; it depends on nothing else in the plan. D0 runs on its checkpoints
  (no `.pt` is tracked in git).
- No network-architecture change, no new training environment (NG-1), no LLM in the loop
  (NG-2). No hyperparameter change other than `max_turns`, and that only on D0's evidence.
- Every new gate ships with the negative control named in spec §8.
- `TrainingConfig` (16/200) is not the arena's source; D1 does not touch it.
- Do not copy `gameplay/victory.py:323`'s tie-break (seat bias) if D3 is approved.
- C1 runs from scratch; resuming champion v268 is a separate later decision.

**Priority:** high. Core (L1, D0–D2, E1–E5, C1, N1, B1–B2) ~3–4 weeks; first learning-trend
artifact in week 1.

**Blocked On:**
- Spec §9.2: D1 (≥9 modules), F3 — confirm-first.
- Spec §9.3: E5 and WS-C's coordinator edits — confirm-first (distributed coordination).
- Spec §9.1: D3/D4/F1 environment-semantics decisions — OpenSpec change each.
- Spec §9.4: NG-7 sign-off before any Vertex spend (WS-C, B3).
- Spec §9.5 / §7: maintainer-only — key rotation, the 15 PR dispositions, GCS lineage inventory.

Executable today without any of the above: WS-L, D0 (once L1 has checkpoints), D2,
E1–E4, N1, B1, B2, WS-T.
