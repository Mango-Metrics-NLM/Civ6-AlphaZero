# Next-Steps Analysis — Civ6-AlphaZero (2026-07-05)

_A codebase + documentation scan to determine the best logical development steps forward._
_Generated against the `claude/civ6-alphazero-implementation-vMnun` integration tip (`d740bc6`,
v1.7.0). CI on `main` and the integration branch is green as of the latest run (2026-06-20)._

---

## TL;DR

The build is healthy (green CI, ~339 source files, ~441 test files) and the **genuine
unfinished-work surface is small and concentrated**. The main problem is not missing code — it
is **(a) documentation that points developers at work already done, (b) a critical fix still
sitting unmerged on the convergence path, and (c) a newly-merged chess/LLM feature that is inert
and un-runnable due to three concrete, well-specified bugs.**

**Recommended order:**

1. **Reconcile the roadmap with reality** (this doc + doc fixes) — the current roadmap is stale.
2. **Land PR #72** (champion-load fix) — it gates the project's own stated #1 priority.
3. **Make the chess/LLM path actually work** — fix packaging, wire MCTS, batch LLM calls.
4. **Finish N-player training (G9 only)** — the env/encoder/loss already exist.
5. **Pay down the ~20 statically-skipped tests** (dominated by the W&B→MLflow migration).

---

## 1. Documentation drift (highest-confidence finding)

Several roadmap/known-issue entries are **contradicted by the current code**. A developer
following the docs today would start work that is already finished.

| Doc claim | Reality (evidence) | Action |
|---|---|---|
| CLAUDE.md §5.3: "MicroCiv hardcoded for 2 players … env needs G7/G8/G9 changes" | **False.** `envs/microciv.py` takes `num_players` (2–6): angular start placement (`:429`), turn rotation `% self.num_players` (`:734`), N-player victory/elimination (`:784`), most-cities truncation over `range(num_players)` (`:325`). Encoder is N-player aware with **dynamic channels** `3 + 4·num_players + 2` (`envs/civ6_encoder.py:353`). **G7 (env) and G8 (encoder) are done.** | Correct the row; scope remaining work to **G9 only** (per-player value in the training pipeline). |
| CLAUDE.md §5.3: "12 DDP tests xfailed … need rewrite" | **False.** The DDP test files use CUDA `skipif`, not `xfail`. Repo-wide there is exactly **one** real `xfail` (`tests/integration/test_phase2.py:381`). The `test_pbt_mutations.py` "xfail" hits are comments noting they *used to be* xfail and now pass. | Correct the row; drop the "DDP xfail" line item. |
| NEXT_STEPS.md header: "v1.3.0 … awaiting validation" (dated 2026-03-01) | Repo is **v1.7.0**. The roadmap predates cloud deploy (v1.6.0), full-Civ6 local gameplay + LLM eval harness (v1.7.0), and the entire **chess/LM-Studio** direction (PR #75, merged). | Supersede with this doc; refresh the header. |
| `test_e2e_simultaneous.py:40` skip reason: "DUCT … not yet implemented" | `DecoupledUCT` **is** implemented (`multiplayer/simultaneous_mcts.py:297`). The `skipif` is a legitimate optional-import guard (degrades when torch is absent locally), but the **reason string is stale**. | Reword the skip reason to "optional deps unavailable". |

## 2. Genuine functional stubs (small)

The entire `alpha_zero/` tree has only:

- **1 substantive runtime stub:** `multiplayer/game_connector.py:374` —
  `_apply_action_to_state()` raises `NotImplementedError` (needs the Unciv action-decoder;
  deferred to "Phase 3+"). This is the real gap for live multiplayer connection.
- **1 intentional redirect:** `hierarchical/goals/goal_node.py:384` (`from_dict` → "use
  `GoalHierarchy.from_dict`"). Not missing work.
- **3 placeholder telemetry values:** `inference/api_server.py:142/143/179`
  (`active_sessions=0`, `cache_hit_rate=0.0`, `mcts_time_ms=total_time`).
- **1 TODO:** `cloud/gcp/vertex_jobs.py:824` (multi-node rendezvous args, when needed).

Everything else that pattern-matches as a "stub" (`...` bodies, `pass`) is legitimate
Protocol/ABC declarations, exception-class bodies, or optional-import guards.

## 3. Test debt (~20 statically-disabled tests)

Not `xfail`/hardware-skips (those are fine), but tests turned **off** at module/function level:

- **W&B → MLflow migration remnant (the bulk):** whole suites disabled —
  `test_wandb_baseline_metrics.py`, `test_wandb_logger.py`,
  `integration/test_wandb_integration.py`, `test_orchestrator_wandb.py`,
  `test_pbt_dashboarding.py`, `test_wandb_coordination.py`. This matches the CLAUDE.md
  "W&B→MLflow swap is partial" known issue — `cloud/`, `pbt/`, `evaluation/`, `optimizations/`
  still call wandb directly.
- **Fixture-hungry suites:** `test_hierarchical_orchestrator.py` (×3, "requires full game
  state with units"), `test_state_extraction.py` (×2), `test_training_smoke.py` (×2),
  cloud `test_training_worker.py` / `test_tournament_worker.py`.

## 4. Unmerged PRs on the critical path

| PR | State | Relevance |
|---|---|---|
| **#72** — champion-load-before-optimizer (v1.5.1) | **open, non-draft, CI-passing** | The fix for the "**0 promotions across 50 iterations / $28.55 wasted**" bug: the cloud `training_worker.py` trained from **random weights** each iteration. NEXT_STEPS names "validate training convergence" as the blocking #1 priority — **that validation cannot succeed until this lands.** Open since March. **Highest leverage; needs a human merge decision.** |
| **#84** — v1.8.0 Civ6 Steam local demo | open, **draft**, in-flight | Continues v1.7.0 bridge work (Lua action handlers, `BridgeMCTSEnv`, installer). Substantial but partial; gated on a live-Steam demo (hard, user-verified). |
| **#49 / #32** — Strategic Planning SDK | open | A large domain-generalization direction (pandemic/climate/supply-chain MCTS). Architecturally divergent from the Civ6 core — **decide keep vs. shelve** before it rots. |
| **#69** — repo structure cleanup | open, draft (Copilot) | Mechanical hygiene (drop `setup.py`, move root scripts). Low-risk; land or close. |

## 5. The chess / LM-Studio direction is merged but inert

PR #75 (merged) added a chess env + Qwen "opening-prior" advisor
(`envs/chess_env.py`, `orchestrator/chess_advisor.py`, `scripts/run_chess_training.py`). Three
root-level `ISSUE_*.md` files document that it does not actually work yet — each has a clear
repro and acceptance criteria:

- **`ISSUE_editable_install_submodule_discovery.md`** — `python scripts/run_chess_training.py`
  fails with `ModuleNotFoundError: alpha_zero.utils.checkpoint` after a clean `pip install -e .`,
  for **every** contributor (a `sys.path`/editable-install interaction, not machine-specific).
  → **Packaging bug that blocks the documented entry point. Most tractable, highest unblock value.**
- **`ISSUE_chess_advisor_placeholder_game_loop.md`** — the script plays **random legal moves**
  (`run_chess_training.py:318/322`); it constructs the advisor but never calls MCTS, so Qwen is
  never consulted. → **Wire MCTS + advisor so the merged feature is functional.**
- **`ISSUE_batch_qwen_calls_across_mcts_leaves.md`** — the advisor issues 30–60 **sequential**
  Qwen calls per move (2+ hr/game on CPU; 0.16 % CPU-busy). Batching leaf expansions ≈ **30×
  wall-clock speedup**. → **Performance; do after the two above.**

---

## Recommended path forward

### Track A — Reconcile & unblock (do first; low-risk, high-leverage)
1. **Docs:** apply the §1 corrections (this doc + CLAUDE.md §5.3 rows + NEXT_STEPS header). _Done in this change._
2. **PR #72:** review and land the champion-load fix — it gates the stated #1 priority.
3. **Then** run the existing local convergence gate: `python scripts/validate_v130_training.py --iterations=5 --games=10` (exists) and confirm loss↓ / win-rate↑.

### Track B — Make a shipped feature real (self-contained, verifiable)
4. Fix `ISSUE_editable_install_submodule_discovery` (packaging) → the chess entry point runs.
5. Wire MCTS+advisor into `run_chess_training.py` (`ISSUE_chess_advisor_placeholder_game_loop`).
6. Batch Qwen calls across MCTS leaves (`ISSUE_batch_qwen_calls_across_mcts_leaves`) — ~30× speedup.

### Track C — Close the N-player loop (env/encoder already done → G9 only)
7. Thread per-player values through the training pipeline (`TrainingExample.value` → vector,
   wire the already-present `multiplayer.compute_nplayer_value_loss`), then train a 3–4 player
   MicroCiv agent. Much smaller than the roadmap implies (G7/G8 are complete).

### Track D — Test-debt paydown (steady background)
8. Finish the W&B→MLflow swap in `cloud/`, `pbt/`, `evaluation/`, `optimizations/` and
   re-enable the ~6 disabled W&B suites (or delete them if W&B is being retired).
9. Add the fixtures the hierarchical / state-extraction / training-smoke suites need.
10. Reword the stale `DUCT … not yet implemented` skip reason.

### Decision needed from maintainer
- **Strategic Planning SDK (#49/#32):** keep as a supported product line or shelve? It is large
  and diverging; leaving it half-merged is the worst state.

---

## Single best next step

If one thing: **land PR #72, then run `validate_v130_training.py`.** Everything downstream
(batch self-play, N-player training, MuZero PBT) is premised on training that actually improves
the champion — and per the project's own history that has never been demonstrated end-to-end.
Track B is the best *autonomous, self-contained* alternative if the convergence run must wait on
GCP/human review, because it turns an already-merged but non-functional feature into a working one.
