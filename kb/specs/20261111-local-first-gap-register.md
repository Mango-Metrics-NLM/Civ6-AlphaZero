# Local-First Gap Register (Appendix A)

Companion to [`20261111-local-first-and-branching-plan.md`](20261111-local-first-and-branching-plan.md). One row per open gap; `LF-` IDs are namespaced so they do not collide with the validated-learning plan's IDs, CLAUDE.md's G-numbers or the validator codes.

## Appendix A — Gap register (82 rows → `kb/specs/20261111-local-first-gap-register.md`)
Auth: A = autonomous · CF = confirm first · M = maintainer only · OS = needs OpenSpec.

| LF | Gap (source) | Rel | Auth | Owner | Acceptance |
|---|---|---|---|---|---|
| A1 | Post-fix learning run + A1-R (VLP:197; EVAL §7) | 2.1 | A | ml-sci | verdict recorded |
| A2 | Trend checker, descriptive (VLP E4) | 2.1 | A | dev | T13 |
| A3 | ScoreSummary return + `eval_<iter>.json` (VLP E1) | 2.1 | CF | dev | T10 |
| A4 | Network-vs-network eval tool, resolved via D6 | 2.1 | A | dev | T11 |
| A5 | Completed-Q alignment memo + decision (CFP S3) | 2.1 memo | CF | ml-sci | T12 |
| A6 | Search flags | 2.1 | CF (in T8) | dev | T8 |
| A7 | PUCT default; gumbel ignores sims | 2.1 | CF (in T8) | dev | T8 |
| A8 | Resume restores optimizer + replay | 2.2 | A | dev | resumed run equals uninterrupted |
| A9 | Encoder A/B re-run + Hub card | 2.3 | A/M | ml-sci | ≥3 seeds, each arm ≤1k |
| A10 | Bradley-Terry | 2.3 | A | ml-sci | fit |
| A11 | Gumbel budget floor, 3 copies | 2.1 | CF | dev | T8 |
| A12 | σ in sequential halving | 2.1 memo | CF | ml-sci | T12 |
| A13 | Prequential diagnostics | 2.1 | A | dev | T10 |
| A14 | Eval coupled to save_interval | 2.1 | A | dev | T10 |
| A15 | `--bootstrap` + checkpoint_0 | 2.1 | A | dev | T9 |
| A16 | Provenance + horizon refusal | 2.1 | A | dev | T9 |
| A17 | S5 seed-overlap memo | 2.1 | A | ml-sci | T12 |
| B1 | Move-mask collision (= VLP WS-D2) | 2.1 | A/OS | dev | T7 |
| B2 | Same-horizon draw re-measure + 3-player (VLP D0) | 2.2 | A | ml-sci | §5.4 rule |
| B3 | One `max_turns` default (VLP D1) | 2.3 | CF | architect | all modules agree |
| B4 | Tie-break (= VLP D3) | 3.0 / core | CF+OS | architect | OpenSpec archived |
| B5 | City capture (= VLP D4) | 3.0 | CF+OS | architect | " |
| B6 | Fog: mask leak + flag + inert knob (VLP F1/F2) | 2.3 | CF+OS | architect | fogged-mask tests |
| C1 | Seed NPlayerMCTS (VLP N1) | 2.3 | A | dev | determinism test |
| C2 | 3-player run (VLP N2), gated by the §5.4 rule or B4 | 3.0 | A | ml-sci | artifact |
| C3 | Wire DeterminizedMCTS (VLP F3) | 3.0 | CF | architect | call site + test |
| C4 | Oracle over batch_gumbel / MuZero | 2.1 (slips first) | A | qa-sdet | T18 |
| C5 | Batch self-play benchmark + config (VLP B1/B2) | 2.2 | A | ml-sci | §5.4 |
| C6 | inference_benchmark counts actions, not turns | 2.2 | A | dev | test |
| D1 | Promotion-default memo | 2.3 | CF+M | ml-sci | memo |
| D2 | Stagnation halt (= L6) | 2.2 | CF | dev | test |
| D3 | Persist score_rate/LLR/verdict (VLP E5) | 2.2 | CF | dev | round-trip |
| D4 | Draw-blind sites | 2.3 | CF (coord) | dev | allowlist empty |
| D5 | First benchmark baseline | 2.2 | A | ml-sci | contract-valid |
| D6 | Benchmark evidence producer | 2.1 | A | dev | T11 |
| D7 | `sprt` policy | 2.2 | A / CF (default) | dev | ASN + truncated operating-characteristic simulation |
| D8 | Release evidence verifier + check | 2.1 | A | dev | T14 |
| E1 | Local loop (= L0–L6) | 2.2 | CF | architect | v2.2 gate |
| E2 | Seeds through workers | 2.2 | CF | dev | byte-identical shards on CPU |
| E3 | Lock-file strategy | 2.3 | CF+M | architect | CI lock check |
| E4 | Data root | 2.1 | A | dev | T1 |
| E5 | GCP isolation | 2.1 | A | dev | T2 |
| E6 | Pinned env + run-card driver | 2.1 | A | dev | T1, T1b |
| E7 | Calibration + Windows baseline | 2.1 | A | ml-sci | T0a/T0b |
| E8 | MLflow local server | 2.2 | A / CF (major version) | dev | integration job |
| F1 | Web extra in CI; un-omit serving/inference | 2.2 | A | dev | coverage |
| F2 | Raise cloud coverage floor | 2.2 | A | qa-sdet | `coverage-floors.json` |
| F3 | Split 3 long mains (4.4 before L3) | 2.2 | A | dev | waivers removed |
| F4 | Tier-B dead code | cont. | CF | architect | ratchet |
| F5 | Tech-debt WS5–7 | cont. | partly CF | dev | `baseline.json` |
| F6 | Safe checkpoint loading | 2.2 / cont. | CF | dev | `weights_only` sites |
| F7 | Blocking scanners, extra ruff rules | cont. | M | qa-sdet | CI |
| F8 | Genuine skips | cont. | A | qa-sdet | census |
| F9 | Windows CI job "Local Pipeline (windows-latest)" | 2.2 | A | dev | required on dev |
| G1 | W&B key revoke + ref purge + PR-head scan | S0 | M | maintainer | §10.1 |
| G2 | Tag v2.0.0 | S0 | M | maintainer | release green |
| G4 | Charter ratification, CoC contact, eval extra | cont. | M | maintainer | — |
| G5 | Archive OpenSpec changes | 2.1 | A | product-owner | T13 |
| G6 | Agent identity, deny rules, PowerShell guard | 2.1 (S0) | A + M | dev | T4, action 4 |
| G7 | Split CLAUDE.md; distribution rename; public backlog | cont. | CF | architect | — |
| G8 | `CIV6_` → `CIV6AZ_` prefix | cont. | CF | dev | — |
| G9 | PR triage + branch prune (after G1) | S0–1 | M | maintainer | — |
| G10 | Doc contradictions | 2.1 | A | product-owner | T13 |
| G11 | Game-cap cost models | 2.1 | A | dev | T5, T1b |
| G12 | Charter amendment | S0 | M (merge) | product-owner | T3a |
| G13 | Role subagents + validator | 2.1 | A | ai-engineer | T15 |
| G14 | Web/inference bind 127.0.0.1 | 2.1 | A | dev | T16 |
| G15 | Tracking bootstrap | S0 | M | scrum-master | §8.3 |
| G16 | `allUsers` IAM; user-scope bucket var | S0 | M | maintainer | §10.2 |
| H1 | Turn-cycle stall | cont. | A + you | dev | sustained session |
| H2 | Live session with a trained checkpoint (NG-1) | cont. | A + you | ml-sci | session log |
| H3 | Salvage atomic IPC from #106 | 2.3 | A | dev | bridge tests |
| H4 | Unciv stub (NG-6: delete) | cont. | CF | architect | — |
| H5 | 84-channel encoder | M5 | — | — | — |
| L0 | Complete `run_iterations` + JobClient + OpenSpec `local-training-loop` | 2.2 | CF | architect | §4.1 |
| L1 | Storage: atomic, run-scoped, 8 sites | 2.2 | CF | dev | concurrency test |
| L2 | Worker contract, seeds, flags | 2.2 | CF | dev | contract test |
| L3 | Champion pointer + bugs 3, 4, 6, 7 | 2.2 | CF | dev | restart keeps champion |
| L4 | Process client + Job Object + shutdown + profiles | 2.2 | CF | dev | client tests |
| L5 | Calibration, runbook, "Vertex dormant" docs | 2.2 | A | dev | `local-calibrate` |
| L6 | Stagnation halt (= D2) | 2.2 | CF | dev | test |
| R1–R4 | Branching PRs (T4, T6, T3, legacy-name removal) | 2.1 | A + M (apply) | dev / release-mgr | §3 |
