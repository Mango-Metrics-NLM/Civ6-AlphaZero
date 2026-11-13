# Local-first plan: panel and peer-review record (2026-11-11)

This is the record of how [`kb/specs/20261111-local-first-and-branching-plan.md`](../../kb/specs/20261111-local-first-and-branching-plan.md) was produced. The gap register is in [`kb/specs/20261111-local-first-gap-register.md`](../../kb/specs/20261111-local-first-gap-register.md). This file records the process and the findings that changed the plan. It is not a second plan.

## Method

| Stage | Agents | Output |
|---|---|---|
| Exploration | 3 | Gap inventory from the project's own records; trace of the local execution path; audit of CI and branch governance |
| Design panel | 3 | Architect / SDLC / release; ML scientist / AI; product owner / scrum |
| Review round 1 | 73 | Six lenses: code claims, ML statistics, GitHub mechanics, Windows operations, delivery, security. Each critical or major finding was checked by one skeptic, then reviewed by a completeness critic and synthesized. **72 findings, 0 refuted, plus 10 critic additions** |
| Review round 2 | 25 | Release-flow simulation, executor-readiness audit and statistics re-check of the rewritten plan, each with a skeptic per finding. **17 confirmed, 0 refuted, plus 8 minor** |

All reviewers were read-only and used GitHub GET calls only.

## Findings that overturned the first draft

**Local loop.**
- `run_iterations` (`alpha_zero/cloud/gcp/vertex_orchestrator.py:1295`) is the CLI loop, not the deployed one. `main()` runs its own inline loop.
- The CLI loop crashes on the first real training phase (`AttributeError 'lr_schedule_type'`).
- The plan therefore completes `run_iterations` and claims no parity with Vertex.

**Isolation.**
- On the workstation, a "local" run could reach production GCS. Three things make this possible: the ADC file is present, `CIV6AZ_GCS_BUCKET` is set at user scope, and `.env` is loaded automatically.
- Fix: `GCSBackend` refuses to run in local mode, through a `BaseException` that the state saves' `except Exception` cannot swallow.

**Stale code.**
- The installed `alpha_zero` is an editable v1.8.0 install of the main checkout, which predates the 2026-09-04 search fixes.
- Fix: every run executes from pinned source in a pinned environment.

**Identity.**
- Agents act with the maintainer's own GitHub credentials, so "maintainer-only" cannot be enforced on the server.
- The guard was registered for the Bash tool only, while PowerShell is the primary shell.
- Fix: a machine user plus a `qa-main-merge` ruleset. The hooks become an accident net.

**Release mechanics.**
- A `pull_request` bypass mode is invalid on tag rulesets.
- Evidence produced on an RC had no legal route back to qa.
- `wait-for-ci` waited on a PR-only check that never runs on the merge commit being tagged.
- Release Evidence was made required before the job that produces it existed.
- `rcfix` reachability cannot be met by cherry-picks; it is now tested by patch equivalence.
- `backmerge` allowed only one merge commit, but strict mode adds one per update.

**A1 protocol.**
- "Random-init" is now a pinned `checkpoint_0`, and the SimpleMCTS bootstrap is switched off.
- The negative control could not fail. It is replaced by a harness-symmetry control (judged at 99%) and a positive control.
- The decision bounds are the hull of the iid and pair-clustered intervals. The verdict table is ordered so exactly one verdict applies.
- The invocation is registered verbatim, so in-loop evaluation and the arena's default opponents cannot push the card past 1,000 games.

**Statistics.**
- In-loop SPRT P(promote) is recomputed with truncation counted as a reject: 0.07 / 0.13 / 0.33 at +5 / +10 / +20 Elo for d = 0.735. The first draft's 0.10 / 0.18 / 0.43 was not truncated.
- The qa→main gate's realised power is 0.80 / 0.68 / 0.61 at d = 0.735 / 0.5 / 0.4.
- Lowering the draw rate costs games at fixed Elo bounds (1.89× from d = 0.735 to d = 0.5); it does not save them.
- The B2 horizon-adoption rule is re-powered to 300 games per cell.

**Machine.**
- About 15 GB of RAM is free, not the 61 GB installed: llama-server, WSL and OneDrive are resident.
- Windows Update may restart the machine between 01:00 and 07:00.
- E: inherits Modify for Authenticated Users.

## Decisions taken by the maintainer (2026-11-10)

- The default branch becomes `dev`. `qa` holds release candidates and `main` holds tagged releases.
- Data root: `E:\civ6az-data`.
- Each run finishes overnight and plays at most 1,000 games.
- Role subagents are persisted in `.claude/agents/`.
- Agent identity defaults to a machine user. The answer to that question was interrupted; the alternatives are recorded in the plan's §3.6.

## Sources

The research sources are listed in the plan's Appendix B. Context7 was queried for `/github/docs`, `/websites/github_en_actions` and `/websites/mlflow`. Review subagents could not reach the Context7 plugin and used docs.github.com instead.
