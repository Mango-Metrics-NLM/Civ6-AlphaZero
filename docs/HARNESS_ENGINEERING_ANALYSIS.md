# Harness Engineering — Analysis & Application to Civ6-AlphaZero

> Source article: *"Building Claude Code with Harness Engineering"* — Fareed Khan,
> Level Up Coding (Apr 2026).
> `https://levelup.gitconnected.com/building-claude-code-with-harness-engineering-d2e8c0da85f0`
>
> **Sourcing note:** the original article sits behind the gitconnected/Medium
> paywall (HTTP 403 on every retrieval path, including the freedium mirror, at the
> time of writing). The thesis and component taxonomy below were reconstructed from
> the indexed search abstract (which quotes the article directly) and corroborated
> against three open primary sources describing the same "harness engineering"
> framework: `shareAI-lab/learn-claude-code`, `AnastasiyaW/claude-code-config`
> (`principles/01-harness-design.md`), and the DEV.to "Learn Harness Engineering by
> Building a Mini Claude Code" walkthrough. Claims that are inference rather than
> direct quotation are marked *(inferred)*.

---

## 1. What the article argues

The central thesis: **Claude Code's success came from building the right *harness*
around a capable model, not from constraining the model with orchestration logic.**
Agency — the capacity to perceive, reason, and act — comes from *model training*;
the harness is the infrastructure that lets that trained intelligence operate
reliably in a specific domain. The slogan is *"trust the model, engineer the
harness."*

The article decomposes the harness into a small set of components, each of which
generalizes well beyond Claude Code:

| # | Harness component | What it does |
|---|-------------------|--------------|
| H1 | **Single-threaded agent loop** | `messages → LLM → response → (tool_use? execute & append : return)`. One loop, never rewritten; everything else hangs off it. |
| H2 | **Typed tool-dispatch registry** | Tool name → handler, each with a strict input schema. New tools register without touching the loop. |
| H3 | **Context-management layer** | Keeps the model focused across arbitrarily long sessions: snipping, micro-compaction, result budgeting, and — preferred over compaction — **context reset with structured handoffs** (`PLAN.md`, `STATE.json`, `FINDINGS.md`). |
| H4 | **Permission-governed dispatch** | Sandboxing, approval workflows, trust boundaries that constrain actions *before* execution. |
| H5 | **Generator–Evaluator separation** | Models overrate their own work (self-evaluation bias), so generation and judgement run as *independent* agents with independent context, the evaluator calibrated with few-shot examples of good work and hidden flaws. |
| H6 | **Sprint Contract** | Generator and evaluator agree on testable, frozen success criteria *before* work starts — prevents endless rejection loops from misaligned expectations. |
| H7 | **Escalation via trace similarity** | When consecutive attempts produce near-identical execution traces, the loop has *stagnated*; escalate (trace-guided debugging, multiple debater agents weighted by historical success) rather than spinning. |
| H8 | **Subagents & isolation** | Complex sub-tasks spawn fresh agents with isolated histories, returning only results to the parent. Worktree/namespace isolation prevents interference. |
| H9 | **On-demand knowledge** | Domain expertise (docs, APIs, style guides) is injected only when needed, not front-loaded. |
| H10 | **Assumption testing** | Harness components encode assumptions about model *limitations*; as models improve those assumptions expire. Periodically disable a guardrail and measure the impact, to avoid permanent over-engineering. |

A recurring economic point *(inferred from the harness-design source)*: solo agents
are cheap and fast but fail on hard tasks; full harnesses cost ~20× and run hours
but produce qualitative leaps. Match harness weight to task difficulty.

---

## 2. The key realization: this repo already runs *two* harnesses

Civ6-AlphaZero is unusually well-positioned to absorb this article, because it
already operates two loops that are each, structurally, a harness around a model:

- **Harness A — the agentic-coding harness.** `CLAUDE.md` (which is explicitly
  framed as "prompting as constraint programming"), `.claude/settings.json`
  (permissions + `SessionStart`/`PostToolUse`/`PreCompact`/`Stop` hooks), the
  `.claude/skills/*-handoff` skills, and the `kb/` knowledge base. This is how
  *Claude builds the project*.

- **Harness B — the AlphaZero training orchestrator.** `alpha_zero/orchestrator/`
  plus the cloud `selfplay_worker → trainer → evaluator` loop. This is the
  *product itself* — a harness wrapped around a neural network (the "model").

The article's framework maps onto **both**, and the second mapping is where the
highest-value, most surprising leverage is.

---

## 3. Application A — harden the agentic-coding harness

The repo already implements most of the framework. The gaps are about *enforcement*
and *isolation*, not invention.

| Article component | Already present | Gap / proposed upgrade |
|-------------------|-----------------|------------------------|
| H3 Context reset w/ structured handoffs | `kb/sessions/*.state.json`, `kb/handoffs/`, `kb/summaries/`, append-only `kb/ledger/ledger.jsonl`, `PreCompact` hook. **This is a textbook implementation** of "handoffs over compaction." | Largely done. Minor: add result-budgeting guidance for tool output. |
| H5 Generator–Evaluator separation | Three roles (Planning → DEV-SQE → Pre-PR) with feedback events (`DESIGN-GAP`, `QA-GAP`, `SCOPE-CHANGE`). | **They run in one context.** The article insists the evaluator needs *independent context*. Run **Pre-PR (and `code-review`) as a subagent** (the `Agent` tool) so judgement is genuinely decoupled from generation. |
| H6 Sprint Contract | `CLAUDE.md §1.2` mechanically-verifiable success criteria; planning handoff docs with acceptance criteria. | Done. Make the criteria the *frozen contract* the Pre-PR subagent checks against, verbatim. |
| H7 Trace-similarity escalation | `CLAUDE.md §4.2`: "same error 3× → stop and escalate." | A coarse proxy. Could formalize as a stop-condition the agent self-checks (identical failing test + identical diff twice ⇒ escalate to `AskUserQuestion`). |
| H4 Permission-governed dispatch | `settings.json` allow/deny lists; deny on `gcloud`/`gsutil`/`kubectl`/`docker push`/`rm -rf`. | The `CLAUDE.md §3.2` **PROHIBITED** list (commit to `main`, training > 1000 games) is *documented but unenforced*. Add a **`PreToolUse` hook** that blocks `git commit`/`git push` onto `main` and self-play/training invocations above the game cap. |
| H2 Typed dispatch / verification loop | `PostToolUse` runs `ruff --fix` + `ruff format` on Python edits. | Verification is otherwise manual (`§4.1`). The repo already ships a `verify` skill — wire a lightweight **gate** (e.g. `pytest -q` on touched modules, or `mypy` on the changed file) into `PostToolUse` or a pre-handoff step. |
| H9 On-demand knowledge | `RAGIntegration`, `data/knowledge/`, `alpha_zero/rag/`. | Already strong. Could expose the strategy/knowledge base to the *coding* agent too, not only to training. |
| H10 Assumption testing | — | **Missing.** Add a recurring "guardrail audit": periodically disable a hook (e.g. the 3×-failure stop, or the autolinter) and confirm it still earns its keep as the model improves. A dated checklist in this doc suffices. |
| H8 Subagents / worktree isolation | `Agent` tool + `isolation: worktree` available. | Underused. The Planning/DEV-SQE/Pre-PR pipeline is the natural place to adopt isolated subagents (one writer to `ledger.jsonl`, per §14.5 rule 6). |

**Concrete, low-risk first moves for Harness A**

1. Add a `PreToolUse` guard in `.claude/settings.json` that denies commits/pushes to
   `main` and blocks `--num_games`/`--num-games` over the configured cap — turning
   the §3.2 PROHIBITED list from prose into enforcement (H4).
2. Run Pre-PR / `code-review` as an isolated subagent so evaluation has independent
   context (H5) — the single highest-fidelity move toward the article's thesis.
3. Add an "Assumption / guardrail audit" log section and revisit it each release (H10).

---

## 4. Application B — treat the AlphaZero orchestrator *as* a harness (highest value)

This is the non-obvious application. The AlphaZero loop *is* a harness around a
model, so the article's components translate directly — and several translate onto
**open problems already on this project's roadmap**.

| Article component | AlphaZero analogue | Leverage |
|-------------------|--------------------|----------|
| H1 Agent loop | `selfplay → train → eval → (promote?)` iteration loop | The loop is the orchestrator; keep it minimal, hang policy off it. |
| H5 Generator–Evaluator | Self-play actor (generator) vs. **promotion evaluator** with `promotion_threshold=0.55`, plus independent **baseline eval** (random/heuristic) | The architecture *already* separates generation from judgement. The article validates the design and argues the evaluator must stay *independently calibrated* — exactly what the random/heuristic baselines provide. Keep the evaluator's checkpoint/seed independent of the trainer's. |
| H6 Sprint Contract | Frozen `promotion_threshold` + win-rate criteria per iteration | Already a frozen, testable contract. Good. |
| H7 **Escalation via trace similarity** | **Detecting training stagnation** | **This is the headline insight.** The v1.2.0 run had *0 promotions* (the train-from-scratch bug, §13). In harness terms: consecutive iterations produced near-identical evaluation traces — flat win-rate vs. incumbent — and the harness **had no stagnation detector to escalate**. Adding one (if win-rate-vs-incumbent or value-correlation flatlines for *N* iterations ⇒ alert / change hyperparameters / halt) would have caught the bug in 2–3 iterations instead of a whole production run. This directly serves §13's "Validate training convergence." |
| H3 Context reset via handoffs | Checkpoints + `SharedStorage` + `DistributedReplayBuffer`; MLflow run history as an analogue of `ledger.jsonl` | The training state *is* persisted as structured handoffs across iterations. The mapping is clean; MLflow is the ledger. |
| H4 Permission boundaries | Spot-VM `PreemptionHandler`, action masking, replay-buffer memory caps (the `CLAUDE.md §1.3` invariants) | These are the training harness's "trust boundaries." Frame them explicitly as such. |
| H8 Subagents / isolation | Distributed self-play actors with isolated game state, returning only trajectories | Already the architecture. The "return only results" pattern = trajectories to the buffer. |
| H10 Assumption testing | Periodically re-validate guardrails (e.g. does paranoid/Max-N still help? is the 0.55 threshold still right as the net improves?) | The opponent-modeling and MCTS-variant choices encode assumptions about a *weak* net; revisit as it strengthens. |

**The single most valuable action this article motivates** is the
**training-stagnation detector** (H7) wired into the orchestrator and MLflow
monitoring: it converts the most expensive past failure mode (a silent
zero-promotion production run) into an early, cheap, automatic escalation. The
monitoring layer (`alpha_zero/monitoring/`, `compute_value_correlation`,
`compute_elo_spread`) already computes the right signals — this is mostly a
threshold + alert + early-stop policy on top of existing metrics.

---

## 5. Prioritized recommendations

Ordered by value-to-effort:

1. **Training-stagnation detector (Harness B, H7).** Add an early-stop/alert policy
   to the orchestrator + MLflow: flat win-rate-vs-incumbent or value-correlation
   for *N* iterations ⇒ escalate. Directly addresses §13 and the past 0-promotion
   failure. *High value, low–medium effort (signals already computed).*
2. **Pre-PR/code-review as an isolated subagent (Harness A, H5).** True
   generator–evaluator separation for the coding workflow. *High value, low effort.*
3. **`PreToolUse` enforcement of the PROHIBITED list (Harness A, H4).** Turn §3.2
   prose into a hook (no `main` commits, game-count cap). *Medium value, low effort.*
4. **Verification gate in the loop (Harness A, H2).** Wire the existing `verify`
   skill / a fast `pytest`/`mypy` check into `PostToolUse` or pre-handoff.
   *Medium value, low effort.*
5. **Guardrail/assumption audit (H10, both harnesses).** A dated checklist to retire
   guardrails as the model and the net improve. *Low effort, compounding value.*

The throughline: **Civ6-AlphaZero already practices harness engineering** — the
`CLAUDE.md`/`kb/` system is one of the cleaner real-world implementations of
"handoffs over compaction." The article's contribution here is (a) a vocabulary
that names what the repo already does, and (b) two specific upgrades — *independent
evaluation* and *stagnation escalation* — that the project currently under-applies,
the second of which maps squarely onto its top open problem.
