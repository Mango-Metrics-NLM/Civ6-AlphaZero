---
name: pre-pr-handoff
description: Run at the end of every Pre-PR session that did meaningful work — runs full verification, dispatches an independent review, writes the PR draft and ledger event, and rewrites session state. Gated by scripts/validate_kb.py.
allowed-tools: Read, Write, Edit, Bash, Agent
---

# Pre-PR Handoff Skill

Run this skill at the end of every Pre-PR session that does meaningful work.

## Role

The **Pre-PR** role handles validation, PR preparation, and documentation.
It receives completed work from DEV-SQE and produces PR-ready output.

## Steps

### 1. Read Current State

Read `kb/sessions/pre-pr.state.json` to understand the current validation task.

### 2. Run Full Verification

```bash
make verify          # lint + format + types + the suites CI runs + validators
make test-all        # coverage, mirroring CI's coverage job
```

`make verify` is the single definition of the gate — see the `verify` skill.
Do not hand-assemble the sequence here. The version that used to live in this
file ran `ruff check alpha_zero/` (CI checks the whole repo and also runs
`ruff format --check`), `mypy alpha_zero/ --ignore-missing-imports` (CI passes
no such flag; the config lives in `pyproject.toml`) and a bare
`pytest tests/ --cov` (CI runs four suites with marker filters and its own
coverage scope). Passing that sequence proved nothing about CI — and this is
the *last* role before a PR, so it is the worst place to run a weaker gate
than the one downstream.

Add the smoke test when the change touches training:

```bash
python -m alpha_zero.training_civ6 --dry_run
```

Record pass/fail status for each. **This verification gate is mandatory: do not
mark PR-READY unless lint, types, and the touched tests pass** (see Rule 6).
If a suite could not run, say so and name CI as its gate — never report a suite
green that you did not run.

### 2b. Independent Review (isolated context)

Harness-engineering principle: a model over-rates its own work, so **evaluation
must run with context independent from generation**. Do not eyeball your own diff
inline. Instead, dispatch the review to a fresh agent so judgement is decoupled:

- Dispatch `sdlc-qa-sdet` (`Agent` with `subagent_type: sdlc-qa-sdet`) with the diff
  and the spec's acceptance criteria. It is the review of record for every DEV-SQE
  output. Edit/Write are limited to `tests/**`, and its shell to the read-only commands
  plus `python -m pytest` on paths under `tests/` (`readonly_shell_guard.sh --qa`).
  Residual risk: pytest executes test code, so this is an accident net, not a sandbox;
  read the tests it ran; **or**
- Run the `code-review` skill on the diff, **or** launch another fresh `Agent`
  (e.g. `subagent_type: Explore`) whose only task is to review the changed files
  against the acceptance criteria and report defects.

Treat the subagent's findings as the review of record; fold fixes back through
DEV-SQE (QA-GAP) if they are non-trivial. Note in the summary that review ran in
an isolated subagent.

### 3. Determine Handoff Type

- **SESSION-END**: Validation not complete, continuing next session.
- **PR-READY**: All checks pass, PR draft is ready.
- **QA-GAP**: Found a quality issue requiring DEV-SQE fix.
- **SCOPE-CHANGE**: Found a scope issue requiring Planning input.

### 4. Write Summary

Append a new entry **at the top** of `kb/summaries/pre-pr.md` (create if it doesn't exist):

```markdown
## [DATE] - [Brief Title]
- **Session:** [session-id]
- **Validating:** [what was validated]
- **Lint:** [pass/fail]
- **Types:** [pass/fail]
- **Tests:** [X pass, Y fail, Z% coverage]
- **Smoke Tests:** [pass/fail]
- **Status:** [PR-ready / QA-gap / scope-change]
- **Next:** [what the next session should pick up]
```

Cap at 20 entries.

### 5. Append Ledger Event

Append a single valid JSON line to `kb/ledger/ledger.jsonl`:

**For PR-READY:**
```json
{ "event": "PR-READY", "timestamp": "<ISO-8601>", "spec_file": "kb/specs/<filename>.md", "pr_draft": "kb/pr-drafts/<filename>.md", "session_id": "<id>", "status": "Awaiting Human Push" }
```

**For QA-GAP:**
```json
{ "event": "QA-GAP", "timestamp": "<ISO-8601>", "session_id": "<id>", "handoff_to": "dev-sqe", "issues": ["<issue1>", "<issue2>"], "blocking_pr": true }
```

**For SCOPE-CHANGE:**
```json
{ "event": "SCOPE-CHANGE", "timestamp": "<ISO-8601>", "session_id": "<id>", "handoff_to": "planning", "reason": "<why scope changed>", "impact": "<what is affected>" }
```

**IMPORTANT:** Only append — never overwrite or truncate the ledger file.

### 5b. Choose the target branch and PR type (PR-READY only)

The repository has three long-lived branches: `dev` (default, integration), `qa`
(release candidates) and `main` (tagged releases); see
`docs/runbooks/branching-and-release.md`. Decide both before drafting:

| PR type | Head branch | Base | Merge method | Who merges |
|---|---|---|---|---|
| Feature / fix / chore / docs | `feature/*`, `fix/*`, `chore/*`, `docs/*`, `claude/*` | `dev` | squash | normal review |
| Release prep (version + CHANGELOG) | `chore/release-vX.Y.Z` | `dev` | squash | normal review |
| Back-merge | `backmerge/*` | `dev` or `qa` | merge commit | maintainer (`merge_backmerge.sh`) |
| Promotion | `dev` | `qa` | merge commit | maintainer |
| Evidence | `evidence/vX.Y.Z-rc.N` | `qa` | merge commit | maintainer (only `docs/evidence/**` may change) |
| RC fix | `rcfix/<topic>` | `qa` | merge commit | maintainer (cherry-picks of dev commits, `-x`) |
| Release | `qa` | `main` | merge commit | maintainer |
| Hotfix | `hotfix/*` | `main` | merge commit | maintainer |

Dispatch `sdlc-release-manager` to set the target branch and PR type from this
table and to draft the promotion, release or back-merge PR text and the bundle
manifest. It returns text; `sdlc-developer` commits it with `authored-by:
sdlc-release-manager`, and the release manager then reviews the PR.

An agent opens only the first two rows (against `dev`) and, when asked, drafts the
others; **it never merges into `qa` or `main` and never tags.** Those are
maintainer-only, enforced by identity once the maintainer provisions the
`civ6az-agent` machine user and the rulesets, and a convention until then. The
Branch Direction check will fail a wrong-direction PR, but it is an accident net
and can be faked by any Write-holder; do not rely on it. Put the chosen PR type and
base in the PR draft and tick the matching box in `.github/PULL_REQUEST_TEMPLATE.md`.

### 6. Create PR Draft (PR-READY only)

Create a PR draft document at `kb/pr-drafts/YYYYMMDD-<slug>.md`:

```markdown
# PR: [Title]

## Target
- **Base branch:** [dev | qa | main]
- **PR type:** [feature | release prep | back-merge | promotion | evidence | rc fix | release | hotfix]

## Summary
[2-3 bullet points describing the changes]

## Changes
[List of files changed with brief description of each]

## Test Results
- **Unit Tests:** X pass, Y fail
- **Integration Tests:** X pass, Y fail
- **Coverage:** Z% (target: 85%)
- **Lint:** clean
- **Type Check:** clean
- **Smoke Tests:** [pass/fail per test]

## Breaking Changes
[List any breaking changes, or "None"]

## Checklist
- [ ] All verification commands pass
- [ ] Coverage >= 85% on changed modules
- [ ] CLAUDE.md updated if needed
- [ ] No hardcoded values (API keys, paths, magic numbers)
- [ ] Type hints on all new public interfaces
```

### 7. Update Session State

Update `kb/sessions/pre-pr.state.json` with:

```json
{
    "last_session_id": "<current-session-id>",
    "last_timestamp": "<ISO-8601>",
    "current_focus": "<what to resume next>",
    "resume_pointers": {
        "current_task": "<next task>",
        "blocked_on": [],
        "next_steps": ["<step1>", "<step2>"]
    },
    "context_files": ["<relevant files to read on resume>"]
}
```

## Rules

1. NEVER overwrite or truncate `kb/ledger/ledger.jsonl` — only append.
2. Handoff files in `kb/handoffs/` are immutable once created.
3. PR drafts in `kb/pr-drafts/` are immutable once created.
4. All JSONL entries must be valid JSON — validate after writing.
5. Summary entries are capped at 20 (remove oldest from bottom).
6. All verification commands MUST pass before marking PR-READY.

## Final gate

This handoff is not complete until the KB validator exits 0:

```bash
python scripts/validate_kb.py
```

If it fails, fix the reported ledger or session-state issue before ending the
session.
