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

Execute the verification commands from CLAUDE.md Section 4.1:

```bash
# 1. Lint check
ruff check alpha_zero/

# 2. Type check
mypy alpha_zero/ --ignore-missing-imports

# 3. Unit tests
pytest tests/unit/ -v --tb=short

# 4. Integration tests
pytest tests/integration/ -v --tb=short

# 5. Full test suite with coverage
pytest tests/ -v --cov=alpha_zero --cov-report=term-missing

# 6. Smoke tests (as applicable)
python -m alpha_zero.training_civ6 --dry_run
```

Record pass/fail status for each. **This verification gate is mandatory: do not
mark PR-READY unless lint, types, and the touched tests pass** (see Rule 6).

### 2b. Independent Review (isolated context)

Harness-engineering principle: a model over-rates its own work, so **evaluation
must run with context independent from generation**. Do not eyeball your own diff
inline. Instead, dispatch the review to a fresh agent so judgement is decoupled:

- Run the `code-review` skill on the diff, **or**
- Launch an `Agent` (e.g. `subagent_type: Explore`) whose only task is to review
  the changed files against the spec's acceptance criteria and report defects.

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

### 6. Create PR Draft (PR-READY only)

Create a PR draft document at `kb/pr-drafts/YYYYMMDD-<slug>.md`:

```markdown
# PR: [Title]

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
