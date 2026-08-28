---
name: dev-sqe-handoff
description: Run at the end of every DEV-SQE session that did meaningful work — runs the verification gate (tests, coverage, lint, types), updates the summary, appends a ledger event, writes any cross-role handoff document, and rewrites the session state. Gated by scripts/validate_kb.py.
allowed-tools: Read, Write, Edit, Bash
---

# DEV-SQE Handoff Skill

Run this skill at the end of every DEV-SQE session that does meaningful work.

## Role

The **DEV-SQE** role handles implementation, testing, and quality engineering.
It receives specs from Planning and hands off to Pre-PR for validation.

## Steps

### 1. Read Current State

Read `kb/sessions/dev-sqe.state.json` to understand the current task and progress.

### 2. Determine Handoff Type

- **SESSION-END**: Continuing DEV-SQE work next session (implementation not complete).
- **DEV-SQE-HANDOFF**: Implementation and tests complete, handing off to Pre-PR.
- **DESIGN-GAP**: Found an issue requiring Planning input (feedback loop).

### 3. Gather Test Results (verification gate)

This is a gate, not a formality: a `DEV-SQE-HANDOFF` must not be emitted unless
lint and types are clean and the touched tests pass. Run and capture:
```bash
make verify          # lint + format + types + the suites CI runs + validators
make test-all        # coverage, mirroring CI's coverage job (gate: 85%)
```

`make verify` is the single definition of the gate — see the `verify` skill.
Do not hand-assemble the sequence here: the version that used to live in this
file ran `ruff check alpha_zero/` (CI checks the whole repo and also runs
`ruff format --check`), `mypy --ignore-missing-imports` (CI passes no such
flag), coverage with no threshold, and none of the sanity, cloud, security or
governance suites. Passing it did not imply passing CI.

For a focused change, also run the specific touched modules
(`pytest tests/unit/<file>.py -q`). If the environment cannot run the suite
(e.g. a missing heavy dependency), say so explicitly in the handoff and flag
that CI must be the gate — do not silently claim green. Include pass/fail
counts in the handoff.

**Coverage honesty:** `alpha_zero/cloud/*` is in the coverage omit list, so
changes confined to the cloud path cannot move the coverage number. Say that
plainly rather than reporting an unchanged percentage as if it were evidence.

**When adding a regression test for a bug you just fixed, run the
`negative-control` skill** and record the result. A test that has never been
seen to fail is not yet known to discriminate.

### 4. Write Summary

Append a new entry **at the top** of `kb/summaries/dev-sqe.md` with:

```markdown
## [DATE] - [Brief Title]
- **Session:** [session-id]
- **Task:** [what was implemented]
- **Files Changed:** [list of modified files]
- **Tests:** [X pass, Y fail, Z% coverage]
- **Status:** [complete/in-progress/blocked]
- **Next:** [what the next session should pick up]
```

Cap at 20 entries — remove oldest from bottom if exceeded.

### 5. Append Ledger Event

Append a single valid JSON line to `kb/ledger/ledger.jsonl`:

**For SESSION-END:**
```json
{ "event": "SESSION-END", "role": "dev-sqe", "timestamp": "<ISO-8601>", "session_id": "<id>", "summary": "<one-line summary>" }
```

**For DEV-SQE-HANDOFF:**
```json
{ "event": "DEV-SQE-HANDOFF", "timestamp": "<ISO-8601>", "spec_file": "kb/specs/<filename>.md", "session_id": "<id>", "handoff_to": "pre-pr", "files_changed": ["<file1>", "<file2>"], "tests_added": ["<test1>"], "summary": "<one-line summary>" }
```

**For DESIGN-GAP:**
```json
{ "event": "DESIGN-GAP", "timestamp": "<ISO-8601>", "session_id": "<id>", "handoff_to": "planning", "gap_description": "<what needs clarification>", "blocking_task": "<task that is blocked>" }
```

**IMPORTANT:** Only append — never overwrite or truncate the ledger file.
Validate that the appended line is valid JSON after writing.

### 6. Create Handoff Document (DEV-SQE-HANDOFF only)

If handing off to Pre-PR, create an immutable file:

`kb/handoffs/YYYYMMDD-devsqe-to-prepr-<slug>.md`

Contents:
```markdown
# DEV-SQE → Pre-PR Handoff

**Date:** YYYY-MM-DD
**Spec:** kb/specs/<filename>.md
**Files Changed:** [list]
**Tests Added:** [list]
**Test Results:** [X pass, Y fail, Z% coverage on changed modules]
**Lint Status:** [clean / N issues]
**Type Check Status:** [clean / N errors]
**Smoke Tests:** [pass/fail for relevant smoke tests]
**Known Issues:** [any caveats or limitations]
```

This file is **immutable** once created.

### 7. Update Session State

Update `kb/sessions/dev-sqe.state.json` with:

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
3. All JSONL entries must be valid JSON — validate after writing.
4. Summary entries are capped at 20 (remove oldest from bottom).
5. Always include test results and coverage in handoff documents.

## Final gate

This handoff is not complete until the KB validator exits 0:

```bash
python scripts/validate_kb.py
```

If it fails, fix the reported ledger or session-state issue before ending the
session.
