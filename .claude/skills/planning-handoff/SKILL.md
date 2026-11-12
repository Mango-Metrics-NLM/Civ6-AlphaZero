---
name: planning-handoff
description: Run at the end of every Planning session that did meaningful work — updates the summary, appends a ledger event, writes any cross-role handoff document, and rewrites the session state. Gated by scripts/validate_kb.py.
allowed-tools: Read, Write, Edit, Bash, Agent
---

# Planning Handoff Skill

Run this skill at the end of every Planning session that does meaningful work.

## Role

The **Planning** role handles task breakdown, architecture decisions, acceptance criteria,
and spec authoring. It hands off to DEV-SQE for implementation.

## Steps

### 1. Read Current State

Read `kb/sessions/planning.state.json` to understand the current focus and resume pointers.

### 2. Determine Handoff Type

- **SESSION-END**: No cross-role handoff needed (continuing Planning work next session).
- **PLANNING-HANDOFF**: Handing off a spec/task to DEV-SQE for implementation.
- **SCOPE-CHANGE**: Responding to a Pre-PR feedback event.

### 2b. Dispatch the role subagents

Planning work is done by persisted role subagents in `.claude/agents/`
(`scripts/validate_claude_agents.py` checks them). Dispatch each as an `Agent`
with `subagent_type` set to the role name, and give it the spec path and the
question, not your conclusion:

| Role | Dispatch when |
|---|---|
| `sdlc-product-owner` | always: stories, acceptance criteria, scope |
| `sdlc-scrum-master` | always: sequencing, the compute queue, run cards, WIP and capacity |
| `sdlc-architect` | always: design, confirm-first triage (CLAUDE.md section 3.2), OpenSpec; reviews anything touching more than 3 modules |
| `sdlc-ml-scientist` | the work involves ML results, a pre-registration or an evidence audit |
| `sdlc-ai-engineer` | the work involves LLM, RAG, the StrategyAdvisor or the agent harness |

These roles are read-only: each returns text (a memo, script, doc, PR body or
manifest). Do not write it yourself. Hand it to `sdlc-developer`, which commits it
verbatim with `authored-by: sdlc-<role>`; the authoring role then reviews that PR,
so the author is never the evaluator of its own text. Until the role files exist
in a checkout, brief a general-purpose subagent from the role's draft text.

### 3. Write Summary

Append a new entry **at the top** of `kb/summaries/planning.md` with:

```markdown
## [DATE] - [Brief Title]
- **Session:** [session-id]
- **Focus:** [what was planned/decided]
- **Decisions:** [key architecture or scope decisions]
- **Output:** [spec file or task list created]
- **Next:** [what the next session should pick up]
```

Cap at 20 entries — if there are more than 20, remove the oldest entries from the bottom.

### 4. Append Ledger Event

Append a single valid JSON line to `kb/ledger/ledger.jsonl`:

**For SESSION-END:**
```json
{ "event": "SESSION-END", "role": "planning", "timestamp": "<ISO-8601>", "session_id": "<id>", "summary": "<one-line summary>" }
```

**For PLANNING-HANDOFF:**
```json
{ "event": "PLANNING-HANDOFF", "timestamp": "<ISO-8601>", "spec_file": "kb/specs/<filename>.md", "session_id": "<id>", "handoff_to": "dev-sqe", "summary": "<one-line summary>" }
```

**IMPORTANT:** Only append — never overwrite or truncate the ledger file.
Validate that the appended line is valid JSON after writing.

### 5. Create Handoff Document (PLANNING-HANDOFF only)

If handing off to DEV-SQE, create an immutable file:

`kb/handoffs/YYYYMMDD-planning-to-devsqe-<slug>.md`

Contents:
```markdown
# Planning → DEV-SQE Handoff

**Date:** YYYY-MM-DD
**Spec:** kb/specs/<filename>.md
**Acceptance Criteria:** [list]
**Constraints:** [any architectural constraints or decisions]
**Priority:** [high/medium/low]
**Blocked On:** [dependencies, if any]
```

This file is **immutable** once created — never modify handoff documents.

### 6. Update Session State

Update `kb/sessions/planning.state.json` with:

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

## Final gate

This handoff is not complete until the KB validator exits 0:

```bash
python scripts/validate_kb.py
```

If it fails, fix the reported ledger or session-state issue before ending the
session.
