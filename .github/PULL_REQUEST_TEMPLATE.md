## Summary

<!-- What does this PR change, and why? One paragraph; the diff is the detail. -->

## Tracking issue

<!-- #NN, or "none — <why this did not need an issue>". -->

## Linked OpenSpec change

<!-- Path under openspec/changes/<id>, or "N/A — no requirement change". -->

## Hot files touched

<!-- Files several in-flight branches edit (pyproject.toml, CLAUDE.md, README.md,
     Makefile, .github/workflows/ci.yml, kb/ledger/ledger.jsonl). List each, or
     "none". The integration lead rebases these last. -->

## Verification

- [ ] `make verify-fast` passes (lint, mypy, governance tests, regression suite, validators)
- [ ] `make verify` passes (every CI gate, including coverage and the eval gate) — required before merge
- [ ] `make typecheck` is clean. mypy is a **blocking** CI gate (since 2026-07-25), not advisory
- [ ] `make validate` passes (openspec, kb, skills, agents, doc-sync)
- [ ] Coverage ≥ 85% on `alpha_zero/` for changed modules; `coverage-floors.json` only ever raised
- [ ] Docs updated (README / MODULE_REFERENCE / CHANGELOG as applicable)
- [ ] `CLAUDE.md` §5 updated if new commands/decisions/known-issues

## Negative control recorded

<!-- For every new or changed test: the defect you planted, the command you ran,
     and the failure you observed before reverting. "Not applicable — no test
     changes" is a valid answer; "passed" alone is not. See
     .claude/skills/negative-control/SKILL.md. -->

## CONFIRM-FIRST items and who authorized

<!-- CLAUDE.md §3.2: architectural changes across >3 modules, breaking public
     APIs, >100-line deletions, distributed-training coordination, network
     architecture, new external services, new token scopes in workflows. For
     each: what, and who approved it (name + where). "None" if none. -->

## Notes

<!-- Breaking changes, follow-ups, or anything a reviewer should know. -->
