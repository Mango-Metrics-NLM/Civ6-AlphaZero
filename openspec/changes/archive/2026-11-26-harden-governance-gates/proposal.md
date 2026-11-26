# Harden the governance gates

## Why

Three gaps were found while decomposing `core/civ6_mcts.py`, each of which let a
whole class of defect through unnoticed:

1. **No secret scanning.** `tests/security/test_security_sanity.py` asserts the
   no-secrets policy against a hand-written pattern list. Nothing scanned git
   history, and nothing used a maintained ruleset. Charter Invariant 7 depends on
   that guarantee.
2. **`Agent.md` persona docs had zero validator coverage.** Thirty files, no gate,
   no hook, no test. They had drifted: one had no persona section and asserted that
   Weights & Biases "remains in place" (retired 2026-07-13), one package had no
   `Agent.md` at all, and several referenced files that no longer exist.
3. **A second skill tree was ungated.** `.agents/skills/` was routed to
   `validate_skills.py` by the PostToolUse hook but scanned by neither pre-commit
   nor CI, so it carried live violations — including a coverage literal 5 points
   below the gate.

Separately, the repo had no mechanical limit on function size. The pathology it
actually has is a 623-statement `main`, and module line count is only a proxy for
that.

## What Changes

- **Secret scanning** (`gitleaks`, pinned in the Makefile and read from there by
  CI): the blocking gate scans the tracked tree and the commit lineage reachable
  from `HEAD`; `make security-secrets-audit` sweeps every ref, non-blocking.
  Measured 2026-08-30: 0 findings on the tree, 0 across the branch lineage the gate covers
  (532 commits then — a count that moves with every push, unlike the scope). The
  repo-wide audit is **not** clean — it found an unrotated credential on an
  unmerged branch (`docs/runbooks/secret-scanning.md`), which is why the two scopes
  are separate targets rather than one. The "0 findings across 525 commits"
  baseline originally quoted here was taken before `git fetch --all` and did not
  cover the branch carrying that credential.
- **`scripts/validate_agents.py`**: a fifth deterministic validator (codes A1–A6)
  for `Agent.md` persona documents — heading form, persona statement, required
  sections, coverage literals, package coverage, and referenced-file existence.
  Wired into the PostToolUse hook, pre-commit, `make validate`, and CI, with 13
  governance tests.
- **`validate_skills.py`** now scans `.agents/skills/` as well, gains an S5 rule
  (skill directory without a `SKILL.md`), and parses YAML block scalars so a folded
  `description:` is validated rather than read as the literal `">"`.
- **ruff `PLR0915`** at `max-statements = 100`, with the three over-limit `main`
  functions listed in `per-file-ignores` as an explicit burndown list.
- A percent-matching bug shared by two validators is fixed: `(\d{2,3})\s*%` matched
  the decimal tail of `89.30%` as a bare `30%`.

## Impact

- Affected specs: `validation-gates`
- Affected code: `scripts/validate_agents.py` (new), `scripts/validate_skills.py`,
  `.gitleaks.toml` (new), `Makefile`, `.github/workflows/ci.yml`,
  `.pre-commit-config.yaml`, `.claude/hooks/posttooluse_governance.sh`,
  `.claude/settings.json`, `tests/governance/test_validate_agents.py` (new),
  `tests/regression/test_repo_hygiene_regression.py` (new)
- No runtime behaviour changes. All additions are gates and documentation.
