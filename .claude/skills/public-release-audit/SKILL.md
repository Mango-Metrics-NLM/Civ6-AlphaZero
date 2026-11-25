---
name: public-release-audit
description: Audit repository hygiene, version alignment, secret/ignore gates, coverage floors, tech-debt ratchets, and play-path boundaries before preparing or verifying a public release branch or tag.
allowed-tools: Read, Grep, Glob, Bash
---

# Public Release Audit

Run this workflow before preparing a public release branch, rolling `CHANGELOG.md`,
or handing a release candidate tag to the maintainer. Read-only checks run first;
mutating release bumps run only after every readiness check is green.

## Steps

### 1. Run deterministic release-readiness and doc-sync gates

```bash
python scripts/check_release_readiness.py --verbose
python scripts/check_doc_sync.py --verbose
make check-release-readiness
```

All R1-R6 checks (public files and README framing, version/citation/changelog/security
alignment, ignore and gitleaks rules, coverage floors and `85%` gate, pinned workflows
and launcher safety, git tree hygiene) must report `PASS` or expected `SKIP`.

### 2. Verify debt ratchets, coverage floors, and secret hygiene

```bash
python scripts/measure_debt.py --compare docs/evidence/tech_debt_baseline/baseline.json
make test-cloud-coverage
make security-secrets
```

Confirm zero regressions against `docs/evidence/tech_debt_baseline/baseline.json`,
confirm `alpha_zero/cloud` meets `coverage-floors.json`, and inspect tracked ignores
in `.gitignore`, `.dockerignore`, and `.gitleaks.toml`.

### 3. Verify release play-path isolation and version contract

Inspect `scripts/release_play_path.txt`, `CITATION.cff`, `SECURITY.md`, and
`CHANGELOG.md`. For a versioned release `vX.Y.Z`, run:

```bash
python -m scripts.release_version check --tag vX.Y.Z
python -m scripts.release_version notes --tag vX.Y.Z
```

### 4. Escalation rules

See `kb/specs/20261110-public-release-readiness.md` and
`docs/runbooks/branching-and-release.md`:

- Keep changes clean across `tests/regression/`, `tests/security/`, and
  `tests/governance/`.
- Do not push branches or tags autonomously; hand off `make validate`,
  `make lint`, `make typecheck`, `make test`, `make test-all`, `make eval-llm`,
  and tag creation to the maintainer.
