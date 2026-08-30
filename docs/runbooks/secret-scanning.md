# Runbook — secret scanning

## What runs, and where

| Scan | Command | Scope | Blocking? |
|---|---|---|---|
| Tracked tree | `gitleaks dir .` | Working tree, minus the artifact paths allowlisted in `.gitleaks.toml` | **Yes** — CI `security` job |
| Branch history | `gitleaks git . --log-opts="HEAD"` | Every commit reachable from the branch being built (527 on this branch) | **Yes** — CI `security` job |
| Repo-wide audit | `make security-secrets-audit` | **Every ref**, including unmerged and stale branches | No — run deliberately |

Locally: `make security-secrets` runs the two blocking scans. The gitleaks version is
pinned once, as `GITLEAKS_VERSION` in the `Makefile`; CI reads it via
`make print-gitleaks-version` rather than restating it.

## Why the PR gate is scoped to `HEAD`

Without `--log-opts`, gitleaks walks every ref in the clone. CI checks out with
`fetch-depth: 0`, so an unscoped history scan reports secrets committed to
*unmerged branches that the PR does not contain and cannot remediate* — a gate
that no PR can turn green, which trains people to ignore it.

A PR gate answers "does this branch introduce a secret?". Auditing the whole ref
graph answers a different question, and its findings are incidents requiring
credential rotation, not code review comments.

## Open finding — W&B API key in `feature/sprint6-validation-and-nplayer-foundation`

**Status: UNREMEDIATED. Requires credential rotation by the maintainer.**

The repo-wide audit reports 6 findings, all in commit
`87cd68a28e3a4d93f1190f82781e05fbf7323b98` (2026-03-19, "docs, scripts, infra:
Sprint 6 companion files and Phase 0 pre-PR prep"):

| File (at that commit) | Finding |
|---|---|
| `scripts/launch_via_docker.py:10` | `WANDB_API_KEY=` — 40-char lowercase hex, the real W&B key format |
| `scripts/run_benchmark_local.py:44` | same value |
| `scripts/deploy_orchestrator_vm.py:202` | same value |
| `scripts/vm_install.sh:35` | same value |
| `scripts/setup_vm_orchestrator.sh:66` | same value |
| `tests/unit/test_model_migration.py:42` | **false positive** — a 12-char conv-layer name in `assert key == "..."`, not a credential |

Verified facts:

- All five script findings are **one distinct value**, matching the W&B key format
  exactly (40 lowercase hex) with no placeholder markers.
- The commit is reachable **only** from `origin/feature/sprint6-validation-and-nplayer-foundation`.
  It is **not** on the default branch and **not** in this or any current PR's lineage.
- All six files are **deleted from the current tree**, which is why the tree scan is
  clean. Deletion does not remove the value from git history.

### Required action

1. **Rotate the W&B API key.** This is the only remediation that matters. Until it is
   rotated the value is live and extractable by anyone who can clone the repository —
   deleting the branch does not help, because the objects may persist and any existing
   clone already has them.
2. After rotation, decide whether to purge the objects (delete the stale branch, then
   `git gc`, or rewrite with `git filter-repo`) — this is cleanup, not remediation, and
   it invalidates existing clones.
3. Re-run `make security-secrets-audit` to confirm.

Weights & Biases was retired from this project on 2026-07-13 (charter NG-5), so the
key is no longer *used* — but an unrotated leaked credential is a live credential
regardless of whether this repo still calls the service.

## Adding an allowlist entry

`.gitleaks.toml` allowlists by **path** for build artifacts and by **value** for
deliberate fixtures. Prefer value-scoped entries: a path allowlist hides a real
secret added to that same file later.

Never add a catch-all (`.*`) — `tests/regression/test_repo_hygiene_regression.py`
fails if one appears, because it would make the gate report success while detecting
nothing.

## Gotcha: testing the config

A negative control built from a vendor's **documentation example** credentials will
not fire — gitleaks allowlists well-known example values upstream. The first probe of
this config used `AKIAIOSFODNN7EXAMPLE` and reported clean, which looked like a broken
config. Use randomly generated values when proving the scanner works.
