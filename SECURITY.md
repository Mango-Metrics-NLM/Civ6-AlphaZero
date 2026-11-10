# Security Policy

## Scope

This policy covers what is in this repository:

- the `alpha_zero` Python package (training, search, evaluation, storage, cloud
  workers) and the `scripts/` that drive it;
- the web interface (`alpha_zero/web/`, FastAPI + WebSocket) and the inference
  server (`alpha_zero/serving/`, `alpha_zero/inference/`);
- the Civilization VI Lua mod and bridge (`mod/`, `alpha_zero/bridge/`);
- the agent harness (`.claude/`, `scripts/validate_*.py`, the governance gates).

**Out of scope:** the maintainer's own Google Cloud project, buckets and
deployments. They are referenced by configuration in this tree but are not a
public service; findings about them should still be reported privately and will
be handled, but they are not vulnerabilities *in this software*.

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.8.x   | ✅ |
| < 1.8   | ❌ |

Rows change at release time: the public-release train rolls the version to the
next major, and this table is updated in the same change. Only the latest
release line receives fixes.

## Reporting a vulnerability

Please report security issues **privately** through GitHub Security Advisories
("Report a vulnerability" on the repository's Security tab) rather than opening a
public issue. Include reproduction steps and affected versions. The project has a
single maintainer; we aim to acknowledge reports within a few business days and
to publish a fix or a mitigation before any public disclosure.

Security items are never tracked as public issues, and pull requests that fix
them are described without the exploit.

## Handling of secrets

- Never commit secrets. API keys and credentials are loaded from the environment
  or a local `.env` file (gitignored); `.env.template` documents the variables.
- The harness denies `Edit`/`Write` to `.env` files, `bandit` runs in CI on
  `alpha_zero/`, and `gitleaks` scans the tracked tree and this branch's commit
  history (`make security-secrets`; see
  [`docs/runbooks/secret-scanning.md`](docs/runbooks/secret-scanning.md)).
- Cloud credentials are provisioned out of band with
  [`scripts/setup_secrets.sh`](scripts/setup_secrets.sh); none are stored in the
  repository.

## Known open items, by category

Kept here so a reader is not surprised by them; details that would help an
attacker are deliberately omitted.

- **Credential rotation.** A tracking-service API key that once lived on an
  unmerged branch was found by the repository-wide history audit. It is not on
  the default branch and the service has been retired, but a leaked key is only
  remediated by rotation, which is in the maintainer's hands. Triage and status:
  [`docs/runbooks/secret-scanning.md`](docs/runbooks/secret-scanning.md).
- **Harness guard bypasses.** The agent's PreToolUse guard is a heuristic, not a
  sandbox; shell wrappers defeat it. Documented with the threat model in
  [`.claude/README.md`](.claude/README.md) and the 2026-09-04 audit.
- **Permission allowlist breadth.** The harness pre-approves `python -c` and
  trusts all project MCP servers; narrowing both is an open maintainer decision.
- **Supply chain.** Workflow actions are SHA-pinned and gated by a regression
  test; dependency, code and workflow scanners are being added as non-blocking
  jobs during the public-release train.
