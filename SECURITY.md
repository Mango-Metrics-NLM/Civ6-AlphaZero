# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.8.x   | ✅ |
| < 1.8   | ❌ |

## Reporting a vulnerability

Please report security issues **privately** through GitHub Security Advisories
("Report a vulnerability" on the repository's Security tab) rather than opening a
public issue. Include reproduction steps and affected versions. We aim to
acknowledge reports within a few business days.

## Handling of secrets

- Never commit secrets. API keys and credentials are loaded from the environment
  or a local `.env` file (gitignored); `.env.template` documents the variables.
- The harness denies edits to `.env` files, and `bandit` runs in CI on
  `alpha_zero/`.
- Cloud credentials are provisioned out of band (see `scripts/setup_secrets.sh`);
  none are stored in the repository.
