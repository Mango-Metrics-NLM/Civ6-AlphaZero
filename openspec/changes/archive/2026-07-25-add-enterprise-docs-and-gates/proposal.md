# Proposal: add-enterprise-docs-and-gates

## Why

The repository's documentation has drifted severely from the application it
describes, and nothing prevents the drift from recurring:

- **README (1,495 lines)** still teaches the retired Weights & Biases tracker
  (MLflow became the sole tracker on 2026-07-13), references a deleted module
  (``alpha_zero/utils/wandb_logger.py``) and a nonexistent ``requirements-dev.txt``,
  shows an MIT badge with **no ``LICENSE`` file**, carries two contradictory preset
  tables and two different test counts, and pins a frozen "ACTIVE" run status.
- **Version drift:** ``pyproject.toml`` says ``1.8.0`` while
  ``alpha_zero/__init__.py`` says ``0.14.0`` and ``setup.py`` says ``0.2.0`` with
  phantom ``rag``/``rag-full`` extras.
- **``docs/`` (43 files, no index):** the C4 architecture is duplicated three ways
  plus ~770 lines inlined in the README; ``docs/MODULE_REFERENCE.md`` (the declared
  single source of truth) is a version behind, states an 80% gate (actual 85%), and
  has three drifted module counts.
- **Missing enterprise files:** ``LICENSE``, ``CONTRIBUTING.md``, ``SECURITY.md``,
  ``CODE_OF_CONDUCT.md``, ``SUPPORT.md``, ``CODEOWNERS``, PR/issue templates,
  ``dependabot.yml``, a docs index, and a documented release process (``release.yml``
  is a complete tag-driven pipeline that has never fired — zero tags exist).
- **Harness gaps:** the deny list omits ``.env`` edits and force-push though
  ``CLAUDE.md`` §15.3 claims both are denied; skills lack YAML frontmatter; the
  pre-commit config is not installed and pins a ruff version that disagrees with CI;
  and there is **no deterministic validation** of the harness, the KB ledger, or the
  documentation at all.

## What Changes

- **OpenSpec adoption:** introduce ``openspec/`` (this change) with a stdlib
  validator as the binding gate.
- **Deterministic validators:** ``validate_openspec.py``, ``validate_kb.py``,
  ``validate_skills.py``, ``check_doc_sync.py`` — wired into Claude Code hooks,
  ``.pre-commit-config.yaml``, and a new CI ``governance`` job.
- **Harness + hooks:** fix the deny list and the fragile inline hooks; add a
  governance validation hook; install the git hooks with CI-aligned pins.
- **Skills:** retrofit the three handoff skills with frontmatter and a validation
  gate; add ``openspec-propose``, ``openspec-archive``, ``doc-sync``, ``release``.
- **Documentation overhaul:** rewrite the README to match the application;
  consolidate the architecture docs; add the missing enterprise files; reorganize
  ``docs/`` with an index and archive the stale process artifacts.

## Impact

- **Affected capabilities:** ``agent-harness`` (modified), ``kb-workflow`` (modified),
  ``documentation`` (new), ``validation-gates`` (new).
- **Affected files:** README, ``docs/**``, ``.claude/**``, ``scripts/**``,
  ``.github/**``, ``Makefile``, ``kb/**``, ``CLAUDE.md``, ``pyproject`` adjacencies
  (``alpha_zero/__init__.py`` version string, ``setup.py`` shim). No ``alpha_zero/``
  runtime logic changes.
- **Non-goals:** no docs-site generator; no glossary; no repo-wide lint/format of
  ``scripts/`` or root runners (pre-existing debt, tracked separately).
