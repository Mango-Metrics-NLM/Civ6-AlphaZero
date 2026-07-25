# Tasks: add-enterprise-docs-and-gates

Sizing key: **S** ≤ 1 file / mechanical · **M** a few files · **L** judgment work.
Each phase is one or more commits, each ≤ 20 files, each green under the gates
active at that point.

## 1. Phase A — OpenSpec bootstrap + CI-trigger fix

- [x] 1.1 (M) ``openspec/project.md`` + ``openspec/AGENTS.md``
- [x] 1.2 (M) Baseline specs: ``agent-harness``, ``kb-workflow``
- [x] 1.3 (M) Change dir: ``proposal.md``, ``tasks.md``, ``design.md``
- [x] 1.4 (M) Four delta specs: ``documentation``, ``validation-gates``, ``agent-harness``, ``kb-workflow``
- [x] 1.5 (L) ``scripts/validate_openspec.py`` + ``tests/governance/test_validate_openspec.py`` + ``pytest.ini``
- [x] 1.6 (S) ``Makefile``: ``validate`` + ``test`` alias
- [x] 1.7 (S) ``ci.yml``: ``claude/**`` trigger, ``concurrency`` group, ``governance`` job
- [x] 1.8 (S) Push; open draft PR (#93); subscribe to PR activity

## 2. Phase B — Validators, harness hooks, git hooks

- [x] 2.1 (L) ``validate_kb.py``, ``validate_skills.py``, ``check_doc_sync.py`` + governance tests
- [x] 2.2 (M) ``.claude/settings.json``: deny ``.env`` edits; repoint hooks at scripts; governance hook
- [x] 2.3 (M) Hook scripts: guard force-push, ``posttooluse_lint``, ``precompact_state``, ``posttooluse_governance``, ``sessionstart_bootstrap``
- [x] 2.4 (M) ``.pre-commit-config.yaml``: ruff 0.8.0, scoped hooks, local validators
- [x] 2.5 (S) ``Makefile``: ``hooks-install``
- [x] 2.6 (S) ``ci.yml``: governance += kb; ``eval-harness`` job (continue-on-error)
- [x] 2.7 (S) ``kb/README.md`` + ``kb/summaries/pre-pr.md`` seed

## 3. Phase C — Skills

- [ ] 3.1 (M) Retrofit ``planning/dev-sqe/pre-pr`` handoff skills (frontmatter, 85%, gate)
- [ ] 3.2 (M) New skills: ``openspec-propose``, ``openspec-archive``, ``doc-sync``, ``release``
- [ ] 3.3 (S) Wire skills route into hook, pre-commit, CI, Makefile

## 4. Phase D — README + core accuracy

- [ ] 4.1 (L) README rewrite (~400 lines)
- [ ] 4.2 (L) ``docs/README.md`` index + architecture consolidation + stubs
- [ ] 4.3 (M) ``MODULE_REFERENCE.md``, ``CHANGELOG.md`` repair, ``.env.template``
- [ ] 4.4 (S) ``alpha_zero/__init__.py`` → 1.8.0; ``setup.py`` shim
- [ ] 4.5 (M) ``NEXT_STEPS.md`` + ``GCP_INFRA.md`` consolidation; ``CLAUDE.md`` §11.3
- [ ] 4.6 (S) Enable ``check_doc_sync.py`` in CI + pre-commit

## 5. Phase E — Enterprise files + Agent.md + archive sweep

- [ ] 5.1 (M) E1: LICENSE, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, SUPPORT, CODEOWNERS, PR/issue templates, dependabot, 3 Agent.md, NAMING rebrand
- [ ] 5.2 (M) E2: ``prd/README.md``, ``adr/`` index + ADR renames, runbook rename
- [ ] 5.3 (L) E3: inbound-link grep; convert ISSUE_*.md to real issues; archive sweep + ``docs/archive/README.md`` + index relink

## 6. Phase F — Close-out

- [ ] 6.1 (M) ``CLAUDE.md`` updates (§4.1, §5.2, §5.3, §7, §14.1, §15.3)
- [ ] 6.2 (S) ``CHANGELOG.md``: log this change under ``[Unreleased]``
- [ ] 6.3 (S) ``ci.yml``: flip ``eval-harness`` to enforcing if green
- [ ] 6.4 (S) Final tasks.md check-off; run handoff skills
