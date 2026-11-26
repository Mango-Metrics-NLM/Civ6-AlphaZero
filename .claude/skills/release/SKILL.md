---
name: release
description: Prepare a release PR -- bump the version in pyproject.toml, alpha_zero/__init__.py and CITATION.cff, roll the changelog, and hand the tag to the maintainer. Use when the user asks to cut or prepare a release. Agents never tag. Gated by scripts/check_doc_sync.py.
allowed-tools: Read, Write, Edit, Bash
---

# Release Skill

Prepare a versioned release. The deliverable of this skill is a **release PR**
whose version triple, `CITATION.cff` and changelog agree. It does **not** tag:
creating a `v*` tag is a maintainer-only step (the `release-tags` ruleset, once the
maintainer has applied it; until then a convention), and the PreToolUse guard denies
`git tag` and tag pushes for agents. Never run `git tag` or push a tag from this skill.

The branch flow (feature branch -> `dev` -> `qa` -> `main`) is in
`docs/runbooks/branching-and-release.md`.

## Steps

### 1. Choose the version

Follow semantic versioning. Confirm the target `X.Y.Z` with the user if unclear.
Release candidates are tagged `vX.Y.Z-rc.N` against the same `X.Y.Z` in the files
below; there is no separate "rc" version in the files.

### 2. Branch

Work on a short-lived branch cut from `dev` (for example `chore/release-vX.Y.Z`). Never
commit to `dev`, `qa` or `main`.

### 3. Bump the version in three places

Set the same `X.Y.Z` in:

- `pyproject.toml` (`[project] version`)
- `alpha_zero/__init__.py` (`__version__`)
- `CITATION.cff` (`version:`, and `date-released:` to the release date)

### 4. Roll the changelog

In `CHANGELOG.md`, rename the top `## [Unreleased]` section to
`## [X.Y.Z] - <date>` and open a fresh empty `## [Unreleased]` above it.
The changelog and `date-released` are final at the first rc: the final-tag rule
forbids later edits outside `docs/evidence/`. Write any claim about validation
verdict-neutrally (for example "verdict in `docs/evidence/release_v2_1_0/report.md`"),
because the evidence does not exist yet when this PR merges.

### 5. Verify consistency

```bash
python scripts/check_doc_sync.py
```

D1 (version triple and `CITATION.cff`) and D7 (changelog head == version) must pass.

### 6. Commit and open the PR into dev

```bash
git commit -am "chore(release): prepare vX.Y.Z"
```

Push the branch and open a PR into `dev` using the release PR type in the template.
Stop here and tell the maintainer.

### 7. Maintainer steps (not for an agent)

After the PR merges into `dev`, the maintainer promotes `dev` -> `qa` (merge commit)
and runs, from their own terminal:

```bash
python -m scripts.maintainer.tag_release vX.Y.Z-rc.N            # dry run
python -m scripts.maintainer.tag_release vX.Y.Z-rc.N --execute  # tag + push
```

and later the same for the final `vX.Y.Z` after `qa` -> `main`. `tag_release` checks
every release gate before tagging and refuses inside a Claude Code session.

## Final gate

The agent's part is complete when:

```bash
python scripts/check_doc_sync.py    # exits 0
git status                          # clean, on the release branch, PR open against dev
```

both hold. There is deliberately no check for a tag at HEAD: an agent that
creates one has gone outside this skill.
