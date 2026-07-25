---
name: release
description: Cut a release — bump the version, roll the changelog, and tag vX.Y.Z so the release workflow fires. Use when the user asks to cut, tag, or publish a release. Gated by scripts/check_doc_sync.py and the presence of a tag at HEAD.
allowed-tools: Read, Write, Edit, Bash
---

# Release Skill

Cut a versioned release. `.github/workflows/release.yml` builds the GitHub
release from the pushed tag, so the deliverable of this skill is a correct,
consistent `vX.Y.Z` tag.

## Steps

### 1. Choose the version

Follow semantic versioning. Confirm the target `X.Y.Z` with the user if unclear.

### 2. Bump the version

Set the version in `pyproject.toml` (`[project] version`) and
`alpha_zero/__init__.py` (`__version__`) to the same `X.Y.Z`.

### 3. Roll the changelog

In `CHANGELOG.md`, rename the top `## [Unreleased]` section to
`## [X.Y.Z] - <date>` and open a fresh empty `## [Unreleased]` above it.

### 4. Verify consistency

```bash
python scripts/check_doc_sync.py
```

D1 (version triple) and D7 (changelog head == version) must pass.

### 5. Commit, tag, and push

```bash
git commit -am "release: vX.Y.Z"
git tag vX.Y.Z
git push origin vX.Y.Z
```

Pushing the tag triggers the release workflow.

## Final gate

The release is not complete until:

```bash
python scripts/check_doc_sync.py    # exits 0
git tag --points-at HEAD            # prints the new vX.Y.Z tag
```

both succeed.
