---
name: hub-release
description: >
  End-to-end Hugging Face Hub release skill for AlphaZero models and datasets.
  Orchestrates: checkpoint extraction → staging → validation → Hub upload →
  git tag. Designed for deterministic, repeatable releases from CI or local
  developer machines. Safe by default: dry-run unless --publish is passed.
---

# Hub Release Skill

## Purpose

Automate the full publish → tag cycle for any registered model type:
`encoder`, `transformer`, or `muzero`. Also supports dataset export (`elo`,
`replay`).

## Pre-conditions (required before running)

1. `HF_TOKEN` env var set (never pass as a CLI flag)
2. Hub extra installed: `pip install -e '.[hub]'`
3. Checkpoint file available (or default config used for fresh models)
4. `git status` is clean (no uncommitted changes)

## Step-by-Step Workflow

### 1. Validate environment
```bash
echo $HF_TOKEN | head -c 4  # Should print 'hf_x' — never log the full token
python -c "import huggingface_hub; print('hub extra ok')"
git status --short  # Should be empty
```

### 2. Dry-run (always first)
```bash
python scripts/upload_to_hub.py \
    --model-type encoder \
    --checkpoint checkpoints/hex_crystalline.pt \
    --output-dir ./release-staging

# Review the printed manifest and README
cat ./release-staging/README.md
python -c "
from pathlib import Path; import json
cfg = json.loads(Path('./release-staging/config.json').read_text())
print('Config:', json.dumps(cfg, indent=2))
"
```

### 3. Validate staged artefacts
```python
# Use the hub-publish-validator skill
from pathlib import Path, re, json

staged = Path("./release-staging")
failures = []
if not (staged / "model.safetensors").exists():
    failures.append("missing weights")
if not (staged / "config.json").exists():
    failures.append("missing config")
if len((staged / "README.md").read_text()) < 500:
    failures.append("placeholder README")

assert not failures, f"Pre-publish validation failed: {failures}"
print("Validation passed — ready to publish")
```

### 4. Publish
```bash
HF_TOKEN=... python scripts/upload_to_hub.py \
    --model-type encoder \
    --checkpoint checkpoints/hex_crystalline.pt \
    --output-dir ./release-staging \
    --publish
```

### 5. Git tag
```bash
VERSION=$(python -c "import alpha_zero; print(alpha_zero.__version__)")
git tag -a "hub-release-$VERSION" -m "Hub release $VERSION"
git push origin "hub-release-$VERSION"
```

### 6. Dataset release (optional)
```bash
python scripts/export_dataset.py \
    --type elo \
    --source-path ./results \
    --output-dir ./elo-staging \
    --publish
```

## Deterministic Validation

After a complete release, verify:

```python
from huggingface_hub import HfApi

api = HfApi()
info = api.repo_info("Mango-Metrics-NLM/civ6-encoder")
assert info is not None
assert info.private == False  # or True if private
print("Release verified at:", info.id)
```

```bash
git tag | grep hub-release  # Should show the new tag
```

## Rollback

```bash
# Delete the Hub version (requires write access)
huggingface-cli repo delete Mango-Metrics-NLM/civ6-encoder --yes

# Delete the git tag
git tag -d hub-release-X.Y.Z
git push origin :refs/tags/hub-release-X.Y.Z
```
