---
name: hub-publish-validator
description: >
  Pre-publish validation for Hugging Face Hub artefacts. Runs automatically
  before every upload_model_to_hub() call, and can be invoked standalone.
  Checks safetensors present, config.json parseable, README not a placeholder,
  no credentials in content, and repo_id is well-formed.
allowed-tools: Read, Bash
---

# Hub Publish Validator

## Purpose

Run this skill before any `upload_model_to_hub()` call or before
pushing a model/dataset to the Hugging Face Hub.

## Checks Performed

| ID | Check | Pass Condition |
| -- | ----- | -------------- |
| V1 | `model.safetensors` exists | File present in staged dir |
| V2 | `config.json` is valid JSON | `json.loads()` succeeds |
| V3 | `README.md` is not a placeholder | Length > 500 chars; no `$` template variables remaining |
| V4 | No credentials in artefacts | No `hf_` prefix strings; no `Bearer ` in file content |
| V5 | `repo_id` is well-formed | Matches `^[\w.-]+/[\w.-]+$` |
| V6 | `dataset_info.json` present (datasets only) | File exists if dataset export |

## Usage

```python
from alpha_zero.hub.upload import upload_model_to_hub
from alpha_zero.hub.model_card import ModelCardMetadata
from pathlib import Path
import json, re


def validate_staged_dir(staged_dir: Path, repo_id: str) -> list[str]:
    """Return list of validation failures; empty list = pass."""
    failures = []

    # V1 — weights file
    if not (staged_dir / "model.safetensors").exists():
        failures.append("V1: model.safetensors not found")

    # V2 — config parseable
    config_path = staged_dir / "config.json"
    if config_path.exists():
        try:
            json.loads(config_path.read_text())
        except json.JSONDecodeError as e:
            failures.append(f"V2: config.json invalid JSON: {e}")
    else:
        failures.append("V2: config.json not found")

    # V3 — README not a placeholder
    readme = staged_dir / "README.md"
    if readme.exists():
        text = readme.read_text()
        if len(text) < 500:
            failures.append("V3: README.md is too short (< 500 chars)")
        if re.search(r"\$\w+", text):
            failures.append("V3: README.md has unsubstituted template variables")
    else:
        failures.append("V3: README.md not found")

    # V4 — no credentials
    for path in staged_dir.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".md", ".txt"):
            content = path.read_text(errors="ignore")
            if "hf_" in content or "Bearer " in content:
                failures.append(f"V4: potential credential in {path.name}")

    # V5 — repo_id format
    if not re.match(r"^[\w.-]+/[\w.-]+$", repo_id):
        failures.append(f"V5: repo_id '{repo_id}' is not in org/repo format")

    return failures
```

## Integration with upload workflow

```bash
# Dry run + validate before real publish
python scripts/upload_to_hub.py --model-type encoder --output-dir /tmp/staged
python -c "
from pathlib import Path
from validate import validate_staged_dir
failures = validate_staged_dir(Path('/tmp/staged'), 'MyOrg/my-model')
if failures:
    print('VALIDATION FAILED:')
    for f in failures: print(' -', f)
    exit(1)
print('All checks passed — safe to publish')
"
python scripts/upload_to_hub.py --model-type encoder --output-dir /tmp/staged --publish
```

## Deterministic Validation

After running the validator, assert:
- Return value is an empty list
- All 4 artefact files present in staged dir
- No `$variable` patterns in README
- No `hf_` tokens in any JSON file
