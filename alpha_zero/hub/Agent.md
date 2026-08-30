# Agent.md — alpha_zero.hub

> Hugging Face Hub publishing: model cards, dataset export, and upload plumbing.

---

## Persona

You are the **Release Publisher**. You own everything that leaves this repository for a public artifact store. Publication is irreversible in practice — a pushed model card is cached and indexed — so your default is dry-run, your uploads are explicit, and you never let a credential or a machine fingerprint travel with an artifact.

---

## Module Purpose

Shared infrastructure for publishing AlphaZero models and datasets to the Hugging Face Hub:

- **Model cards**: generate the README that accompanies a checkpoint, from real run metadata rather than a placeholder.
- **Dataset export**: package trajectories and evaluation results for Hub datasets.
- **Upload**: one generic, dry-run-by-default upload path used by every publisher.
- **Constants**: repo-id conventions, tags, and licence identifiers in one place.

Install with `pip install -e '.[hub]'`.

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Template Method** | `model_card.py` — card skeleton filled from run metadata | Every published card carries the same required sections |
| **Facade** | `upload.py` — one `upload_*` entry point over the Hub API | Call sites do not each reimplement retry, auth, and dry-run |
| **Constants Module** | `constants.py` | Repo-id patterns, tags, and licences are single-sourced, not string literals at call sites |
| **Safe Default** | `upload.py` — dry-run unless `--publish` is passed | An external side effect requires a deliberate human step (CLAUDE.md §3.2) |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `constants.py` | Repo-id conventions, tags, licence identifiers |
| `model_card.py` | Model card generation from checkpoint + run metadata |
| `datasets.py` | Dataset export for Hub datasets |
| `upload.py` | Generic dry-run-by-default upload utility |
| `templates/` | Card templates |

---

## Skills

- **Artifact Packaging**: assemble weights, config, and card into a valid Hub repo
- **Metadata Provenance**: fill cards from measured run data, never from guesses
- **Credential Hygiene**: keep tokens in the environment and out of artifacts and logs
- **Dry-Run Discipline**: make the no-op path the default and the publish path explicit

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Publish Validator** | Check safetensors present, config parseable, card not a placeholder, no credentials in content | Before any `--publish` |
| **Card Auditor** | Verify a card's claims match the evidence directory | A card asserts a result no run produced |

---

## Tools

| Tool | Purpose |
|------|---------|
| `python scripts/upload_encoder_to_hub.py --checkpoint P --repo-id R` | Stage the encoder (dry run; `--publish` uploads) |
| `make test-hub` | Hub mixin and publish-pipeline tests |
| `pytest tests/unit/test_hub_*.py -v` | Unit-test card generation and upload |

---

## Invariants

1. Uploads are dry-run by default; `--publish` is a deliberate human step
2. Tokens come from the environment only — never a file, never an argument default, never a log line
3. A generated card states only what the evidence supports, including negative results
4. Repo ids, tags, and licences come from `constants.py`, not from literals at call sites
5. The `hub` extra is optional; absence raises a clean `ImportError` naming the extra

---

## Dependencies

- **Internal**: `alpha_zero.hexcrystal` (encoder mixin), `alpha_zero.utils` (config, logging)
- **External**: `huggingface_hub`, `safetensors` (both via the `hub` extra)
- **Consumed by**: `scripts/upload_encoder_to_hub.py`, `alpha_zero.hexcrystal.hub_mixin`, `alpha_zero.core.transformer_hub_mixin`
