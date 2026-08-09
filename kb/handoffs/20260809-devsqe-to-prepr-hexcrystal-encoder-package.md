# DEV-SQE → Pre-PR Handoff

**Date:** 2026-08-09
**Spec:** `openspec/changes/hexcrystal-encoder-package/proposal.md`
**Branch:** `claude/grid-crystal-mapping-search-q9wwne`
**Session:** session_011aLW3xW6HK5PeY8q4YFtZS

## What shipped

The hex-crystalline encoder is now a standalone, publishable artifact. Three
deliverables, seven commits, each phase individually revertible.

1. **`alpha_zero/hexcrystal/`** — the encoder extracted into a torch-only
   subpackage. It was always torch-only by source, but importing it executed the
   root package's eager chain into numpy and gymnasium, so it could not be reused
   outside this repository.
2. **Hub support** — an opt-in `PyTorchModelHubMixin` subclass giving
   `save_pretrained`/`from_pretrained` with safetensors weights and a flat
   `config.json`.
3. **A/B evidence** — a seeded, device-agnostic local driver that trains both
   encoder arms from identical code and budget and plays them head to head, with
   a Wilson interval and a consistency gate on the output.

### Status of the evidence run

Both arms trained to completion on this machine (6 iterations × 6 games, 25 MCTS
simulations, board 8, max 30 turns, seed 42, CPU): `patch_embed` 494,137
parameters, `hex_crystalline` 553,663. Checkpoints and sidecars are in
`checkpoints/encoder_ab/` (uncommitted — `*.pt` is gitignored by design).

The 150-game arena was **still running when this session ended**, so
`docs/evidence/hexcrystal_ab/` is not yet populated and no numbers are claimed
anywhere in this PR. Finish it with:

```bash
python scripts/run_encoder_ab.py arena --num_games 150 --mcts-sims 25 --max-turns 30
python scripts/run_encoder_ab.py report
```

Both subcommands were smoke-verified end to end earlier in the session on a
4-game arena, including the consistency gate and the report renderer. Note the
arena is the expensive half: measured ~20 s/game on 4 vCPU, so 150 games is
roughly an hour on an idle machine and considerably longer under contention.

At 150 games the Wilson interval is roughly ±8 points, so unless the result is
lopsided the honest conclusion will be "this run does not distinguish the two
encoders" — which the report emits automatically. If a resolvable comparison is
wanted, 800 games is the point where a 5-point edge becomes detectable.

## Files changed

**New:** `alpha_zero/hexcrystal/{__init__,hex_conv,crystalline_layers,hex_encoder,encoder_registry,patch_embed,hub_mixin}.py`, `alpha_zero/hexcrystal/Agent.md`, `scripts/run_encoder_ab.py`, `scripts/upload_encoder_to_hub.py`, `docs/model_cards/hexcrystal_encoder.md`, `openspec/changes/hexcrystal-encoder-package/`.

**Modified:** `alpha_zero/__init__.py` (PEP 562 lazy resolution), the four
`alpha_zero/core/*` encoder modules (now re-export shims),
`alpha_zero/core/transformer_network.py` (PatchEmbed moved out, re-export alias),
`alpha_zero/core/Agent.md`, `alpha_zero/utils/checkpoint.py`
(`validate_encoder_state`), `alpha_zero/cloud/gcp/selfplay_worker.py` (optional
`seed`), `scripts/compare_training_runs.py` (MLflow source, guarded matplotlib),
`scripts/local_training_run.py` (`--encoder_type`), `pyproject.toml`,
`tests/sanity/test_import_sanity.py`, `docs/MODULE_REFERENCE.md`, `CLAUDE.md`,
`CHANGELOG.md`, `docs/README.md`.

## Tests added

`tests/unit/`: `test_hexcrystal_backcompat.py`, `test_hexcrystal_state_dict_keys.py`,
`test_hexcrystal_isolation.py`, `test_root_lazy_api.py`,
`test_hexcrystal_config_serialization.py`, `test_hexcrystal_encoder_roundtrip.py`,
`test_hexcrystal_hub_mixin.py`, `test_encoder_ab_driver.py`,
`test_validate_encoder_state.py`, `test_selfplay_seed_threading.py`,
`test_compare_runs_mlflow.py`, `test_upload_encoder_to_hub.py`.

## Verification

| Gate | Result |
|---|---|
| `pytest tests/unit tests/sanity -m "not slow"` | 10,399 passed, 49 skipped, **2 timeouts under CPU contention** (see below) |
| `pytest tests/integration tests/cloud/unit tests/security` | 1,569 passed, 13 skipped, 1 xfailed |
| Local-only encoder suites (smoke, regression, e2e) | 86 passed |
| `ruff check .` / `ruff format --check .` | clean |
| `mypy alpha_zero/` | 101 errors before, 100 after — **none introduced** |
| `validate_openspec` / `validate_kb` / `validate_skills` / `check_doc_sync` | all exit 0 |
| `tests/governance` | 50 passed |
| Coverage | **NOT verified locally** — numpy 2.4 + coverage C-tracer crash (CLAUDE.md §5.3). CI is the gate. Do not claim green. |
| CI on PR #107 | 5/7 green at handoff: Governance Gates, LLM Eval Harness, Security Scan, **Lint & Type Check**, Cloud Unit Tests. Test (3.11) and Test (3.12) still running. |

The CI Lint & Type Check pass is the useful datapoint on mypy: that job runs
`mypy alpha_zero/` as a **blocking** gate with the project's pinned 1.19.1 and
the stubs this sandbox lacks, and it went green. The 101 local errors are
environmental.

### The two failures are timeouts, not regressions

`test_enhanced_training_tracking.py::test_training_runs_without_wandb` and
`test_gumbel_mcts.py::test_performance_better_than_random` failed with
`Failed: Timeout (>600.0s) from pytest-timeout` while the A/B arena was
saturating the CPU. Evidence they are not regressions:

- An earlier full unit run, before the arena started, was **10,277 passed / 0
  failed** on the same code path.
- `test_performance_better_than_random` contains **no assertions** — its
  docstring says "just checking it doesn't crash" — so it can only fail by
  crashing or timing out.
- Neither test touches the encoder, the subpackage, the hub mixin, or the A/B
  driver.
- The wandb test takes ~300s standalone on an idle machine; under contention it
  exceeded even the 600s ceiling.

Pre-PR should re-run both on an idle machine to close this out.

## Known issues / carried forward

- Coverage remains unverifiable locally (numpy 2.4 C-tracer); CI is the gate.
- `mypy alpha_zero/` has 101 pre-existing errors in this environment. CI pins the
  same mypy (1.19.1) but installs stubs this sandbox lacks. Not addressed here —
  out of scope and far beyond the file budget.
- The A/B evidence is reduced-scale and single-seed. The report and model card
  both state this explicitly and refuse to claim significance the interval does
  not support.
- **Publication has not happened.** The upload tool is dry-run by default and
  `--publish` is a deliberate human step, per CLAUDE.md §3.2 (external API calls
  with side effects). Nothing has been pushed to the Hub.

## Notes for Pre-PR

- The riskiest change (root-package lazification) is isolated in commit `f11ed07`
  and reverts cleanly on its own; the extraction in `3cac75b` survives without it.
- `openspec/changes/hexcrystal-encoder-package/` must be linked in the PR's
  "Linked OpenSpec change" field. Archiving is a separate post-merge PR.
- A deliberate public-API change needs calling out in review:
  `alpha_zero.__all__` no longer lists `RealCiv6Env`/`RewardConfig`. Both remain
  resolvable attributes; only `import *` stops pulling them in. The rationale is
  in `alpha_zero/__init__.py` and the OpenSpec design document.
