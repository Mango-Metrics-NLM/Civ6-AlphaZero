# Package the hex-crystalline encoder for standalone reuse and publication

## Why

The hex-crystalline spatial encoder is the repository's most externally reusable
artifact — hex-topology-aware convolution plus symmetry-group lattice transforms,
depending on nothing but `torch` and the standard library — yet it cannot be
reused, loaded, or evaluated by anyone outside this repository. Four defects were
measured on 2026-08-09:

1. **The encoder cannot be imported standalone.** Although
   `core/hex_conv.py`, `core/crystalline_layers.py`, `core/hex_encoder.py`, and
   `core/encoder_registry.py` import only torch and the stdlib, importing any of
   them executes `alpha_zero/__init__.py:32` → `alpha_zero/core/__init__.py:3` →
   `core/civ6_actions.py:20 import numpy`, and further down
   `envs/microciv.py` → `gymnasium`. Verified: `python -c "import
   alpha_zero.core.hex_conv"` raises `ModuleNotFoundError: No module named
   'numpy'` in an otherwise torch-only environment.

2. **There is no `from_pretrained()` path.** `huggingface_hub` appears in no
   dependency list, no network class exposes a hub-compatible
   `save_pretrained`/`from_pretrained` surface, and no checkpoint in the
   repository is written as safetensors — every writer emits a pickled
   `torch.save` payload, which `utils/checkpoint.py:126-132` itself warns about.

3. **There is no committed A/B evidence for the encoder.** `CHANGELOG.md:617-621`
   records an encoder A/B (`ab_patch_embed` vs `ab_hex_crystalline`) but no result
   numbers, and the two arms ran different container images (`v0.14.1` in
   `scripts/vm_deploy_ab.sh:8` vs `v0.14.2` in `scripts/vm_launch_run_b.sh:23`),
   so they were not code-identical. No `reports/` directory exists. Separately,
   the comparison tool cannot run: `scripts/compare_training_runs.py:33` imports
   `matplotlib` at module scope and matplotlib is declared in neither
   `pyproject.toml` nor `requirements.txt`, and the script reads only deprecated
   W&B runs or regex-parsed logs — its own docstring at `:9-11` requests an
   MLflow port, MLflow having been the sole tracker since v1.7.0. The local
   training entry points expose no `encoder_type`, and no head-to-head driver
   exists.

4. **The living documentation misdescribes the architecture.** `CLAUDE.md:381`
   and `CHANGELOG.md:559-566` claim axial coordinates, ring-distance positional
   encoding, multi-scale convolutions, and classes named `HexConv2d`,
   `CrystallineAttention`, and `EncoderRegistry`. The code implements even-q
   offset coordinates only (`hex_conv.py:72-73` raises for anything else), has no
   positional encoding of its own, uses a flat same-resolution convolution stack,
   and names those classes `HexConvolution`, `CrystallineLatticeLayer`, and a
   module-level `_ENCODER_REGISTRY` dict. `HexEncoderConfig.offset_type`'s
   docstring advertises an `"odd-q"` value that raises at runtime, and
   `encoder_registry.py:166` documents a `num_scales` kwarg that is silently
   discarded. No gate catches prose drift.

## What Changes

- Extract the four encoder modules plus `PatchEmbed` into a new torch-only
  `alpha_zero/hexcrystal/` subpackage, leaving explicit re-export shims at the
  existing `alpha_zero.core.*` paths so every current import, checkpoint, and
  `state_dict` key namespace keeps working unchanged.
- Convert `alpha_zero/__init__.py` to PEP 562 lazy attribute resolution so the
  root package no longer eagerly imports numpy and gymnasium, making the new
  subpackage importable in a torch-only environment.
- Add dict/JSON serialization to `HexEncoderConfig` and correct the drifted
  architecture documentation and docstrings.
- Add an opt-in `PyTorchModelHubMixin` subclass behind a new `hub` extra, giving
  the encoder `save_pretrained`/`from_pretrained` with safetensors weights and a
  flat `config.json`.
- Add a seeded, device-agnostic local A/B driver that trains both encoder types
  under an identical budget and runs a head-to-head arena, port the run
  comparator to read MLflow, and commit the resulting evidence.
- Publish the encoder to the Hugging Face Hub with an honest model card, via a
  dry-run-by-default upload script whose publish step is human-confirmed.

## Impact

- Affected capability: `hexcrystal-encoder-package` (new; ADDED: Standalone
  encoder subpackage, Backwards-compatible encoder import surface, Hub-publishable
  encoder artifact, Reproducible encoder A/B evidence, Human-gated model
  publication).
- Affected code: new `alpha_zero/hexcrystal/` (7 modules), shims and re-export
  alias in `alpha_zero/core/`, `alpha_zero/__init__.py`,
  `alpha_zero/utils/checkpoint.py`, `alpha_zero/cloud/gcp/selfplay_worker.py`
  (optional `seed` parameter), `scripts/run_encoder_ab.py` (new),
  `scripts/upload_encoder_to_hub.py` (new), `scripts/compare_training_runs.py`,
  `scripts/local_training_run.py`, `pyproject.toml`, and the documentation set.
- Backwards compatibility is a hard requirement of this change, not a
  side effect: no existing import path, checkpoint format, `state_dict` key, or
  CLI invocation may change behavior. The one deliberate surface change is that
  `alpha_zero.__all__` becomes static and no longer lists the optional
  `RealCiv6Env`/`RewardConfig` pair, which remain resolvable as attributes.
- No training-time behavior changes for existing runs: the encoder registry, the
  default `patch_embed` encoder, and all network construction paths are
  untouched apart from import locations.
