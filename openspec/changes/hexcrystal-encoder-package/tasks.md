# Tasks — hexcrystal-encoder-package

## 1. Extraction and import-chain break

- [ ] 1.1 (M) Create `alpha_zero/hexcrystal/` with `hex_conv.py`,
      `crystalline_layers.py`, `hex_encoder.py`, `encoder_registry.py`,
      `patch_embed.py`, and a torch-only `__init__.py`; move `PatchEmbed` out of
      `core/transformer_network.py` and remove the registry's back-edge to it
- [ ] 1.2 (S) Leave explicit named re-export shims (each with `__all__`) at the
      four historical `alpha_zero/core/*` paths and a `PatchEmbed` alias in
      `transformer_network.py`; repoint the single production call site
- [ ] 1.3 (S) Update `docs/MODULE_REFERENCE.md` (new `hexcrystal` row, submodule
      count 27 → 28), add `alpha_zero/hexcrystal/Agent.md`, refresh the stale
      rows in `alpha_zero/core/Agent.md` — same commit as the move
- [ ] 1.4 (M) Convert `alpha_zero/__init__.py` to PEP 562 lazy resolution; keep
      the `_compat` import and the `__version__` literal eager; make `__all__`
      static and resolve the optional Civ6 environment names via `__getattr__`
- [ ] 1.5 (S) Tests: back-compat identity, `state_dict` key snapshot (captured
      pre-move: patch_embed 41 keys / 231,225 params; hex_crystalline 65 keys /
      290,751 params), dependency isolation via subprocess, lazy root API

## 2. Configuration serialization and documentation accuracy

- [ ] 2.1 (S) Add `to_dict`/`from_dict`/`to_json`/`from_json` to
      `HexEncoderConfig`; unknown keys are logged and ignored
- [ ] 2.2 (S) Correct the `offset_type` docstring and remove the phantom
      `num_scales` kwarg documentation
- [ ] 2.3 (S) Rewrite the architecture description in `CLAUDE.md` §5.2 and fix
      the stale submodule count in §7; record the correction in the changelog
- [ ] 2.4 (S) Tests: config round-trip, symmetry-type serialization, standalone
      encoder `state_dict` round-trip, adjacency buffer device movement,
      unsupported offset type

## 3. Hugging Face Hub integration

- [ ] 3.1 (S) Add a `hub` extra and mirror its two dependencies into `dev` so the
      round-trip is exercised and measured by CI; add to the `all` aggregate
- [ ] 3.2 (M) Add the opt-in hub mixin module with flat primitive configuration
      capture; register it in the import-sanity suite's optional-module set
- [ ] 3.3 (S) Tests: offline save/load round-trip, artifact manifest, and the
      dependency-absent failure mode

## 4. A/B driver, arena, and comparison tooling

- [ ] 4.1 (M) Add `scripts/run_encoder_ab.py` with `train`, `arena`, and `report`
      subcommands composing the existing encoder-aware, storage-free workers
- [ ] 4.2 (S) Thread an optional seed through self-play environment construction
      so game streams are reproducible; default preserves current behavior
- [ ] 4.3 (S) Add encoder-identity validation to the checkpoint utilities
- [ ] 4.4 (M) Give the run comparator an MLflow source, guard its plotting import,
      and parameterize its hardcoded threshold and project defaults
- [ ] 4.5 (S) Pass the encoder type through the local training entry point
- [ ] 4.6 (S) Tests: checkpoint payload contract, encoder validation, MLflow
      source with a mocked client, seed threading

## 5. Evidence run

- [ ] 5.1 (S) Smoke the driver end to end at minimal budget
- [ ] 5.2 (L) Train both arms at the pinned budget and run the head-to-head arena
- [ ] 5.3 (S) Commit the report, arena result, and summary table with sample size
      and confidence interval, in an artifact-only commit

## 6. Publication

- [ ] 6.1 (M) Write the model card describing the implemented architecture, with
      the evaluation-scope caveat
- [ ] 6.2 (M) Add the dry-run-by-default upload tool; publication itself is a
      human-confirmed step taken after review
- [ ] 6.3 (S) Tests: dry-run default, missing-credential failure, wrong-encoder
      rejection

## 7. Verification and close-out

- [ ] 7.1 (S) Run the full gated suite plus the three local-only encoder suites
- [ ] 7.2 (S) Lint, type check, and all four governance validators
- [ ] 7.3 (S) Update the changelog and the session knowledge base; open the draft
      pull request
