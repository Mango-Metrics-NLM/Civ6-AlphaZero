# Tasks

## 1. Test first
- [x] 1.1 `tests/unit/test_checkpoint_provenance.py`; RED on the old code (import error)
- [x] 1.2 Round trip under `weights_only=True`; mismatch raises on resume, `load_network` and the benchmark loader; legacy warns
- [x] 1.3 `--bootstrap none` constructs no SimpleMCTS; `checkpoint_0` resume safety

## 2. Implement
- [x] 2.1 `save(metadata=)`, `build_provenance`, `check_provenance`, `HorizonMismatch`
- [x] 2.2 `--bootstrap`, `checkpoint_0.pt`, provenance on every save
- [x] 2.3 Negative controls: disabling the `max_turns` comparison, and the resume guard on `checkpoint_0`, each fail tests

## 3. Documentation
- [x] 3.1 CLAUDE.md §5.2 row
