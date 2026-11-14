# Tasks

## 1. Pinned environment and driver
- [x] 1.1 `TorchPin` (exact `torch==` pin, plain https index); pre-install before the wheel; venv keyed `<sha12>-t<hash>`
- [x] 1.2 Card keys `torch_spec` / `torch_index_url`; `device_problem`; `--probe`; device plan in `run.json`
- [x] 1.3 Negative controls: no device check, pin not passed, no pre-install, venv keyed on SHA alone -- each fails a test

## 2. Determinism and evidence
- [x] 2.1 `enforce_deterministic_backend`, `cuda_ordinal`; wired into the trainer, the benchmark and the calibration
- [x] 2.2 Calibration refuses an unavailable explicit GPU and records the resolved device; `gpu_model` names the device used
- [x] 2.3 Negative controls: each call removed, the refusal replaced by `pass`, `gpu_model` pinned to GPU 0, the helper setting nothing -- each fails a test
- [x] 2.4 `tests/regression/test_cuda_determinism_regression.py` (skipped without CUDA); helpers checked on CPU

## 3. Registration
- [x] 3.1 Section 7.1 commands, "Devices" paragraph, "After a VOID" wording, registration-log row, `REGISTRATION_DIGEST`
- [x] 3.2 `a1.yaml`, `a1r.yaml`, `t0b.yaml` (612 games); tests and the release-evidence fixture follow the registration
- [x] 3.3 Runbook, plan spec notes, CLAUDE.md, CHANGELOG

## 4. On the GPU machine (maintainer)
- [ ] 4.1 `pytest tests/regression/test_cuda_determinism_regression.py -v` passes with the pinned torch build
- [ ] 4.2 `python scripts/run_card.py configs/run_cards/t0b.yaml --probe` reports PROBED
- [ ] 4.3 T0b runs; `a1_protocol calibrate` output and the three throughput files go into the slot fill-in PR
