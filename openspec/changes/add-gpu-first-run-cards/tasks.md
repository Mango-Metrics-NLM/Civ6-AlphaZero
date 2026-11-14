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
- [x] 3.2 `a1.yaml`, `a1r.yaml`, `t0b.yaml` (612 games at this step; 660 after 4.5); tests and the release-evidence fixture follow the registration
- [x] 3.3 Runbook, plan spec notes, CLAUDE.md, CHANGELOG

## 4. Review round (QA, architecture, ML science)
- [x] 4.1 `alpha_zero/local/devices.py` and `card.py` split out of `run_card.py` (877 lines); `cuda_ordinal` in `utils/device.py`; one `hardware_summary`
- [x] 4.2 Devices read through the runner's own parser; a pinned card names every device; preflight records architectures, cuDNN, driver and one real CUDA op per device; cards name the GPU expected at each ordinal
- [x] 4.3 Section 7.1 `a1-devices` block; cards held equal to it; the release verifier voids a run on other devices and rejects a `--probe` record
- [x] 4.4 "After a VOID": a re-run is a reproduction check and a differing one cannot PASS; `evaluation/reproduction.py` (`compare`)
- [x] 4.5 `scripts/check_determinism.py` as runner `determinism_check`, T0b's first invocation (660 games)
- [x] 4.6 `a1_protocol digest`, `--arena-throughput`, `make verify-registration`, skill `a1-reregister`
- [x] 4.7 Negative controls recorded in each test module; debt ratchet passes without a baseline change
- [ ] 4.8 Follow-up: the release verifier reads the `reproduction compare` record and applies "cannot PASS" itself
- [ ] 4.9 Follow-up: record the torch wheel's sha256 in `.pinned.json` and `run.json`
- [ ] 4.10 Follow-up: move a never-launched run directory aside so a preflight refusal does not use up the `card_id`

## 5. On the GPU machine (maintainer)
- [ ] 5.1 `python scripts/run_card.py configs/run_cards/t0b.yaml --probe` reports PROBED and names the two registered GPUs in order
- [ ] 5.2 T0b runs; its `determinism` invocation reports REPRODUCED
- [ ] 5.3 `a1_protocol calibrate` output, `determinism.json`, `run.json` and the three throughput files go into the slot fill-in PR
