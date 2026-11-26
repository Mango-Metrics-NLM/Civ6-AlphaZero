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
- [x] 4.7 Negative controls recorded in each test module; debt ratchet passes without a baseline change (until 4b.2)
- [x] 4.8 Follow-up: the release verifier reads the `reproduction compare` record and applies "cannot PASS" itself (done in 4c.5)
- [x] 4.9 Follow-up: record the torch wheel's sha256 in `.pinned.json` and `run.json` (done in 4c.2)
- [x] 4.10 Follow-up: a preflight refusal does not use up the `card_id` (done in 4c.1 by rehearsing first; nothing is moved aside)

## 4b. Automated review of PR #138 (Copilot)
- [x] 4b.1 A torch pin may name its build (`torch==2.11.0+cu128`); the device check, the release verifier and the GPU pytest module compare the installed version in full through `utils.device.torch_build_matches`; section 7.1 and the three cards name the build (digest `e460a353…7c17`)
- [x] 4b.2 Arena per-game records carry `moves` and `actions_sha256`; the reproduction and determinism checks compare them. `coverage_omitted_lines` baseline 66,852 -> 66,858 (six lines in `alpha_zero/cloud/gcp/arena.py`, covered by `tests/cloud/unit/test_arena_game_records.py` in the `cloud-tests` job)
- [x] 4b.3 Negative controls: tag ignored (9 tests fail), no action digest (1), records without the new keys (3), tag rejected by the pin (9)
- [x] 4b.4 The other three findings (parsed device, test state isolation, CUDA variables in the run-card test) were already fixed in section 4

## 4c. Local close-out after PR #138 and #139 (2026-11-15)
- [x] 4c.1 `run_card` rehearses before it claims the run directory; a used directory is refused before the rehearsal; `run.json` records `rehearsal` and `launched`
- [x] 4c.2 The torch wheel pip installed is recorded (`pip install --report`) in the venv marker and `run.json`; record-only, outside the compared torch pin
- [x] 4c.3 Checkpoints and the benchmark's `result.json` are written through a temporary file (`alpha_zero/utils/atomic_io.py`)
- [x] 4c.4 A card that pins torch and names a CUDA device is refused when the NVIDIA driver version cannot be read
- [x] 4c.5 `reproduction compare` records identities and digests; `build_release_bundle.py --compare`; the verifier recomputes the comparison, requires a record for an interrupted attempt and reads a non-reproducing PASS as INCONCLUSIVE
- [x] 4c.6 Negative controls recorded in each test module; no edit inside the hashed region (digest unchanged)
- [ ] 4c.7 Follow-up: the verifier does not model the bundle section 7.1 requires after a non-reproducing re-run (a new registration that lists both attempts)
- [x] 4c.8 Review fix round (run cards): the run-directory claim is atomic (the second of two racing runs is refused as a resumed card); a report listing torch twice records no wheel; a reused environment without a wheel record warns; tests for a failure after a passed rehearsal and for an interrupted rehearsal
- [x] 4c.9 Review fix round (verifier): a role's attempts are ordered by `started_at` alone and every listed attempt must record one; an interrupted attempt must be followed by its re-run; no attempt may start after a non-reproducing re-run; the torch wheel hash is compared when both runs record it; a null identity in a record never matches
- [x] 4c.10 Second review round (verifier): a role's non-VOID attempt is the last of its role to start and no two attempts of a role tie, so a second draw cannot be dated at or before a non-reproducing re-run and the listing order never picks the re-run; `docs/evidence/SCHEMA.md` states the conservative readings and the limits in one place
- [ ] 4c.11 Follow-up (not checked today, stated in `docs/evidence/SCHEMA.md`): the torch wheel comparison is skipped when either run has no record; a control-VOID attempt's second draw is not required to be at another commit; a `started_at` is not checked against the commit date of its `sha`

## 5. On the GPU machine (maintainer)
- [ ] 5.1 `python scripts/run_card.py configs/run_cards/t0b.yaml --probe` reports PROBED and names the two registered GPUs in order
- [ ] 5.2 T0b runs; its `determinism` invocation reports REPRODUCED
- [ ] 5.3 `a1_protocol calibrate` output, `determinism.json`, `run.json` and the three throughput files go into the slot fill-in PR
