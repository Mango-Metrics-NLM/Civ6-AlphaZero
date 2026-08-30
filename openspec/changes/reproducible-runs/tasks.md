# Tasks

## 1. Seedable search
- [x] 1.1 Add `seed` to `MCTSConfig`, `GumbelMCTSConfig`, `MuZeroMCTSConfig`, `BatchMCTSConfig`
- [x] 1.2 Give each variant `self._rng`; replace all eight global draws
- [x] 1.3 Thread an `rng` parameter through `gumbel_sample_actions`' three callers
- [x] 1.4 Give `SimpleMCTS` a `seed` parameter and one generator instead of two
- [x] 1.5 Carry the RNG through `MicroCivEnv.clone()` (found by a failing verification, not by reading)

## 2. Seedable training
- [x] 2.1 `--seed` on `training_civ6.py`; `seed` on `TrainingConfig`
- [x] 2.2 Thread to self-play envs (`seed + game_idx`), search, and evaluation streams
- [x] 2.3 Seed the replay buffer sampler and the DataLoader shuffle
- [x] 2.4 Seed torch before network construction
- [x] 2.5 Verify by diffing two `--seed 42` dry runs (this is what found 2.3)

## 3. Seedable evaluation
- [x] 3.1 `--seed` on `run_baseline_eval.py` and `post_run_evaluation.py`
- [x] 3.2 Give `NeuralNetworkPlayer` a seed and its own generator
- [x] 3.3 Confirm two probes at one seed agree and a different seed differs

## 4. The gate
- [x] 4.1 Behavioural determinism tests against the real `train()` entry point
- [x] 4.2 Static check that no search module draws from the global RNG
- [x] 4.3 Run every negative control; record which fire and which cannot
- [x] 4.4 Re-measure the five checkpoint probes at a fixed seed and commit the evidence
      (`docs/evidence/seeded_probe_reproducibility/`: seeded 5/5 identical at 0.325;
      unseeded spans 0.30-0.45, reproducing the original phenomenon as a control)

## 5. Documentation
- [x] 5.1 CHANGELOG, CLAUDE.md decision row, known-issue update
- [ ] 5.2 Archive this change after merge (separate PR)
