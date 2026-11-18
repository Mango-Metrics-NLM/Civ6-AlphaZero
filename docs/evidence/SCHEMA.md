# Evidence artifact contract

Every directory under `docs/evidence/` that reports a measurement must contain:

1. exactly one report (`report.md`, `*_report.md` or `README.md`) that a reader can
   follow without the JSON;
2. at least one JSON file carrying an `evidence` block with the fields below;
3. a row in [`README.md`](README.md) (the index).

`tech_debt_baseline/` is exempt: it is a code-metric snapshot read by a gate, not a
measurement of the agent.

## The `evidence` block

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `git_sha` | 7–40 hex | commit the measurement was generated from |
| `generated_utc` | ISO-8601 | when it was generated |
| `search_epoch` | `pre-2026-09-04` \| `post-2026-09-04` | whether the search had the sign-correct backup, completed-Q target and per-iteration seeds (`e9d0572`). Pre-fix numbers are mechanism/wall-clock records, never strength evidence |
| `seed` | int \| null | base seed; if null, `seed_null_reason` is required |
| `command` | string | how to reproduce it |
| `hardware` | object | `cpu_model`, `gpu_model`, `logical_cores`, `omp_num_threads` (null + `hardware_null_reason` when unknown) |
| `python`, `torch`, `numpy` | string \| null | versions |
| `games_total` | int | games the measurement rests on |
| `claims` | list | what the artifact supports |
| `non_claims` | list | what it must not be quoted for |
| `backfilled` | string, optional | present when the block was added after generation |

The contract is enforced by `tests/regression/test_evidence_contract_regression.py`:
every directory has a report and an indexed row; every `pre-2026-09-04` report
carries the superseded banner; the `git_sha` of a `pre` artifact is an ancestor of
`e9d0572` where git history allows the check.

## Writing a new artifact

Produce the JSON with the block (the benchmark CLI does this), write the report,
add the index row, and label the epoch honestly. A result generated on
post-2026-09-04 code is the first kind of artifact that may be quoted as evidence
of strength, and only with its score-rate interval (see `docs/EVALUATION.md`).

## Release evidence bundle (`release_vX_Y_Z/bundle.json`)

A release's evidence directory (plan section 6, LF-D8) holds `bundle.json`,
`report.md`, an index row, and the components the bundle names. It is a normal
artifact under the contract above: `bundle.json` carries the `evidence` block.
`scripts/verify_release_evidence.py` checks it, and the `Release Evidence`
workflow (`.github/workflows/release-evidence.yml`, pull requests only) runs it
on every pull request into `main` except `hotfix/*` (exit 0 pass, 1 fail, 2
missing). `scripts/build_release_bundle.py` produces it from the run directories
`run_card` leaves (and runs the verifier on it before writing anything);
`tests/scripts/test_release_evidence_e2e.py` drives both on real producer output at tiny
scale. Generated example bundles live in `tests/scripts/release_evidence_fixture.py`.

| Field | Type | Meaning |
|---|---|---|
| `bundle_schema_version` | int | `1` |
| `release` | `X.Y.Z` | equals `pyproject.toml`; the directory is `release_vX_Y_Z` |
| `rc_tag`, `rc_sha` | tag, 40 hex | the **highest** rc of the release; the tag must point at `rc_sha`, and `rc_sha` must be an ancestor of the release head |
| `protocol_sha256` | 64 hex | sha256 of `docs/EVALUATION.md` section 7.1, at HEAD and at every attempt's commit |
| `runs` | list | every attempt (below): `A1`, and `A1-R` when it ran, VOID attempts included |
| `games_total` | int | sum of the attempts' `games`; reported, not gated |
| `learning_recorded` | `true` | required: a measured verdict (PASS, FAIL or INCONCLUSIVE) is recorded whatever it is |
| `learning_claimed` | bool | `true` only when the judged A1 and A1-R both recompute to PASS; the only thing that permits "learns", "learned" or "learnt" in `report.md`, the index row or the release's CHANGELOG section (a word inside backticks or quotes is a mention, not a claim) |
| `evidence` | object | the block above, with `games_total` equal to the bundle's |

Each `runs[]` entry is one attempt:

| Field | Meaning |
|---|---|
| `run_id`, `role`, `seed` | the card's `card_id` (unique; a re-run gets a new one); `A1` (seed 20261111) or `A1-R` (20261112) |
| `git_sha` | the card's pinned commit (`run.json`'s `sha`); every component must record it, and it must be an ancestor of `rc_sha`. For the judged attempt, nothing in `scripts/release_play_path.txt` may change from it to `rc_sha` or to the release head |
| `games`, `approval_ref` | the run's priced games (the sum of `run.json`'s invocations) and `run.json`'s own `approval_ref`; over 1,000 games needs one |
| `run_json`, `iterations` | `{path, sha256}` of the run's `run.json` and `iterations.jsonl` |
| `arms` | `{name: {path, sha256, wins, draws, losses, score_rate, confidence, hull}}`, one per registered arena invocation (`positive-control` -> `positive_control`); `hull` is at 99% for `symmetry`, 95% otherwise |
| `verdict` | `a1_protocol.compute_verdict(...).to_dict()`: `verdict`, `reasons`, the three hulls, and for INCONCLUSIVE `mde_alpha025_80pct`, `hull_ratio`, `mde_alpha025_80pct_hull`. `label` is present only on a non-reproducing re-run (below), where it is `"non-reproducing re-run"` |
| `reproduction` | only on an interrupted attempt that has a re-run: `{path, sha256, rerun, outcome, reasons}` naming its compare record (below), the `run_id` it was compared with, and what the comparison means for that re-run: `REPRODUCED`, `EMPTY` or `NON_REPRODUCING`. `reasons` are prose and are not compared |
| `checkpoints` | written by the builder: `{path under the run directory: sha256}` of every checkpoint an arm names (the files are not committed). The builder refuses an arm whose `checkpoint:<sha256>` is not the file on disk; the verifier ignores this field |

Paths are POSIX and relative to the bundle directory (`..` and absolute paths
are refused). For every attempt the verifier:

- requires `run.json`'s invocations to be, in order and token for token, the
  commands section 7.1 registers **at the attempt's commit**
  (`alpha_zero/evaluation/a1_registration.py`: the base block, the 40/60 branch in
  force, `<D>` for the `card_id`, the role's seed), launched by `run_card`
  (`launched_argv` = `argv` + `--max_games <priced>`), with `total_priced_games`
  equal to the registered card's price and each invocation's price its
  registered price (`run_card.price_invocation`);
- re-hashes every component and re-validates each arm's `result.json` with the
  benchmark's `validate_envelope`, ties its `evidence.command` to the launched
  invocation, and recomputes its counts, `scoring.summarize`, both intervals and
  their hull from the per-game records;
- recomputes the verdict; a stored number that disagrees fails the bundle.

**VOID records no learning result.** Each role present needs exactly one
non-VOID attempt, and that attempt is judged; a role with only VOID attempts
fails the bundle. After a VOID, fix the cause and re-run under a new `card_id`
at the same seed, and keep the VOID attempt listed (`docs/EVALUATION.md`
section 7.1, "After a VOID").

An attempt whose `run.json` status is not `COMPLETED` (VOID, FAILED, or a
lingering RUNNING) is VOID by status. It lists **whatever exists**: `run_json`
always; `iterations` if any iteration finished; an arm only if it wrote
`result.json`; an invocation or arm that never ran is omitted. It is checked for
honesty, not completeness:

- hashes, `card_id`, `sha`, seed and ancestry, as for any attempt;
- `run.json`'s invocations are a token-for-token prefix of the registered list;
- `total_priced_games` is the whole registered card's price (run_card writes it
  at start), and each recorded invocation's price is its registered price;
- every arm that `run.json` records as finished (exit 0) **must** be listed;
  besides those, only the next registered arm may be listed (it wrote
  `result.json` before run_card recorded it);
- each listed arm is recomputed;
- if `run.json` records training as finished, `iterations` must be listed and
  cover every iteration; each line carries the run's seed and sha;
- the stored verdict is `VOID`. A FAIL or INCONCLUSIVE verdict passes when it is
recorded honestly: a negative A1 ships the release with the claim withheld.

An attempt whose `run.json` records no launched invocation (`"launched": []`: a
refused device, a failed preflight) played nothing. It is not an attempt and may
not be listed. A `run.json` written before the `launched` key existed counts as
launched.

### Reproduction compare record (`<run_id>.reproduction.json`)

Section 7.1 ("After a VOID") makes the re-run after an interruption a
reproduction check first. Its record is the file written by

```text
python -m alpha_zero.evaluation.reproduction compare <VOID attempt's run dir> <re-run's run dir> --out FILE
```

and `scripts/build_release_bundle.py --compare FILE` copies it into the bundle
root as `<VOID attempt's run_id>.reproduction.json`, beside both attempts, and
names it in the VOID attempt's `reproduction` entry. The record names both runs,
so the builder matches it to them by `card_id`.

| Field | Meaning |
|---|---|
| `schema_version` | `1` (`reproduction.RECORD_SCHEMA_VERSION`) |
| `attempt`, `rerun` | each run's identity from its `run.json`: `card_id`, `sha`, `status`, and `run_json_sha256`, the sha256 of that file. `null` for a directory without a `run.json` |
| `digests` | `{attempt, rerun}`: what `reproduction.run_digest` gives for each run directory: every checkpoint's tensors (one sha256 each), every finished arm's protocol, seed and per-game records, and every iteration record without its wall-clock fields |
| `status`, `differences` | `REPRODUCED`, `DIFFERS` or `NOTHING TO COMPARE`, and each difference found. `NOTHING TO COMPARE` means the record's digest of the attempt holds nothing beyond `checkpoint_0.pt`, which the seed alone fixes. That is what the record shows, not something the bundle proves (see "What the record cannot show") |
| `components`, `rerun_components`, `components_compared`, `attempt_dir`, `rerun_dir`, `tool` | informational |

**Which attempts need one.** An attempt that is VOID for an interruption-class
reason (its `run.json` is not `COMPLETED`, an iteration record was resumed, an
`MCTS_*` variable was set, or its `device_plan` is not the registered one) and
that launched something. Its re-run is the first `COMPLETED` attempt of the same
role that started after it; that re-run need not be the attempt that is judged.
A symmetry or positive-control VOID needs none, even when it was also
interrupted: section 7.1 treats its re-run as a second draw, because the arena or
harness is fixed first. **Nothing checks that the code did change**: a
control-VOID attempt's re-run is taken as a second draw whatever its `sha`. A
`reproduction` entry on an attempt that needs none fails the bundle, as a stored
claim that nothing recomputes.

**Order.** Attempts are ordered by `run.json`'s `started_at` and by nothing else;
the order of `runs[]` decides nothing.

- Every listed attempt must record a `started_at` that parses as a timestamp.
- Two listed attempts of one role may not record the same instant. `run_card`
  writes `started_at` to the second and two runs of a role do not start in one,
  so an honest bundle has no tie; a tie would leave the order to the listing.
- **The role's non-VOID attempt is its last.** In a role that has a non-VOID
  attempt, every other listed attempt of that role must have started strictly
  before it: section 7.1 judges the one non-VOID attempt, and nothing of the role
  runs after it. This is one rule for two things. An interrupted attempt must be
  followed by its re-run, so it cannot be dated after the attempt that is judged
  (it would have nothing to be compared with). And a second draw cannot be dated
  at or before a non-reproducing re-run to escape the rule below.
- Two interrupted attempts before one re-run need a record each.

**What the verifier does with it** (`scripts/release_reproduction.py`). It
re-hashes the file, then requires that:

- both identities are the bundle's (`card_id`, `sha`, and `run_json_sha256` equal
  to the bundled `run.json`'s hash; a `null` never matches);
- the record's arena and iteration digests equal the digests of the bundled
  `result.json` and `iterations.jsonl` files of both runs, in both directions, so
  a record that leaves out a **bundled** component fails. That covers what the
  bundle must carry: every arm `run.json` records as finished, and the iteration
  records of an attempt whose training finished. It does not cover what a VOID
  attempt's bundle may omit: an arm that wrote `result.json` after the last
  invocation `run.json` records, and the iteration records of an attempt killed
  during training. A record can leave those out along with the bundle;
- `status` and `differences` are what `reproduction.compare_status` gives for the
  record's own digests.

It then recomputes the outcome, which must equal the stored `outcome` and `rerun`:

| Outcome | When | The re-run |
|---|---|---|
| `REPRODUCED` | no difference, the same conditions | is judged |
| `EMPTY` | the record's digests give `NOTHING TO COMPARE`, the same conditions | is judged; the bundle says the comparison was empty (this `outcome`, which is recomputed from the record, and a line the builder writes in `report.md`, which is not checked) |
| `NON_REPRODUCING` | any difference between the digests, **or** another `sha`, another `device_plan` (`CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER`, `CUBLAS_WORKSPACE_CONFIG`, `torch_pin`, `requested`, `expected_gpus`, and `torch.version`, `.cuda`, `.cudnn`, `.devices`, `.capabilities`), another torch wheel, or another NVIDIA driver on any GPU. Where section 7.1 registers devices, a driver that cannot be read from `preflight.nvidia_smi` on either side counts as another driver | cannot PASS: its verdict is recomputed with the label, and a PASS row is read as INCONCLUSIVE with the MDE that verdict reports |

The torch wheel is `run.json`'s `torch_wheel.sha256`, compared only when both
runs record one: pip does not always report it, and a run without the record is
not held to it.

**Nothing starts after a non-reproducing re-run.** Section 7.1 lets no further
A1-family run start until it is re-registered. Any listed attempt, of either
role, whose `started_at` is later than a non-reproducing re-run's fails the
bundle, with a message that a new registration is required first.

**What the record cannot show.** The record is the output of a tool the operator
ran, and the verifier can hold it only to what the bundle carries.

- The checkpoint tensor digests are not anchored to anything: checkpoints are
  too large to bundle, so the record is the only account of them.
- The arena and iteration digests are anchored only for the components that are
  bundled. What a VOID attempt's bundle may omit (an arm that wrote `result.json`
  after the last invocation `run.json` records, the iteration records of an
  attempt killed during training) a record can omit with it.
- So `EMPTY` means the record shows nothing beyond `checkpoint_0`; it does not
  prove the attempt finished nothing more. For an attempt killed during training,
  an `EMPTY` outcome is reachable by leaving its iteration records out of both
  the bundle and the record, and its later checkpoints out of the record.

**Not modelled.** After a non-reproducing re-run, section 7.1 requires a new
registration with a new confirmatory seed pair, whose bundle lists both
attempts. One half of that is checked: no listed attempt may have started after
the non-reproducing re-run (above). The other half is not: the verifier knows one
registration and one seed per role, so it cannot verify the bundle of the new
registration, and it says so in the message.

**Limits no check here can close.**

- *An attempt that is not listed.* A bundle verifier sees the runs it is given.
  An operator who leaves an attempt out of `runs[]` (an interrupted one, or a
  completed re-run that did not reproduce) leaves no trace in the bundle, and the
  rules above then never see it. The role's first `card_id` is not registered
  (section 7.1 names the card files, not their ids, and a card refused before it
  launched anything legitimately moves on to a new `card_id` without being
  listed), so a judged attempt under another id proves nothing.
- *`started_at` is the operator's own record.* It is written by `run_card` on
  the operator's machine into a file the operator holds. The order rules refuse
  the orderings they name (an unreadable or tied start, an attempt not before its
  role's non-VOID attempt, anything after a non-reproducing re-run). A forged
  clock that stays consistent with them is not detectable.

**Not checked.**

- The torch wheel comparison is skipped when either run has no `torch_wheel`
  record.
- A control-VOID attempt's second draw is not required to be at another `sha`,
  or to have a changed play path.
- A `started_at` is not checked against anything outside the bundle, such as the
  commit date of the run's `sha`.

**Conservative readings.** Where section 7.1 is silent or looser, the verifier
takes the stricter reading. These are the maintainer's to overrule.

- *A VOID with no re-run that can PASS.* An attempt that is VOID because its
  `device_plan` departs from the registered one, or because an `MCTS_*` variable
  was set, is treated as interruption-class: it needs a record. Its correct
  re-run necessarily runs under other conditions (the registered devices, no
  override), so it is a non-reproducing re-run, it cannot PASS, and no further
  A1-family run may start. The same holds for a `FAILED` run whose fix needs a
  code change: the re-run is at another `sha`. Section 7.1 speaks only of an
  interruption and of "a `FAILED` run whose cause needs no code change"; it is
  silent on the first two cases.
- *The cutoff after a non-reproducing re-run is its `started_at`*, not when it
  finished. For the other role that is stricter than section 7.1 if the two runs
  overlapped: a run that started while the re-run was still running is refused,
  though nothing was known yet. On the registered device plan both cards use
  both GPUs, so they do not overlap.
