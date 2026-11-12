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
missing). Generated example bundles live in
`tests/scripts/release_evidence_fixture.py`.

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
| `verdict` | `a1_protocol.compute_verdict(...).to_dict()`: `verdict`, `reasons`, the three hulls, and for INCONCLUSIVE `mde_alpha025_80pct`, `hull_ratio`, `mde_alpha025_80pct_hull` |

Paths are POSIX and relative to the bundle directory (`..` and absolute paths
are refused). For every attempt the verifier:

- requires `run.json`'s invocations to be, in order and token for token, the
  commands section 7.1 registers **at the attempt's commit**
  (`alpha_zero/evaluation/a1_registration.py`: the base block, the 40/60 branch in
  force, `<D>` for the `card_id`, the role's seed), launched by `run_card`
  (`launched_argv` = `argv` + `--max_games <priced>`);
- re-hashes every component and re-validates each arm's `result.json` with the
  benchmark's `validate_envelope`, ties its `evidence.command` to the launched
  invocation, and recomputes its counts, `scoring.summarize`, both intervals and
  their hull from the per-game records;
- recomputes the verdict; a stored number that disagrees fails the bundle.

**VOID records no learning result.** Each role present needs exactly one
non-VOID attempt, and that attempt is judged; a role with only VOID attempts
fails the bundle. After a VOID, fix the cause and re-run under a new `card_id`
at the same seed, and keep the VOID attempt listed (`docs/EVALUATION.md`
section 7.1, "After a VOID"). A FAIL or INCONCLUSIVE verdict passes when it is
recorded honestly: a negative A1 ships the release with the claim withheld.
