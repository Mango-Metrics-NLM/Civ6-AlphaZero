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
