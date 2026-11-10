# evaluation-protocol — deltas

## ADDED Requirements

### Requirement: Score rate is the statistic for claims

Any statement about agent strength in documentation or evidence SHALL use the
score rate (wins plus half the draws, over games) with its ternary confidence
interval from ``alpha_zero/evaluation/scoring.py``. A draw-blind win rate MAY be
logged for compatibility but MUST be accompanied by ``score_rate`` and
``draw_rate`` wherever it is logged, and new draw-blind computations MUST NOT be
added outside the allowlist in ``tests/regression/draw_aware_allowlist.toml``.

#### Scenario: A new draw-blind win rate is introduced

- **WHEN** a module under ``alpha_zero/`` or ``scripts/`` divides a ``wins`` quantity without pairing it with a score rate
- **THEN** ``pytest tests/regression/test_draw_aware_metrics_regression.py`` fails naming the file and function

#### Scenario: A win rate is logged alone

- **WHEN** a metrics dictionary passed to a logger contains a ``win_rate`` key without ``score_rate`` and ``draw_rate``
- **THEN** the draw-aware regression test fails naming the call site

### Requirement: Evidence artifacts carry provenance and an epoch label

Every directory under ``docs/evidence/`` that reports a measurement SHALL contain
one ``report.md`` and at least one JSON file whose ``evidence`` block records the
git SHA, generation time, seed (or a reason it is null), command, hardware,
library versions, game count and a ``search_epoch`` of ``pre-2026-09-04`` or
``post-2026-09-04``, and the directory MUST be listed in ``docs/evidence/README.md``.

#### Scenario: An artifact without provenance is committed

- **WHEN** an evidence directory lacks a JSON ``evidence`` block with a ``git_sha``
- **THEN** ``pytest tests/regression/test_evidence_contract_regression.py`` fails naming the directory

#### Scenario: A pre-fix artifact is quoted as current

- **WHEN** a report in a ``pre-2026-09-04`` directory lacks the superseded banner
- **THEN** the evidence-contract regression test fails naming the report

### Requirement: Benchmarks run through the promotion judge's code path

The benchmark protocol SHALL be executed by the same arena function the
promotion gate uses, with a seeded search and paired seats, and its output
SHALL embed the versioned evaluation record plus ``draw_rate``,
``decisive_win_rate`` and the minimum detectable effect at the game count used.

#### Scenario: The protocol is reproducible

- **WHEN** the benchmark is run twice with the same seed and protocol
- **THEN** the two result files are identical except for the generation timestamp

#### Scenario: The protocol can detect a known difference

- **WHEN** a random-rollout search with 64 simulations plays a random agent for 40 paired games
- **THEN** the lower bound of the score-rate interval exceeds 0.5

### Requirement: Promotion feasibility is reported

Before an evaluation that feeds a promotion decision, the system SHALL compute
and log the minimum observed score the configured policy can promote at the
configured game count and expected draw rate, and the Elo that score implies,
and it MUST warn when that implied Elo exceeds a configurable ceiling.

#### Scenario: The deployed sample size cannot support the threshold

- **WHEN** the policy is evaluated at 20 games with an expected draw rate of 0.7
- **THEN** the feasibility report states an implied Elo above 90 and a warning is logged
