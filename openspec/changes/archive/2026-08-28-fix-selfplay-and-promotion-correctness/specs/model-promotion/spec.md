# Model Promotion — deltas for fix-selfplay-and-promotion-correctness

## ADDED Requirements

### Requirement: Every promotion decision site uses the policy registry

Each component that decides whether to promote a candidate model SHALL obtain
that decision from the registered promotion policy rather than computing a
comparison inline, so the decision rule is single, testable and swappable.
Components that are not part of the deployed path SHALL say so in their own
documentation, so a reader can tell which decision actually ships.

#### Scenario: The default policy reproduces the historical gate

- **WHEN** a head-to-head record is scored with no promotion policy configured
- **THEN** the statistic equals wins divided by total games, and the promotion
  outcome matches the historical comparison exactly

#### Scenario: A draw-aware policy can be opted into

- **WHEN** a draw-aware promotion policy is selected by configuration and a
  draw-heavy record is scored
- **THEN** the decision follows that policy's statistic, and a candidate that
  the legacy statistic could never promote can be promoted

#### Scenario: A component's own threshold is honoured

- **WHEN** a component configures a promotion threshold explicitly
- **THEN** that threshold governs the decision, and is not replaced by the
  registry's default

### Requirement: Early stopping measures what the final gate measures

An evaluation that stops early SHALL base that decision on the same statistic the
final promotion gate will use. Stopping on a different statistic can end a run
that the final gate would have promoted.

#### Scenario: Early stopping follows the configured policy

- **WHEN** a record is scored differently by the legacy and draw-aware policies,
  and evaluation is run under each in turn
- **THEN** the early-stop outcome follows the configured policy in each case

### Requirement: Promotion results record draw-aware shadow statistics

A persisted evaluation result SHALL carry the draw-aware score rate and its
confidence bounds alongside the legacy win rate, so a policy change can be
assessed against historical evaluations before it is adopted. The full decision
SHALL be recorded, not only a promoted/not-promoted boolean, because an
inconclusive result is not a rejection.

#### Scenario: Shadow statistics are persisted

- **WHEN** an evaluation completes
- **THEN** the saved result carries the score rate alongside the legacy win rate,
  and confidence bounds are present whenever the configured policy computes them

#### Scenario: An inconclusive verdict is distinguishable

- **WHEN** a policy returns an inconclusive verdict
- **THEN** the reported outcome distinguishes it from a decisive rejection

### Requirement: Promotion configuration is validated before games are played

Promotion configuration SHALL be rejected at validation time when it names a
confidence level the scoring implementation cannot evaluate, rather than raising
during evaluation after games have been played.

#### Scenario: An unsupported confidence level is rejected up front

- **WHEN** a promotion configuration names a confidence level outside the
  supported set
- **THEN** validation fails and names the supported levels
