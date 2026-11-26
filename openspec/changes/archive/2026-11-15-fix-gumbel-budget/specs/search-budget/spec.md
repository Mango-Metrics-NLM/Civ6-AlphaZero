# search-budget -- deltas

## ADDED Requirements

### Requirement: A Gumbel search spends exactly its configured budget

Every sequential-halving Gumbel search (`GumbelMCTS`, `BatchGumbelMCTS` and
`MuZeroMCTS`) SHALL follow the single schedule in `gumbel_ops.halving_schedule`.
It MUST spend exactly the configured `n` simulations whenever two or more
actions are legal, and none for a forced move. Each search SHALL report what it
measured in `last_search_stats`, and `sim_evals` MUST equal
`planned_halving_sims(k, n, num_legal)`.

#### Scenario: The registered budget is spent exactly

- **WHEN** a search runs with `k = 8`, `n = 24` and ample legal moves
- **THEN** `last_search_stats.sim_evals` is 24 for all three searches

#### Scenario: More candidates requested than legal moves

- **WHEN** a search runs with `k = 16`, `n = 64` and 5 legal moves
- **THEN** it spends 64 simulations, where the old floor spent 40

#### Scenario: A lone candidate is still searched

- **WHEN** a search runs with `k = 1`, `n = 24` and two or more legal moves
- **THEN** all 24 simulations go to the one candidate

#### Scenario: A forced move is not searched

- **WHEN** exactly one action is legal
- **THEN** the search spends no simulations

### Requirement: The returned move was searched

A Gumbel search MUST return a move from among the candidates with the maximum
root visit count, so that a truncated round can never return a move the search
did not visit.

#### Scenario: Truncated first round

- **WHEN** a search runs at `k = 16`, `n = 8` with a one-hot target, a root value of +0.9 and leaf values of -0.9
- **THEN** the returned move is one of the 8 candidates the search visited, for every seed
