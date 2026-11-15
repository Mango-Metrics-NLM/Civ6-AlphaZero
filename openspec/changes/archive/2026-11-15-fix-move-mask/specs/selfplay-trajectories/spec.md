# selfplay-trajectories -- deltas

## ADDED Requirements

### Requirement: The move mask agrees with the executor

The MicroCiv action mask SHALL mark a move legal only if the executor would
accept it: the target is on the board, is not a mountain, and holds no unit of
any player. A move the mask marks legal MUST NOT be rejected by the executor,
because masked policy loss and search treat the mask as ground truth and a
rejected move forfeits the unit's remaining moves.

#### Scenario: Seeded random play over legal actions is never rejected

- **WHEN** 200 seeded games are played choosing only mask-legal actions
- **THEN** every step reports `action_success` true and no "Invalid move" occurs

#### Scenario: A unit blocks a move onto its tile

- **WHEN** a mover has an own or enemy unit on the adjacent target tile
- **THEN** the move action toward that tile is masked out
