# muzero-search -- deltas

## ADDED Requirements

### Requirement: MuZero's latent search is two-player sign-correct

`MuZeroMCTS` SHALL track the player to move on every node it creates, using a
turn model (`EndTurnModel` for MicroCiv: END_TURN passes the move). Its backup,
its root Q and its in-tree selection MUST score an edge as
`perspective_sign(parent.to_play, child.to_play) * (reward + discount * value)`,
where the edge reward is in the frame of the player to move after the
transition. Both the Gumbel path and the PUCT path (`use_gumbel=False`) SHALL
use this backup.

#### Scenario: The opponent makes the final move

- **WHEN** MuZero searches the alternating forced-win game, at a depth where
  the root player's win arrives through an opponent's transition
- **THEN** the returned policy prefers the winning action on both search paths

#### Scenario: The mover does not change

- **WHEN** MuZero searches the same-player forced-win game
- **THEN** it prefers the winning action at every depth, and replacing
  `perspective_sign` with plain negamax makes it fail

#### Scenario: The turn model agrees with the environment

- **WHEN** seeded random MicroCiv games are played
- **THEN** `EndTurnModel` predicts `current_player` after every step

### Requirement: MuZero training targets share the search's frames

Every MuZero self-play path SHALL label the value of a position with the game
outcome for the player to move there. It SHALL label the reward of every
transition as zero, except the final one, which carries the outcome for the
player to move after it. The absorbing state after the game MUST be worth zero.

#### Scenario: A decided game

- **WHEN** player 1 wins a game whose final transition leaves player 1 to move
- **THEN** player 1's positions have value `+1`, player 0's have `-1`, and the
  final reward target is `+1` with every other reward target `0`
