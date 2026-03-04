# Epic 5: N-Player MicroCiv Foundation — PRD

## User Story

**As a** reinforcement learning researcher  
**I want** `MicroCivEnv` to support N players (2-6) via a `num_players` parameter  
**So that** I can train agents for multiplayer Civ6 scenarios beyond 1v1

## Background

`MicroCivEnv` currently hardcodes 2 players throughout:

- Turn cycling: `1 - self.current_player`
- Victory check: explicitly checks P0 and P1
- Starting positions: only 2 spawn points
- Info dict: hardcodes indices 0 and 1

The `MicroCivEncoder` uses `own`/`enemy` perspective encoding and **does not need changes** for N-player support — it already divides the world into "self" and "everyone else."

## Acceptance Criteria

### AC1: Backwards Compatibility  

**Given** `MicroCivEnv(board_size=8)` (no `num_players` arg)  
**When** the game runs  
**Then** it behaves identically to the current 2-player version

### AC2: N-Player Construction  

**Given** `MicroCivEnv(board_size=8, num_players=4)`  
**When** the game resets  
**Then** 4 sets of starting units are placed and `get_info()` lists 4 players

### AC3: N-Player Turn Cycling  

**Given** a 4-player game  
**When** each player ends their turn  
**Then** `current_player` cycles 0→1→2→3→0 and turn increments after player 3

### AC4: N-Player Victory  

**Given** a 4-player game  
**When** all but one player is eliminated  
**Then** the surviving player wins

### AC5: N-Player Truncation Tiebreak  

**Given** a 4-player game reaching max_turns  
**When** the game truncates  
**Then** the player with the most cities wins (None for tie)

## Out of Scope

- Encoder changes (own/enemy encoding already handles N players)
- Network architecture changes (2→N policy heads, etc.)
- Alliance/diplomacy mechanics

## Success Metrics

- All existing 2-player tests still pass (regression-free)
- New N-player tests pass for 3, 4, and 6 players
- ≥80% test coverage on the modified methods
