# SpacetimeDB Server Module -- Design Document

## 1. Architecture Overview

```
 Web Clients (TypeScript SDK)         Python AI Service
 ================================     =========================
 | subscribe game_state         |     | subscribe ai_turn_request |
 | subscribe move_history       |     | call submit_move          |
 | call create_game             |     | call register_ai_player   |
 | call submit_move             |     |                           |
 | call spectate_game           |     |                           |
 ================================     =========================
              |                                 |
              v                                 v
 +---------------------------------------------------------+
 |                 SpacetimeDB Module (Rust)                |
 |                                                         |
 |  Tables:                                                |
 |    player, game, game_state, move_history,              |
 |    game_state_snapshot, ai_turn_request, spectator,     |
 |    matchmaking_queue, tournament, tournament_bracket,   |
 |    tournament_registration, chat_message                |
 |                                                         |
 |  Reducers:                                              |
 |    register_player, register_ai_player, create_game,    |
 |    join_game, submit_move, request_ai_turn, forfeit,    |
 |    spectate_game, stop_spectating, join_matchmaking,    |
 |    leave_matchmaking, send_chat, create_tournament,     |
 |    register_for_tournament, start_tournament            |
 |                                                         |
 |  Internal:                                              |
 |    initialize_game_state, apply_action, compute_*,      |
 |    update_elo, try_matchmake, advance_bracket           |
 +---------------------------------------------------------+
              |
              v
       SpacetimeDB Storage
       (WASM execution + B-tree indexes)
```

## 2. Schema Design

### 2.1 Entity-Relationship Diagram

```
player (1) ----< (N) game         (as player0 or player1)
player (1) ----< (N) spectator
player (1) ----< (N) matchmaking_queue
player (1) ----< (N) tournament_registration

game (1) -----> (1) game_state     (current board snapshot)
game (1) ----< (N) move_history    (append-only move log)
game (1) ----< (N) game_state_snapshot  (periodic snapshots)
game (1) ----< (N) ai_turn_request (pending AI actions)
game (1) ----< (N) spectator
game (1) ----< (N) chat_message

tournament (1) ----< (N) tournament_registration
tournament (1) ----< (N) tournament_bracket
tournament_bracket (1) ----> (0..1) game
```

### 2.2 Table Sizing Estimates

| Table | Row Size | Active Rows | Growth Rate | Notes |
|-------|----------|-------------|-------------|-------|
| player | ~200 B | 1K-100K | slow | One per registered user/AI |
| game | ~150 B | 100-10K active | ~100/day | Completed games stay for history |
| game_state | ~4 KB | = active games | overwrites | One row per active game |
| move_history | ~1.7 KB | 30 per game avg | ~3K/day | With MCTS policy blobs |
| game_state_snapshot | ~4 KB | 4 per game avg | ~400/day | Every 10 moves + final |
| ai_turn_request | ~3.6 KB | 0-50 | transient | Deleted after AI responds |
| spectator | ~40 B | 0-500 | transient | Active spectators only |
| matchmaking_queue | ~40 B | 0-100 | transient | Cleared on match |
| tournament | ~200 B | 1-50 | rare | Per tournament |
| tournament_bracket | ~80 B | log2(N) * N/2 | per round | Single-elimination |
| chat_message | ~300 B | variable | ~1K/day | Could add TTL cleanup |

**Storage for 1000 completed games:**
- move_history: 1000 * 30 moves * 1.7 KB = ~50 MB
- game_state_snapshot: 1000 * 4 * 4 KB = ~16 MB
- Total: ~70 MB (very manageable)

### 2.3 Index Strategy

SpacetimeDB automatically creates B-tree indexes on `#[primary_key]` and
`#[unique]` columns. For query patterns we need beyond those:

| Table | Indexed Columns | Query Pattern |
|-------|----------------|---------------|
| player | identity (unique) | Lookup by caller identity in every reducer |
| player | player_id (PK) | FK joins from game, spectator, etc. |
| game | game_id (PK) | Direct lookup by ID |
| game_state | game_id (PK) | 1:1 with game, direct lookup |
| move_history | move_id (PK, auto_inc) | Sequential access |
| ai_turn_request | request_id (PK) | Subscription-driven (no manual queries) |

**Note:** SpacetimeDB does not currently support secondary indexes beyond
PK/unique. For queries like "find all games where player_id = X" or
"find all moves for game_id = Y", the module iterates with `.filter()`.
This is acceptable because:
1. Active game counts are small (< 10K).
2. Move history is only queried for replay (not latency-critical).
3. SpacetimeDB WASM execution is fast for small table scans.

If table sizes grow beyond 100K rows, consider:
- Archiving completed games to a separate "game_archive" table.
- Partitioning move_history by game_id ranges.
- Adding SpacetimeDB secondary indexes when the feature ships.

## 3. Serialization Format

### 3.1 Board State (GameState)

The board state is stored as compact binary blobs inside the `game_state` row.

**Terrain** (64 bytes for 8x8):
```
terrain[y * 8 + x] = TerrainType as u8
  0 = Plains
  1 = Hills
  2 = Mountain
```

**Units** (8 bytes per unit, variable count):
```
Byte 0: player (0 or 1)
Byte 1: unit_type (0=warrior, 1=settler, 2=archer)
Byte 2: x position
Byte 3: y position
Byte 4: health (0-100)
Byte 5: moves_remaining
Byte 6: strength
Byte 7: unit id
```

**Cities** (9 bytes per city, variable count):
```
Byte 0:   player
Byte 1:   x
Byte 2:   y
Byte 3:   population
Byte 4-5: production (u16 LE)
Byte 6:   defense
Byte 7:   production_queue (0xFF = none, 0-2 = unit type)
Byte 8:   city id
```

**Action Mask** (47 bytes):
```
Packed bits, LSB first. Bit i is 1 if action i is valid.
47 bytes = 376 bits >= 376 actions.
```

**Spatial Tensor** (3328 bytes):
```
13 channels * 8 * 8 = 832 float32 values.
Little-endian f32, channel-major (CHW layout).
Channels 0-12 match Python MicroCivEncoder.
```

**Vector Features** (256 bytes):
```
64 float32 values, little-endian.
Indices 0-18 have semantic meaning (see compute_vector_features).
Indices 19-63 are zero-padded (reserved for future features).
```

### 3.2 Total State Size Per Turn

| Component | Size |
|-----------|------|
| Terrain | 64 B |
| Units (4 avg) | 32 B |
| Cities (2 avg) | 18 B |
| Action mask | 47 B |
| Spatial tensor | 3,328 B |
| Vector features | 256 B |
| Metadata | 16 B |
| **Total** | **~3.8 KB** |

This is much smaller than the 42 KB uncompressed Python representation
because we use compact binary encoding instead of JSON/dict structures.

### 3.3 Move History Record Size

| Component | Size |
|-----------|------|
| Fixed fields | ~30 B |
| Description | ~30 B avg |
| MCTS policy (AI only) | 1,504 B |
| MCTS value (AI only) | 4 B |
| Action mask snapshot | 47 B |
| **Total (AI move)** | **~1,615 B** |
| **Total (human move)** | **~107 B** |

## 4. Subscription Patterns

### 4.1 Active Game View (Web Client)

```sql
-- Subscribe to real-time game state updates
SELECT * FROM game_state WHERE game_id = :game_id

-- Subscribe to move history for live move feed
SELECT * FROM move_history WHERE game_id = :game_id

-- Subscribe to game metadata (status, winner, turn info)
SELECT * FROM game WHERE game_id = :game_id

-- Subscribe to spectator list
SELECT * FROM spectator WHERE game_id = :game_id

-- Subscribe to chat
SELECT * FROM chat_message WHERE game_id = :game_id
```

When a client subscribes to `game_state`, SpacetimeDB sends the full current
row immediately, then pushes diffs whenever the row changes. Since we update
the entire row on each move, the client receives the complete new state as a
single update. The client-side SDK caches the table locally.

### 4.2 AI Service

```sql
-- Subscribe to pending AI turn requests
SELECT * FROM ai_turn_request
```

The AI service only needs this one subscription. When a new row appears
(on_insert callback), it:
1. Deserializes spatial_tensor, vector_features, and action_mask.
2. Runs MCTS inference.
3. Calls `submit_move` reducer with the action and MCTS data.
4. The reducer deletes the ai_turn_request row (triggering on_delete on
   the AI service, which it ignores).

### 4.3 Lobby / Game List

```sql
-- All active games (for game browser)
SELECT * FROM game WHERE status = 1

-- All games waiting for players (for join)
SELECT * FROM game WHERE status = 0

-- Leaderboard (top 100 by ELO)
SELECT * FROM player ORDER BY elo_rating DESC LIMIT 100

-- Matchmaking queue status
SELECT * FROM matchmaking_queue
```

### 4.4 Tournament View

```sql
-- Tournament details
SELECT * FROM tournament WHERE tournament_id = :id

-- Bracket structure
SELECT * FROM tournament_bracket WHERE tournament_id = :id

-- Registered players
SELECT * FROM tournament_registration WHERE tournament_id = :id
```

### 4.5 Replay

```sql
-- Load all moves for a completed game
SELECT * FROM move_history WHERE game_id = :game_id

-- Load snapshots for fast seeking
SELECT * FROM game_state_snapshot WHERE game_id = :game_id
```

The client loads all snapshots, finds the nearest one before the desired
turn, then replays moves from that point forward to reconstruct the state.

## 5. Reducer Reference

### 5.1 Player Management

| Reducer | Args | Auth | Description |
|---------|------|------|-------------|
| `register_player` | display_name: String | identity | Create/update player profile |
| `register_ai_player` | display_name: String, model_version: String | identity | Register AI player |

### 5.2 Game Lifecycle

| Reducer | Args | Auth | Description |
|---------|------|------|-------------|
| `create_game` | mode: u8, max_turns: u16, map_seed: u64 | registered player | Create new game |
| `join_game` | game_id: u64 | registered player | Join as player 1 |
| `submit_move` | game_id: u64, action_index: u16, mcts_policy: Vec<u8>, mcts_value: Vec<u8> | game participant | Submit a move |
| `request_ai_turn` | game_id: u64 | game participant | Explicitly request AI turn |
| `forfeit` | game_id: u64 | game participant | Forfeit the game |

### 5.3 Social

| Reducer | Args | Auth | Description |
|---------|------|------|-------------|
| `spectate_game` | game_id: u64 | registered player | Start spectating |
| `stop_spectating` | game_id: u64 | registered player | Stop spectating |
| `send_chat` | game_id: u64, content: String | registered player | Send chat message |
| `join_matchmaking` | desired_mode: u8 | registered player | Enter queue |
| `leave_matchmaking` | (none) | registered player | Leave queue |

### 5.4 Tournaments

| Reducer | Args | Auth | Description |
|---------|------|------|-------------|
| `create_tournament` | name: String, max_players: u16 | registered player | Create tournament |
| `register_for_tournament` | tournament_id: u64 | registered player | Register |
| `start_tournament` | tournament_id: u64 | tournament creator | Start tournament |

## 6. Python AI Service Integration

### 6.1 Connection Flow

```
1. AI service starts
2. Loads model checkpoint (Civ6TransformerNetwork)
3. Creates GumbelMCTS with 25 simulations
4. Connects to SpacetimeDB via Python SDK
5. Calls register_ai_player("AlphaZero-v268", "v268")
6. Subscribes to SELECT * FROM ai_turn_request
7. Enters callback loop:
   a. on_insert(ai_turn_request): deserialize, run MCTS, call submit_move
   b. on_delete(ai_turn_request): no-op
```

### 6.2 Data Flow Per AI Move

```
Server (Rust)                    AI Service (Python)
=============                    ==================
apply_action()
  -> compute_spatial_tensor()
  -> compute_vector_features()
  -> compute_action_mask()
  -> INSERT ai_turn_request
       spatial: 3328 bytes
       vector:  256 bytes
       mask:    47 bytes
                                 <- on_insert callback
                                 unpack_spatial_tensor() -> (13,8,8) f32
                                 unpack_vector_features() -> (64,) f32
                                 unpack_action_mask() -> (376,) f32
                                 network.forward(spatial, vector, mask)
                                   -> policy_logits, value
                                 GumbelMCTS.search(env_shim)
                                   -> action_probs, root_value
                                 action = argmax(action_probs)
                                 call submit_move(
                                   game_id,
                                   action,       # u16
                                   policy_bytes, # 1504 bytes
                                   value_bytes,  # 4 bytes
                                 )
submit_move()                    ->
  validate action_mask[action]
  record move_history
  apply_action()
  DELETE ai_turn_request
  (if next player is AI,
   INSERT new ai_turn_request)
```

### 6.3 Latency Budget

| Step | Target | Notes |
|------|--------|-------|
| SpacetimeDB subscription push | < 5 ms | Over WebSocket |
| Tensor deserialization | < 1 ms | numpy frombuffer |
| MCTS inference (25 sims, CPU) | ~50-200 ms | GumbelMCTS on CPU |
| MCTS inference (25 sims, T4 GPU) | ~10-30 ms | With CUDA |
| submit_move reducer | < 5 ms | WASM execution |
| **Total per AI move** | **~60-210 ms** | End-to-end |

For Human vs AI games, this latency is imperceptible. For AI vs AI training
at scale, the bottleneck is MCTS inference, not SpacetimeDB overhead.

## 7. Testing Strategy

### 7.1 Unit Tests (Rust)

Test the internal game logic functions in isolation:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_generation_deterministic() {
        let t1 = generate_terrain(8, 12345);
        let t2 = generate_terrain(8, 12345);
        assert_eq!(t1, t2);
        // Different seed -> different terrain
        let t3 = generate_terrain(8, 99999);
        assert_ne!(t1, t3);
    }

    #[test]
    fn test_action_mask_end_turn_always_valid() {
        let terrain = vec![0u8; 64]; // all plains
        let units = vec![]; // no units
        let cities = vec![]; // no cities
        let mask = compute_action_mask(&terrain, &units, &cities, 8, 0);
        assert!(is_action_valid(&mask, 0)); // end_turn
    }

    #[test]
    fn test_action_mask_bits_roundtrip() {
        let mut mask = vec![0u8; 47];
        set_action_valid(&mut mask, 0);
        set_action_valid(&mut mask, 100);
        set_action_valid(&mut mask, 375);
        assert!(is_action_valid(&mask, 0));
        assert!(is_action_valid(&mask, 100));
        assert!(is_action_valid(&mask, 375));
        assert!(!is_action_valid(&mask, 1));
        assert!(!is_action_valid(&mask, 374));
    }

    #[test]
    fn test_unit_encoding_roundtrip() {
        let mut buf = Vec::new();
        encode_unit(&mut buf, 0, 1, 3, 4, 80, 2, 15, 7);
        let decoded = decode_units(&buf);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].player, 0);
        assert_eq!(decoded[0].unit_type, 1);
        assert_eq!(decoded[0].x, 3);
        assert_eq!(decoded[0].y, 4);
        assert_eq!(decoded[0].health, 80);
        assert_eq!(decoded[0].id, 7);
    }

    #[test]
    fn test_city_encoding_roundtrip() {
        let cities = vec![DecodedCity {
            player: 1, x: 5, y: 6, population: 3,
            production: 25, defense: 20,
            production_queue: 0xFF, id: 2,
        }];
        let encoded = encode_cities(&cities);
        let decoded = decode_cities(&encoded);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].player, 1);
        assert_eq!(decoded[0].production, 25);
        assert_eq!(decoded[0].production_queue, 0xFF);
    }

    #[test]
    fn test_describe_action() {
        assert_eq!(describe_action(0), "End turn");
        assert_eq!(describe_action(1), "Unit 0 moves NW");
        assert_eq!(describe_action(9), "Unit 1 moves NW");
        assert_eq!(describe_action(161), "Unit 0 attacks NW");
        assert_eq!(describe_action(321), "Unit 0 skips");
        assert_eq!(describe_action(341), "Unit 0 founds city");
        assert_eq!(describe_action(361), "City 0 produces warrior");
        assert_eq!(describe_action(362), "City 0 produces settler");
    }

    #[test]
    fn test_elo_calculation() {
        // Equal players, player 0 wins -> both move ~16 points
        let e0 = 1500.0;
        let e1 = 1500.0;
        let expected_0 = 1.0 / (1.0 + 10.0_f64.powf((e1 - e0) / 400.0));
        let new_elo0 = 1500 + (32.0 * (1.0 - expected_0)) as i32;
        let new_elo1 = 1500 + (32.0 * (0.0 - (1.0 - expected_0))) as i32;
        assert_eq!(new_elo0, 1516);
        assert_eq!(new_elo1, 1484);
    }

    #[test]
    fn test_spatial_tensor_dimensions() {
        let terrain = generate_terrain(8, 42);
        let mut units = Vec::new();
        encode_unit(&mut units, 0, 0, 2, 3, 100, 2, 20, 0);
        let cities = Vec::new();
        let tensor = compute_spatial_tensor(&terrain, &units, &cities, 8, 0);
        // 13 channels * 8 * 8 * 4 bytes = 3328
        assert_eq!(tensor.len(), 3328);
    }

    #[test]
    fn test_vector_features_dimensions() {
        let units = Vec::new();
        let cities = Vec::new();
        let features = compute_vector_features(&units, &cities, 0, 0);
        // 64 * 4 bytes = 256
        assert_eq!(features.len(), 256);
    }

    #[test]
    fn test_movement_applies_correctly() {
        let terrain = vec![0u8; 64]; // all plains
        let mut units = Vec::new();
        encode_unit(&mut units, 0, 0, 3, 3, 100, 2, 20, 0); // warrior at (3,3)
        encode_unit(&mut units, 1, 0, 7, 7, 100, 2, 20, 1); // enemy at (7,7)
        let cities = Vec::new();
        let mask = compute_action_mask(&terrain, &units, &cities, 8, 0);

        let state = GameState {
            game_id: 1,
            terrain: terrain.clone(),
            units,
            cities: Vec::new(),
            turn_number: 0,
            current_player: 0,
            next_unit_id: 2,
            next_city_id: 0,
            action_mask: mask,
            spatial_tensor: Vec::new(),
            vector_features: Vec::new(),
            updated_at: Timestamp::UNIX_EPOCH,
        };

        // Move unit 0 East (direction index 4 = (1, 0))
        let action = MOVE_OFFSET + 0 * 8 + 4; // unit 0, dir E
        let (new_state, game_over, winner) = apply_action(state, action, 8);
        assert!(!game_over);
        assert!(winner.is_none());

        let decoded = decode_units(&new_state.units);
        let own = decoded.iter().find(|u| u.player == 0).unwrap();
        assert_eq!(own.x, 4); // moved from 3 to 4
        assert_eq!(own.y, 3); // y unchanged
        assert_eq!(own.moves_remaining, 1); // 2 -> 1
    }
}
```

### 7.2 Integration Tests (SpacetimeDB testbed)

Use `spacetime test` or the SpacetimeDB testing framework:

```rust
// tests/integration.rs (pseudo-code for SpacetimeDB test framework)

#[spacetimedb::test]
fn test_full_game_lifecycle(ctx: &TestContext) {
    // 1. Register two players
    ctx.call("register_player", &["Alice"]);
    ctx.call("register_player", &["Bob"]);

    // 2. Create game
    ctx.call_as("Alice", "create_game", &[
        GameMode::HumanVsHuman as u8,
        30, // max_turns
        42, // seed
    ]);
    let game = ctx.read_table::<Game>().first().unwrap();
    assert_eq!(game.status, GameStatus::WaitingForPlayers as u8);

    // 3. Join game
    ctx.call_as("Bob", "join_game", &[game.game_id]);
    let game = ctx.read_table::<Game>().first().unwrap();
    assert_eq!(game.status, GameStatus::InProgress as u8);

    // 4. Verify initial state
    let state = ctx.read_table::<GameState>().first().unwrap();
    assert_eq!(state.current_player, 0);
    assert!(is_action_valid(&state.action_mask, 0)); // end_turn

    // 5. Play some moves
    ctx.call_as("Alice", "submit_move", &[game.game_id, 0, vec![], vec![]]);
    // Alice ends turn -> Bob's turn
    let state = ctx.read_table::<GameState>().first().unwrap();
    assert_eq!(state.current_player, 1);

    // 6. Forfeit
    ctx.call_as("Bob", "forfeit", &[game.game_id]);
    let game = ctx.read_table::<Game>().first().unwrap();
    assert_eq!(game.status, GameStatus::Completed as u8);
    assert_eq!(game.winner, 0); // Alice wins
}

#[spacetimedb::test]
fn test_ai_integration(ctx: &TestContext) {
    // 1. Register human + AI
    ctx.call("register_player", &["Human"]);
    ctx.call("register_ai_player", &["AI-v268", "v268"]);

    // 2. Create HvAI game
    ctx.call_as("Human", "create_game", &[
        GameMode::HumanVsAi as u8, 30, 42,
    ]);
    let game = ctx.read_table::<Game>().first().unwrap();

    // 3. AI joins
    ctx.call_as("AI-v268", "join_game", &[game.game_id]);

    // 4. Human moves
    ctx.call_as("Human", "submit_move", &[game.game_id, 0, vec![], vec![]]);

    // 5. Verify AI turn request was created
    let requests = ctx.read_table::<AiTurnRequest>();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].game_id, game.game_id);
    assert_eq!(requests[0].player_slot, 1);
    assert!(!requests[0].spatial_tensor.is_empty());
}
```

### 7.3 Python Integration Tests

```python
# tests/test_ai_service.py

import numpy as np
import pytest
from server_module.ai_service import (
    unpack_spatial_tensor,
    unpack_vector_features,
    unpack_action_mask,
    pack_policy,
    pack_value,
    SPATIAL_CHANNELS,
    BOARD_SIZE,
    VECTOR_DIM,
    ACTION_SPACE_SIZE,
    ACTION_MASK_BYTES,
)


def test_spatial_tensor_roundtrip():
    """Verify Python can decode what Rust produces."""
    original = np.random.randn(SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype("<f4")
    blob = original.tobytes()
    decoded = unpack_spatial_tensor(blob)
    np.testing.assert_array_equal(original, decoded)


def test_vector_features_roundtrip():
    original = np.random.randn(VECTOR_DIM).astype("<f4")
    blob = original.tobytes()
    decoded = unpack_vector_features(blob)
    np.testing.assert_array_equal(original, decoded)


def test_action_mask_roundtrip():
    """Verify bit-packing matches between Rust and Python."""
    # Create a mask with known valid actions
    valid_actions = [0, 1, 100, 200, 375]
    mask_bytes = bytearray(ACTION_MASK_BYTES)
    for a in valid_actions:
        mask_bytes[a // 8] |= 1 << (a % 8)

    decoded = unpack_action_mask(bytes(mask_bytes))
    for a in range(ACTION_SPACE_SIZE):
        if a in valid_actions:
            assert decoded[a] == 1.0, f"Action {a} should be valid"
        else:
            assert decoded[a] == 0.0, f"Action {a} should be invalid"


def test_policy_pack_size():
    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    blob = pack_policy(policy)
    assert len(blob) == ACTION_SPACE_SIZE * 4  # 1504 bytes


def test_value_pack():
    blob = pack_value(0.75)
    assert len(blob) == 4
    import struct
    val = struct.unpack("<f", blob)[0]
    assert abs(val - 0.75) < 1e-6


def test_spatial_tensor_wrong_size():
    with pytest.raises(ValueError, match="spatial_tensor"):
        unpack_spatial_tensor(b"\x00" * 100)


def test_action_mask_wrong_size():
    with pytest.raises(ValueError, match="action_mask"):
        unpack_action_mask(b"\x00" * 10)
```

### 7.4 Cross-Language Consistency Tests

These tests verify that the Rust game engine produces identical results to
the Python MicroCivEnv. Run them as part of CI:

```python
# tests/test_cross_language_consistency.py

def test_action_space_offsets_match():
    """Verify Rust and Python agree on action space layout."""
    from alpha_zero.core.civ6_actions import MicroCivActionSpace

    py = MicroCivActionSpace(board_size=8, max_units=20, max_cities=5)
    assert py.total_size == 376
    assert py.end_turn_offset == 0
    assert py.move_offset == 1
    assert py.attack_offset == 161
    assert py.skip_offset == 321
    assert py.found_city_offset == 341
    assert py.produce_offset == 361


def test_direction_ordering_matches():
    """Verify direction arrays are identical."""
    from alpha_zero.core.civ6_actions import MicroCivActionSpace
    py = MicroCivActionSpace(board_size=8)
    # Rust DIRECTIONS: [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    rust_dirs = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    assert py.directions == rust_dirs
```

## 8. Performance Considerations

### 8.1 Throughput Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Moves/sec (single game) | > 100 | Reducer execution < 10ms |
| Concurrent active games | > 1000 | Limited by memory |
| AI move latency (25 sims, CPU) | < 300ms | End-to-end |
| AI move latency (25 sims, T4) | < 50ms | End-to-end |
| Subscription push latency | < 10ms | SpacetimeDB WebSocket |
| Matchmaking latency | < 1s | Per-call scan |
| Replay load time (30-move game) | < 50ms | Move history fetch |

### 8.2 Hot Path Analysis

The hottest code path is `submit_move`, called once per move:

```
submit_move()                      ~3ms total
  +-- validate caller identity     ~0.1ms (B-tree lookup)
  +-- load game + game_state       ~0.2ms (PK lookups)
  +-- validate action mask         ~0.01ms (bit check)
  +-- insert move_history          ~0.1ms (append)
  +-- apply_action()               ~0.5ms
  |   +-- decode units/cities      ~0.01ms
  |   +-- execute action logic     ~0.01ms
  |   +-- encode units/cities      ~0.01ms
  |   +-- compute_action_mask()    ~0.1ms (iterate units)
  |   +-- compute_spatial_tensor() ~0.2ms (fill 3328 bytes)
  |   +-- compute_vector_features()~0.05ms (fill 256 bytes)
  +-- update game_state            ~0.1ms (PK update)
  +-- check AI turn + queue        ~0.2ms
```

### 8.3 Optimization Opportunities

1. **Lazy tensor computation**: Only compute spatial_tensor and
   vector_features when an AI turn request is needed. For human-vs-human
   games, skip tensor computation entirely. Saves ~0.25ms per move.

2. **Incremental action mask**: Instead of recomputing the full mask,
   maintain a delta based on the applied action. Complex to implement
   but would reduce mask computation from O(units * directions) to O(1).

3. **Batch matchmaking**: Instead of running matchmaking on every
   `join_matchmaking` call, run it on a timer (e.g. every 5 seconds)
   to amortize the sorting cost.

4. **Archive old games**: Move completed games older than N days to an
   archive table that is not subscribed to by active clients.

5. **Snapshot frequency**: Adjust snapshot frequency based on game length.
   Short games (< 20 moves) need fewer snapshots.

### 8.4 Memory Estimates

Per active game in SpacetimeDB memory:
- game row: ~150 B
- game_state row: ~4 KB
- Subscription overhead: ~500 B per subscriber

For 1000 concurrent games with 3 subscribers each:
- Tables: 1000 * 4.2 KB = ~4.2 MB
- Subscriptions: 1000 * 3 * 500 B = ~1.5 MB
- Total: ~6 MB (trivial)

## 9. Deployment

### 9.1 Local Development

```bash
# Install SpacetimeDB CLI
curl -fsSL https://install.spacetimedb.com | bash

# Build and publish the module
cd server-module
spacetime build
spacetime publish civ6az

# Start AI service
cd ..
python -m server_module.ai_service \
    --host http://localhost:3000 \
    --db-name civ6az \
    --model-path checkpoints/v268.pt \
    --mcts-sims 25

# Connect web client (TypeScript SDK)
# See web-client/ directory
```

### 9.2 Production (SpacetimeDB Cloud)

```bash
# Publish to SpacetimeDB Cloud
spacetime publish civ6az --server https://cloud.spacetimedb.com

# AI service on GCP (alongside existing Vertex AI infrastructure)
# Use the same VM (civ6az-orchestrator) or a dedicated e2-medium
```

### 9.3 Version Compatibility

The Rust module and Python AI service share these constants that MUST stay
in sync:

| Constant | Rust | Python | Where |
|----------|------|--------|-------|
| BOARD_SIZE | 8 | 8 | lib.rs / ai_service.py |
| ACTION_SPACE_SIZE | 376 | 376 | lib.rs / ai_service.py |
| SPATIAL_CHANNELS | 13 | 13 | lib.rs / ai_service.py |
| VECTOR_DIM | 64 | 64 | lib.rs / ai_service.py |
| Direction order | DIRECTIONS array | self.directions | lib.rs / civ6_actions.py |
| Unit encoding | 8 bytes/unit | N/A (via blobs) | lib.rs |
| Action mask packing | LSB-first bits | LSB-first bits | lib.rs / ai_service.py |

If any of these change, both sides must be updated simultaneously.
Add a version handshake or constant assertion in CI to catch mismatches.
