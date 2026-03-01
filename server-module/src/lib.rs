// =============================================================================
// Civ6 AlphaZero -- SpacetimeDB Server Module
// =============================================================================
//
// This module implements the authoritative game server for MicroCiv, a
// simplified Civilization environment used in AlphaZero self-play training.
//
// Architecture overview:
//   - Tables define the persistent schema (games, moves, players, ratings).
//   - Reducers are transactional RPC endpoints invoked by clients.
//   - SpacetimeDB subscriptions push real-time row diffs to connected clients.
//   - A Python AI service communicates via the SpacetimeDB SDK, calling
//     `ai_turn_request` and receiving board states through subscriptions.
//
// Game constants (must stay in sync with Python MicroCivEnv):
//   Board size:        8x8
//   Spatial channels:  13
//   Action space:      376
//   Unit types:        warrior (0), settler (1), archer (2)
//   Directions:        8 cardinal + diagonal
//   Max units/player:  20
//   Max cities/player: 5
//
// Action layout (376 total):
//   [0]       end_turn
//   [1..161]  move:       20 units * 8 dirs
//   [161..321] attack:    20 units * 8 dirs
//   [321..341] skip:      20 units
//   [341..361] found_city: 20 units
//   [361..376] produce:   5 cities * 3 unit_types
// =============================================================================

use spacetimedb::{table, reducer, Identity, Timestamp, ReducerContext, Table};
use log;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BOARD_SIZE: u8 = 8;
const ACTION_SPACE_SIZE: u16 = 376;
const MAX_UNITS_PER_PLAYER: u8 = 20;
const MAX_CITIES_PER_PLAYER: u8 = 5;
const NUM_UNIT_TYPES: u8 = 3; // warrior, settler, archer
const NUM_DIRECTIONS: u8 = 8;
const SPATIAL_CHANNELS: u8 = 13;
const VECTOR_DIM: u8 = 64;
const DEFAULT_MAX_TURNS: u16 = 30;
const DEFAULT_ELO: i32 = 1500;
const ELO_K_FACTOR: f64 = 32.0;

// Action space offsets (must match Python MicroCivActionSpace)
const END_TURN_OFFSET: u16 = 0;
const MOVE_OFFSET: u16 = 1;
const ATTACK_OFFSET: u16 = 161;  // 1 + 20*8
const SKIP_OFFSET: u16 = 321;    // 161 + 20*8
const FOUND_CITY_OFFSET: u16 = 341; // 321 + 20
const PRODUCE_OFFSET: u16 = 361; // 341 + 20

// Direction deltas (same order as Python): 8 directions
const DIRECTIONS: [(i8, i8); 8] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
];

// Unit stats
const UNIT_STRENGTH: [u8; 3] = [20, 0, 15];  // warrior, settler, archer
const UNIT_MOVES: [u8; 3] = [2, 2, 2];
const UNIT_COSTS: [u16; 3] = [30, 50, 40]; // production cost (turns * 10)

// ---------------------------------------------------------------------------
// Enums (stored as u8 in tables)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum GameStatus {
    WaitingForPlayers = 0,
    InProgress = 1,
    Completed = 2,
    Abandoned = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum GameMode {
    HumanVsHuman = 0,
    HumanVsAi = 1,
    AiVsAi = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum PlayerSlot {
    Player0 = 0,
    Player1 = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum TerrainType {
    Plains = 0,
    Hills = 1,
    Mountain = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum UnitType {
    Warrior = 0,
    Settler = 1,
    Archer = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum MatchResult {
    Player0Win = 0,
    Player1Win = 1,
    Draw = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum TournamentStatus {
    Registering = 0,
    InProgress = 1,
    Completed = 2,
}

// ---------------------------------------------------------------------------
// Table: Player
// ---------------------------------------------------------------------------
// Persistent player profile with ELO rating and identity binding.
// Indexed by identity for O(1) lookup from reducer context.
// ---------------------------------------------------------------------------

#[table(name = player, public)]
pub struct Player {
    #[primary_key]
    #[auto_inc]
    pub player_id: u64,

    #[unique]
    pub identity: Identity,

    pub display_name: String,
    pub elo_rating: i32,
    pub games_played: u32,
    pub games_won: u32,
    pub games_lost: u32,
    pub games_drawn: u32,
    pub is_ai: bool,
    /// Model version if AI player (e.g. "v268")
    pub ai_model_version: String,
    pub created_at: Timestamp,
    pub last_active: Timestamp,
}

// ---------------------------------------------------------------------------
// Table: Game
// ---------------------------------------------------------------------------
// One row per game. Holds metadata, not full board state. Board state is
// reconstructed from GameState (latest snapshot) or replayed from MoveHistory.
//
// Design decision: We store a compressed board snapshot in GameState rather
// than replaying all moves, because clients subscribing to an active game
// need the current state immediately (< 1ms) without replay overhead.
// ---------------------------------------------------------------------------

#[table(name = game, public)]
pub struct Game {
    #[primary_key]
    #[auto_inc]
    pub game_id: u64,

    pub status: u8,           // GameStatus as u8
    pub mode: u8,             // GameMode as u8
    pub player0_id: u64,      // FK -> player.player_id
    pub player1_id: u64,      // FK -> player.player_id (0 = waiting)
    pub current_player: u8,   // 0 or 1
    pub turn_number: u16,
    pub max_turns: u16,
    pub winner: u8,           // 0, 1, or 255 (no winner / draw)
    pub result: u8,           // MatchResult as u8; 255 = not finished

    /// Random seed used to generate the map (for reproducibility)
    pub map_seed: u64,
    pub board_size: u8,

    pub created_at: Timestamp,
    pub started_at: Timestamp,
    pub finished_at: Timestamp,

    /// Tournament this game belongs to (0 = casual)
    pub tournament_id: u64,
    pub tournament_round: u16,
}

// ---------------------------------------------------------------------------
// Table: GameState
// ---------------------------------------------------------------------------
// Stores the full authoritative board state for one game as a binary blob.
// Updated after every move. One row per active game. Deleted when game ends
// (final state is preserved in GameStateSnapshot for replay).
//
// Serialization format (total ~1.2 KB for 8x8 board):
//   terrain:  board_size * board_size bytes (TerrainType as u8)
//   units:    variable, see Unit encoding below
//   cities:   variable, see City encoding below
//   metadata: turn, current_player, next_unit_id, next_city_id
//
// For real-time subscriptions, clients subscribe to:
//   SELECT * FROM game_state WHERE game_id = ?
// They receive diffs whenever the blob changes.
// ---------------------------------------------------------------------------

#[table(name = game_state, public)]
pub struct GameState {
    #[primary_key]
    pub game_id: u64,

    /// Flat terrain array: board_size * board_size bytes, row-major.
    /// terrain[y * board_size + x] = TerrainType as u8.
    pub terrain: Vec<u8>,

    /// Encoded units. Each unit: 8 bytes
    ///   [player:u8, unit_type:u8, x:u8, y:u8, health:u8, moves_remaining:u8, strength:u8, id:u8]
    pub units: Vec<u8>,

    /// Encoded cities. Each city: 12 bytes
    ///   [player:u8, x:u8, y:u8, population:u8, production:u16(le), defense:u8,
    ///    production_queue:u8(0xFF=none), id:u8, name_len:u8, ...name_bytes]
    /// Simplified: fixed 12 bytes each with truncated name
    pub cities: Vec<u8>,

    pub turn_number: u16,
    pub current_player: u8,
    pub next_unit_id: u16,
    pub next_city_id: u16,

    /// Precomputed action mask: 376 bits packed into 47 bytes (ceil(376/8))
    /// Bit i is set if action i is valid for the current player.
    pub action_mask: Vec<u8>,

    /// Compressed spatial tensor for AI consumption.
    /// 13 channels * 8 * 8 = 832 f32 values = 3328 bytes.
    /// Stored as little-endian f32 array. Only updated when AI needs it.
    pub spatial_tensor: Vec<u8>,

    /// Vector features: 64 f32 values = 256 bytes (little-endian).
    pub vector_features: Vec<u8>,

    pub updated_at: Timestamp,
}

// ---------------------------------------------------------------------------
// Table: MoveHistory
// ---------------------------------------------------------------------------
// Append-only log of every move in a game. Primary data source for replays.
// Indexed by (game_id, move_number) for efficient range queries.
// ---------------------------------------------------------------------------

#[table(name = move_history, public)]
pub struct MoveHistory {
    #[primary_key]
    #[auto_inc]
    pub move_id: u64,

    pub game_id: u64,
    pub move_number: u32,
    pub player: u8,
    pub action_index: u16,

    /// Human-readable description: "Unit 3 moves NE", "End turn", etc.
    pub description: String,

    /// Optional MCTS policy distribution (376 f32 values = 1504 bytes).
    /// Stored for AI moves to enable training data extraction.
    pub mcts_policy: Vec<u8>,

    /// MCTS value estimate at time of move (-1.0 to 1.0, as f32 LE bytes)
    pub mcts_value: Vec<u8>,

    /// Snapshot of action_mask at time of move (47 bytes, same as GameState).
    pub action_mask_snapshot: Vec<u8>,

    pub timestamp: Timestamp,
}

// ---------------------------------------------------------------------------
// Table: GameStateSnapshot
// ---------------------------------------------------------------------------
// Stores periodic snapshots of GameState for efficient replay seeking.
// One snapshot every N moves (e.g. every 10 moves + final state).
// Clients can seek to any point by loading nearest snapshot then replaying.
// ---------------------------------------------------------------------------

#[table(name = game_state_snapshot, public)]
pub struct GameStateSnapshot {
    #[primary_key]
    #[auto_inc]
    pub snapshot_id: u64,

    pub game_id: u64,
    pub move_number: u32,

    /// Full GameState blob at this point. Same encoding as GameState.
    pub terrain: Vec<u8>,
    pub units: Vec<u8>,
    pub cities: Vec<u8>,
    pub turn_number: u16,
    pub current_player: u8,
    pub next_unit_id: u16,
    pub next_city_id: u16,
    pub action_mask: Vec<u8>,

    pub created_at: Timestamp,
}

// ---------------------------------------------------------------------------
// Table: AiTurnRequest
// ---------------------------------------------------------------------------
// When a game reaches an AI player's turn, a row is inserted here.
// The Python AI service subscribes to this table, picks up requests,
// runs MCTS inference, and calls submit_move() with the chosen action.
// The row is deleted after the AI submits its move.
// ---------------------------------------------------------------------------

#[table(name = ai_turn_request, public)]
pub struct AiTurnRequest {
    #[primary_key]
    #[auto_inc]
    pub request_id: u64,

    pub game_id: u64,
    pub player_slot: u8,        // 0 or 1
    pub model_version: String,  // e.g. "v268"

    /// Spatial tensor (13*8*8 f32 LE = 3328 bytes) for direct inference
    pub spatial_tensor: Vec<u8>,
    /// Vector features (64 f32 LE = 256 bytes)
    pub vector_features: Vec<u8>,
    /// Action mask (47 bytes packed bits)
    pub action_mask: Vec<u8>,

    pub requested_at: Timestamp,
}

// ---------------------------------------------------------------------------
// Table: Spectator
// ---------------------------------------------------------------------------
// Tracks which players are spectating which games. Used for access control
// on subscriptions and to show spectator counts.
// ---------------------------------------------------------------------------

#[table(name = spectator, public)]
pub struct Spectator {
    #[primary_key]
    #[auto_inc]
    pub spectator_id: u64,

    pub game_id: u64,
    pub player_id: u64,
    pub joined_at: Timestamp,
}

// ---------------------------------------------------------------------------
// Table: Matchmaking
// ---------------------------------------------------------------------------
// Queue for players seeking games. The matchmaker reducer periodically
// scans this table, pairs players by ELO proximity, and creates games.
// ---------------------------------------------------------------------------

#[table(name = matchmaking_queue, public)]
pub struct MatchmakingQueue {
    #[primary_key]
    #[auto_inc]
    pub entry_id: u64,

    pub player_id: u64,
    pub desired_mode: u8,   // GameMode as u8
    pub elo_rating: i32,    // Denormalized for fast range matching
    pub queued_at: Timestamp,
}

// ---------------------------------------------------------------------------
// Table: Tournament
// ---------------------------------------------------------------------------

#[table(name = tournament, public)]
pub struct Tournament {
    #[primary_key]
    #[auto_inc]
    pub tournament_id: u64,

    pub name: String,
    pub status: u8,          // TournamentStatus as u8
    pub max_players: u16,
    pub current_round: u16,
    pub total_rounds: u16,   // ceil(log2(max_players)) for single-elim
    pub created_by: u64,     // player_id
    pub created_at: Timestamp,
    pub started_at: Timestamp,
    pub finished_at: Timestamp,
}

// ---------------------------------------------------------------------------
// Table: TournamentRegistration
// ---------------------------------------------------------------------------

#[table(name = tournament_registration, public)]
pub struct TournamentRegistration {
    #[primary_key]
    #[auto_inc]
    pub registration_id: u64,

    pub tournament_id: u64,
    pub player_id: u64,
    pub seed: u16,           // Bracket seed (0 = not yet seeded)
    pub eliminated: bool,
    pub registered_at: Timestamp,
}

// ---------------------------------------------------------------------------
// Table: TournamentBracket
// ---------------------------------------------------------------------------
// Stores bracket structure. Each row is one match slot in the bracket.
// ---------------------------------------------------------------------------

#[table(name = tournament_bracket, public)]
pub struct TournamentBracket {
    #[primary_key]
    #[auto_inc]
    pub bracket_id: u64,

    pub tournament_id: u64,
    pub round: u16,
    pub match_index: u16,   // Position within the round
    pub player0_id: u64,    // 0 = TBD (bye or not yet determined)
    pub player1_id: u64,
    pub game_id: u64,       // FK -> game (0 = not yet created)
    pub winner_id: u64,     // 0 = not yet determined
}

// ---------------------------------------------------------------------------
// Table: ChatMessage
// ---------------------------------------------------------------------------
// In-game and lobby chat. Spectators and players can send messages.
// ---------------------------------------------------------------------------

#[table(name = chat_message, public)]
pub struct ChatMessage {
    #[primary_key]
    #[auto_inc]
    pub message_id: u64,

    pub game_id: u64,      // 0 = lobby / global chat
    pub sender_id: u64,    // player_id
    pub content: String,
    pub sent_at: Timestamp,
}

// ===========================================================================
// REDUCERS
// ===========================================================================

// ---------------------------------------------------------------------------
// register_player: Create or update player profile
// ---------------------------------------------------------------------------

#[reducer]
pub fn register_player(ctx: &ReducerContext, display_name: String) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    // Check if player already exists
    if let Some(existing) = ctx.db.player().identity().find(identity) {
        // Update display name and last_active
        ctx.db.player().player_id().update(Player {
            display_name,
            last_active: now,
            ..existing
        });
        log::info!("Player {} updated display name", existing.player_id);
        return Ok(());
    }

    // Create new player
    let player = ctx.db.player().insert(Player {
        player_id: 0, // auto_inc
        identity,
        display_name: display_name.clone(),
        elo_rating: DEFAULT_ELO,
        games_played: 0,
        games_won: 0,
        games_lost: 0,
        games_drawn: 0,
        is_ai: false,
        ai_model_version: String::new(),
        created_at: now,
        last_active: now,
    });
    log::info!("New player registered: {} (id={})", display_name, player.player_id);
    Ok(())
}

// ---------------------------------------------------------------------------
// register_ai_player: Register an AI player (called by AI service)
// ---------------------------------------------------------------------------

#[reducer]
pub fn register_ai_player(
    ctx: &ReducerContext,
    display_name: String,
    model_version: String,
) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    if let Some(existing) = ctx.db.player().identity().find(identity) {
        ctx.db.player().player_id().update(Player {
            display_name,
            ai_model_version: model_version,
            last_active: now,
            ..existing
        });
        return Ok(());
    }

    ctx.db.player().insert(Player {
        player_id: 0,
        identity,
        display_name,
        elo_rating: DEFAULT_ELO,
        games_played: 0,
        games_won: 0,
        games_lost: 0,
        games_drawn: 0,
        is_ai: true,
        ai_model_version: model_version,
        created_at: now,
        last_active: now,
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// create_game: Create a new game
// ---------------------------------------------------------------------------

#[reducer]
pub fn create_game(
    ctx: &ReducerContext,
    mode: u8,
    max_turns: u16,
    map_seed: u64,
) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    let creator = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered. Call register_player first.".to_string())?;

    let max_turns = if max_turns == 0 { DEFAULT_MAX_TURNS } else { max_turns };

    let game = ctx.db.game().insert(Game {
        game_id: 0, // auto_inc
        status: GameStatus::WaitingForPlayers as u8,
        mode,
        player0_id: creator.player_id,
        player1_id: 0,
        current_player: 0,
        turn_number: 0,
        max_turns,
        winner: 255,
        result: 255,
        map_seed,
        board_size: BOARD_SIZE,
        created_at: now,
        started_at: Timestamp::UNIX_EPOCH,
        finished_at: Timestamp::UNIX_EPOCH,
        tournament_id: 0,
        tournament_round: 0,
    });

    log::info!(
        "Game {} created by player {} (mode={}, seed={})",
        game.game_id, creator.player_id, mode, map_seed
    );

    // If AI vs AI, the AI service will join both slots
    // If Human vs AI, queue an AI join request
    if mode == GameMode::HumanVsAi as u8 {
        // Insert a placeholder AI turn request that the AI service picks up
        // to know it should join this game as player 1
        log::info!("Game {} awaiting AI opponent", game.game_id);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// join_game: Join an existing game as player 1
// ---------------------------------------------------------------------------

#[reducer]
pub fn join_game(ctx: &ReducerContext, game_id: u64) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    let joiner = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    let game = ctx.db.game().game_id().find(game_id)
        .ok_or_else(|| format!("Game {} not found.", game_id))?;

    if game.status != GameStatus::WaitingForPlayers as u8 {
        return Err("Game is not waiting for players.".to_string());
    }
    if game.player0_id == joiner.player_id {
        return Err("Cannot join your own game.".to_string());
    }

    // Update game with player 1 and start it
    ctx.db.game().game_id().update(Game {
        player1_id: joiner.player_id,
        status: GameStatus::InProgress as u8,
        started_at: now,
        ..game
    });

    // Generate initial board state
    initialize_game_state(ctx, game_id, game.map_seed)?;

    log::info!(
        "Player {} joined game {}. Game started!",
        joiner.player_id, game_id
    );

    // If the first player is AI, queue an AI turn request
    let p0 = ctx.db.player().player_id().find(game.player0_id);
    if let Some(p0) = p0 {
        if p0.is_ai {
            queue_ai_turn(ctx, game_id, 0, &p0.ai_model_version)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// submit_move: Submit a move for the current player
// ---------------------------------------------------------------------------

#[reducer]
pub fn submit_move(
    ctx: &ReducerContext,
    game_id: u64,
    action_index: u16,
    mcts_policy: Vec<u8>,
    mcts_value: Vec<u8>,
) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    // Validate caller is a participant
    let caller = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    let game = ctx.db.game().game_id().find(game_id)
        .ok_or_else(|| format!("Game {} not found.", game_id))?;

    if game.status != GameStatus::InProgress as u8 {
        return Err("Game is not in progress.".to_string());
    }

    // Determine which player slot the caller occupies
    let player_slot = if caller.player_id == game.player0_id {
        0u8
    } else if caller.player_id == game.player1_id {
        1u8
    } else {
        return Err("You are not a player in this game.".to_string());
    };

    if game.current_player != player_slot {
        return Err(format!(
            "Not your turn. Current player is {}.",
            game.current_player
        ));
    }

    // Validate action index
    if action_index >= ACTION_SPACE_SIZE {
        return Err(format!(
            "Invalid action index {}. Must be < {}.",
            action_index, ACTION_SPACE_SIZE
        ));
    }

    // Load current game state
    let state = ctx.db.game_state().game_id().find(game_id)
        .ok_or_else(|| "Game state not found (internal error).".to_string())?;

    // Validate action against action mask
    if !is_action_valid(&state.action_mask, action_index) {
        return Err(format!(
            "Action {} is not valid in the current game state.",
            action_index
        ));
    }

    // Record move in history (before applying)
    let move_number = ctx.db.move_history().iter()
        .filter(|m| m.game_id == game_id)
        .count() as u32;

    let description = describe_action(action_index);

    ctx.db.move_history().insert(MoveHistory {
        move_id: 0, // auto_inc
        game_id,
        move_number,
        player: player_slot,
        action_index,
        description,
        mcts_policy,
        mcts_value,
        action_mask_snapshot: state.action_mask.clone(),
        timestamp: now,
    });

    // Save periodic snapshot for replay seeking (every 10 moves)
    if move_number % 10 == 0 {
        ctx.db.game_state_snapshot().insert(GameStateSnapshot {
            snapshot_id: 0,
            game_id,
            move_number,
            terrain: state.terrain.clone(),
            units: state.units.clone(),
            cities: state.cities.clone(),
            turn_number: state.turn_number,
            current_player: state.current_player,
            next_unit_id: state.next_unit_id,
            next_city_id: state.next_city_id,
            action_mask: state.action_mask.clone(),
            created_at: now,
        });
    }

    // Apply the action to the game state
    let (new_state, game_over, winner) = apply_action(state, action_index, game.board_size);

    // Update game state
    ctx.db.game_state().game_id().update(GameState {
        updated_at: now,
        ..new_state
    });

    // Clean up AI turn request if this was an AI move
    let ai_requests: Vec<_> = ctx.db.ai_turn_request().iter()
        .filter(|r| r.game_id == game_id)
        .collect();
    for req in ai_requests {
        ctx.db.ai_turn_request().request_id().delete(req.request_id);
    }

    // Handle game completion
    if game_over {
        let result_val = match winner {
            Some(0) => MatchResult::Player0Win as u8,
            Some(1) => MatchResult::Player1Win as u8,
            _ => MatchResult::Draw as u8,
        };
        let winner_val = winner.unwrap_or(255);

        ctx.db.game().game_id().update(Game {
            status: GameStatus::Completed as u8,
            winner: winner_val,
            result: result_val,
            finished_at: now,
            ..game
        });

        // Save final snapshot
        let final_state = ctx.db.game_state().game_id().find(game_id);
        if let Some(fs) = final_state {
            let final_move = move_number + 1;
            ctx.db.game_state_snapshot().insert(GameStateSnapshot {
                snapshot_id: 0,
                game_id,
                move_number: final_move,
                terrain: fs.terrain.clone(),
                units: fs.units.clone(),
                cities: fs.cities.clone(),
                turn_number: fs.turn_number,
                current_player: fs.current_player,
                next_unit_id: fs.next_unit_id,
                next_city_id: fs.next_city_id,
                action_mask: fs.action_mask,
                created_at: now,
            });
        }

        // Update ELO ratings
        update_elo(ctx, game.player0_id, game.player1_id, result_val);

        // Update tournament bracket if this is a tournament game
        if game.tournament_id > 0 {
            advance_tournament_bracket(
                ctx,
                game.tournament_id,
                game_id,
                winner.map(|w| if w == 0 { game.player0_id } else { game.player1_id }),
            );
        }

        log::info!(
            "Game {} finished. Winner: {:?}, Result: {}",
            game_id, winner, result_val
        );
    } else {
        // Update turn info on Game table
        let updated_state = ctx.db.game_state().game_id().find(game_id).unwrap();
        ctx.db.game().game_id().update(Game {
            current_player: updated_state.current_player,
            turn_number: updated_state.turn_number,
            ..game
        });

        // Check if next player is AI and queue turn request
        let next_player_id = if updated_state.current_player == 0 {
            game.player0_id
        } else {
            game.player1_id
        };
        if let Some(next_player) = ctx.db.player().player_id().find(next_player_id) {
            if next_player.is_ai {
                queue_ai_turn(
                    ctx,
                    game_id,
                    updated_state.current_player,
                    &next_player.ai_model_version,
                )?;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// ai_turn_request: Explicitly request AI to take a turn (called by game host)
// ---------------------------------------------------------------------------

#[reducer]
pub fn request_ai_turn(ctx: &ReducerContext, game_id: u64) -> Result<(), String> {
    let identity = ctx.sender;

    let caller = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    let game = ctx.db.game().game_id().find(game_id)
        .ok_or_else(|| format!("Game {} not found.", game_id))?;

    if game.status != GameStatus::InProgress as u8 {
        return Err("Game is not in progress.".to_string());
    }

    // Verify caller is a participant
    if caller.player_id != game.player0_id && caller.player_id != game.player1_id {
        return Err("You are not a player in this game.".to_string());
    }

    // Determine if current player is AI
    let current_player_id = if game.current_player == 0 {
        game.player0_id
    } else {
        game.player1_id
    };

    let current_player = ctx.db.player().player_id().find(current_player_id)
        .ok_or_else(|| "Current player not found (internal error).".to_string())?;

    if !current_player.is_ai {
        return Err("Current player is not an AI.".to_string());
    }

    queue_ai_turn(ctx, game_id, game.current_player, &current_player.ai_model_version)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// forfeit: A player forfeits the game
// ---------------------------------------------------------------------------

#[reducer]
pub fn forfeit(ctx: &ReducerContext, game_id: u64) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    let caller = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    let game = ctx.db.game().game_id().find(game_id)
        .ok_or_else(|| format!("Game {} not found.", game_id))?;

    if game.status != GameStatus::InProgress as u8 {
        return Err("Game is not in progress.".to_string());
    }

    let player_slot = if caller.player_id == game.player0_id {
        0u8
    } else if caller.player_id == game.player1_id {
        1u8
    } else {
        return Err("You are not a player in this game.".to_string());
    };

    // The other player wins
    let winner = 1 - player_slot;
    let result_val = if winner == 0 {
        MatchResult::Player0Win as u8
    } else {
        MatchResult::Player1Win as u8
    };

    ctx.db.game().game_id().update(Game {
        status: GameStatus::Completed as u8,
        winner,
        result: result_val,
        finished_at: now,
        ..game
    });

    update_elo(ctx, game.player0_id, game.player1_id, result_val);

    // Record forfeit in move history
    let move_count = ctx.db.move_history().iter()
        .filter(|m| m.game_id == game_id)
        .count() as u32;

    ctx.db.move_history().insert(MoveHistory {
        move_id: 0,
        game_id,
        move_number: move_count,
        player: player_slot,
        action_index: u16::MAX, // Sentinel for forfeit
        description: format!("Player {} forfeited", player_slot),
        mcts_policy: Vec::new(),
        mcts_value: Vec::new(),
        action_mask_snapshot: Vec::new(),
        timestamp: now,
    });

    log::info!(
        "Game {} forfeited by player {}. Winner: player {}",
        game_id, player_slot, winner
    );

    // Tournament advancement
    if game.tournament_id > 0 {
        let winner_id = if winner == 0 { game.player0_id } else { game.player1_id };
        advance_tournament_bracket(ctx, game.tournament_id, game_id, Some(winner_id));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// spectate_game: Start spectating a game
// ---------------------------------------------------------------------------

#[reducer]
pub fn spectate_game(ctx: &ReducerContext, game_id: u64) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    let viewer = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    let game = ctx.db.game().game_id().find(game_id)
        .ok_or_else(|| format!("Game {} not found.", game_id))?;

    if game.status != GameStatus::InProgress as u8 {
        return Err("Can only spectate games that are in progress.".to_string());
    }

    // Check if already spectating
    let already = ctx.db.spectator().iter()
        .any(|s| s.game_id == game_id && s.player_id == viewer.player_id);

    if already {
        return Err("Already spectating this game.".to_string());
    }

    ctx.db.spectator().insert(Spectator {
        spectator_id: 0,
        game_id,
        player_id: viewer.player_id,
        joined_at: now,
    });

    log::info!("Player {} now spectating game {}", viewer.player_id, game_id);
    Ok(())
}

// ---------------------------------------------------------------------------
// stop_spectating: Stop spectating a game
// ---------------------------------------------------------------------------

#[reducer]
pub fn stop_spectating(ctx: &ReducerContext, game_id: u64) -> Result<(), String> {
    let identity = ctx.sender;

    let viewer = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    let entries: Vec<_> = ctx.db.spectator().iter()
        .filter(|s| s.game_id == game_id && s.player_id == viewer.player_id)
        .collect();

    if entries.is_empty() {
        return Err("Not spectating this game.".to_string());
    }

    for entry in entries {
        ctx.db.spectator().spectator_id().delete(entry.spectator_id);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// join_matchmaking: Enter the matchmaking queue
// ---------------------------------------------------------------------------

#[reducer]
pub fn join_matchmaking(ctx: &ReducerContext, desired_mode: u8) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    let player = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    // Check not already in queue
    let already = ctx.db.matchmaking_queue().iter()
        .any(|e| e.player_id == player.player_id);

    if already {
        return Err("Already in matchmaking queue.".to_string());
    }

    ctx.db.matchmaking_queue().insert(MatchmakingQueue {
        entry_id: 0,
        player_id: player.player_id,
        desired_mode: desired_mode,
        elo_rating: player.elo_rating,
        queued_at: now,
    });

    log::info!("Player {} joined matchmaking (mode={})", player.player_id, desired_mode);

    // Attempt to match immediately
    try_matchmake(ctx)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// leave_matchmaking: Leave the matchmaking queue
// ---------------------------------------------------------------------------

#[reducer]
pub fn leave_matchmaking(ctx: &ReducerContext) -> Result<(), String> {
    let identity = ctx.sender;

    let player = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    let entries: Vec<_> = ctx.db.matchmaking_queue().iter()
        .filter(|e| e.player_id == player.player_id)
        .collect();

    for entry in entries {
        ctx.db.matchmaking_queue().entry_id().delete(entry.entry_id);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// send_chat: Send a chat message
// ---------------------------------------------------------------------------

#[reducer]
pub fn send_chat(ctx: &ReducerContext, game_id: u64, content: String) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    let sender = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    if content.is_empty() || content.len() > 500 {
        return Err("Message must be 1-500 characters.".to_string());
    }

    ctx.db.chat_message().insert(ChatMessage {
        message_id: 0,
        game_id,
        sender_id: sender.player_id,
        content,
        sent_at: now,
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// create_tournament: Create a new tournament
// ---------------------------------------------------------------------------

#[reducer]
pub fn create_tournament(
    ctx: &ReducerContext,
    name: String,
    max_players: u16,
) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    let creator = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    if max_players < 2 || max_players > 256 {
        return Err("Tournament must have 2-256 players.".to_string());
    }

    // Calculate rounds for single elimination
    let total_rounds = (max_players as f64).log2().ceil() as u16;

    let tournament = ctx.db.tournament().insert(Tournament {
        tournament_id: 0,
        name: name.clone(),
        status: TournamentStatus::Registering as u8,
        max_players,
        current_round: 0,
        total_rounds,
        created_by: creator.player_id,
        created_at: now,
        started_at: Timestamp::UNIX_EPOCH,
        finished_at: Timestamp::UNIX_EPOCH,
    });

    // Auto-register creator
    ctx.db.tournament_registration().insert(TournamentRegistration {
        registration_id: 0,
        tournament_id: tournament.tournament_id,
        player_id: creator.player_id,
        seed: 0,
        eliminated: false,
        registered_at: now,
    });

    log::info!(
        "Tournament '{}' created (id={}, max={}, rounds={})",
        name, tournament.tournament_id, max_players, total_rounds
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// register_for_tournament: Register for a tournament
// ---------------------------------------------------------------------------

#[reducer]
pub fn register_for_tournament(
    ctx: &ReducerContext,
    tournament_id: u64,
) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    let player = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    let tournament = ctx.db.tournament().tournament_id().find(tournament_id)
        .ok_or_else(|| format!("Tournament {} not found.", tournament_id))?;

    if tournament.status != TournamentStatus::Registering as u8 {
        return Err("Tournament is not accepting registrations.".to_string());
    }

    // Check capacity
    let current_count = ctx.db.tournament_registration().iter()
        .filter(|r| r.tournament_id == tournament_id)
        .count() as u16;

    if current_count >= tournament.max_players {
        return Err("Tournament is full.".to_string());
    }

    // Check not already registered
    let already = ctx.db.tournament_registration().iter()
        .any(|r| r.tournament_id == tournament_id && r.player_id == player.player_id);

    if already {
        return Err("Already registered for this tournament.".to_string());
    }

    ctx.db.tournament_registration().insert(TournamentRegistration {
        registration_id: 0,
        tournament_id,
        player_id: player.player_id,
        seed: 0,
        eliminated: false,
        registered_at: now,
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// start_tournament: Start a tournament (creator only)
// ---------------------------------------------------------------------------

#[reducer]
pub fn start_tournament(ctx: &ReducerContext, tournament_id: u64) -> Result<(), String> {
    let identity = ctx.sender;
    let now = ctx.timestamp;

    let caller = ctx.db.player().identity().find(identity)
        .ok_or_else(|| "Player not registered.".to_string())?;

    let tournament = ctx.db.tournament().tournament_id().find(tournament_id)
        .ok_or_else(|| format!("Tournament {} not found.", tournament_id))?;

    if tournament.created_by != caller.player_id {
        return Err("Only the tournament creator can start it.".to_string());
    }

    if tournament.status != TournamentStatus::Registering as u8 {
        return Err("Tournament already started.".to_string());
    }

    // Collect registered players, seed by ELO
    let mut registrations: Vec<_> = ctx.db.tournament_registration().iter()
        .filter(|r| r.tournament_id == tournament_id)
        .collect();

    if registrations.len() < 2 {
        return Err("Need at least 2 players to start.".to_string());
    }

    // Sort by ELO descending for seeding
    let player_elos: Vec<(u64, i32)> = registrations.iter()
        .filter_map(|r| {
            ctx.db.player().player_id().find(r.player_id)
                .map(|p| (p.player_id, p.elo_rating))
        })
        .collect();

    let mut sorted_players: Vec<(u64, i32)> = player_elos;
    sorted_players.sort_by(|a, b| b.1.cmp(&a.1));

    // Assign seeds
    for (seed_idx, (pid, _elo)) in sorted_players.iter().enumerate() {
        for reg in registrations.iter_mut() {
            if reg.player_id == *pid {
                let updated = TournamentRegistration {
                    seed: (seed_idx + 1) as u16,
                    ..reg.clone()
                };
                ctx.db.tournament_registration().registration_id().update(updated);
                break;
            }
        }
    }

    // Generate round 1 bracket
    let num_players = sorted_players.len();
    let matches_round1 = (num_players / 2) as u16;

    for i in 0..matches_round1 {
        let p0_idx = i as usize;
        let p1_idx = num_players - 1 - i as usize;

        let p0_id = sorted_players[p0_idx].0;
        let p1_id = if p1_idx < num_players && p1_idx != p0_idx {
            sorted_players[p1_idx].0
        } else {
            0 // bye
        };

        ctx.db.tournament_bracket().insert(TournamentBracket {
            bracket_id: 0,
            tournament_id,
            round: 1,
            match_index: i,
            player0_id: p0_id,
            player1_id: p1_id,
            game_id: 0,
            winner_id: 0,
        });
    }

    // Update tournament status
    ctx.db.tournament().tournament_id().update(Tournament {
        status: TournamentStatus::InProgress as u8,
        current_round: 1,
        started_at: now,
        ..tournament
    });

    log::info!(
        "Tournament {} started with {} players, {} first-round matches",
        tournament_id, num_players, matches_round1
    );

    Ok(())
}

// ===========================================================================
// INTERNAL FUNCTIONS (not exposed as reducers)
// ===========================================================================

// ---------------------------------------------------------------------------
// initialize_game_state: Generate map and starting units
// ---------------------------------------------------------------------------

fn initialize_game_state(
    ctx: &ReducerContext,
    game_id: u64,
    map_seed: u64,
) -> Result<(), String> {
    let now = ctx.timestamp;
    let board = BOARD_SIZE as usize;

    // Deterministic terrain generation from seed
    let terrain = generate_terrain(board, map_seed);

    // Place starting units
    let mut units: Vec<u8> = Vec::new();
    let mut next_unit_id: u16 = 0;

    // Player 0: settler + warrior near (1,1)
    let p0_settler_pos = find_valid_position(&terrain, board, 1, 1, 3, map_seed);
    encode_unit(
        &mut units, 0, UnitType::Settler as u8,
        p0_settler_pos.0, p0_settler_pos.1,
        100, 2, 0, next_unit_id as u8,
    );
    next_unit_id += 1;

    let p0_warrior_pos = find_adjacent_valid(&terrain, board, p0_settler_pos);
    encode_unit(
        &mut units, 0, UnitType::Warrior as u8,
        p0_warrior_pos.0, p0_warrior_pos.1,
        100, 2, 20, next_unit_id as u8,
    );
    next_unit_id += 1;

    // Player 1: settler + warrior near (board-3, board-3)
    let p1_start = (board as u8 - 3, board as u8 - 3);
    let p1_settler_pos = find_valid_position(
        &terrain, board, p1_start.0, p1_start.1, 3, map_seed.wrapping_add(1),
    );
    encode_unit(
        &mut units, 1, UnitType::Settler as u8,
        p1_settler_pos.0, p1_settler_pos.1,
        100, 2, 0, next_unit_id as u8,
    );
    next_unit_id += 1;

    let p1_warrior_pos = find_adjacent_valid(&terrain, board, p1_settler_pos);
    encode_unit(
        &mut units, 1, UnitType::Warrior as u8,
        p1_warrior_pos.0, p1_warrior_pos.1,
        100, 2, 20, next_unit_id as u8,
    );
    next_unit_id += 1;

    let cities: Vec<u8> = Vec::new();

    // Compute initial action mask for player 0
    let action_mask = compute_action_mask(
        &terrain, &units, &cities,
        board as u8, 0, // current_player = 0
    );

    // Compute initial spatial tensor and vector features
    let spatial_tensor = compute_spatial_tensor(
        &terrain, &units, &cities, board as u8, 0,
    );
    let vector_features = compute_vector_features(
        &units, &cities, 0, 0, // turn 0, player 0
    );

    ctx.db.game_state().insert(GameState {
        game_id,
        terrain,
        units,
        cities,
        turn_number: 0,
        current_player: 0,
        next_unit_id,
        next_city_id: 0,
        action_mask,
        spatial_tensor,
        vector_features,
        updated_at: now,
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Terrain generation (deterministic from seed)
// ---------------------------------------------------------------------------

fn generate_terrain(board: usize, seed: u64) -> Vec<u8> {
    let total = board * board;
    let mut terrain = vec![TerrainType::Plains as u8; total];

    // Simple LCG-based PRNG for deterministic generation
    let mut rng_state = seed;
    for i in 0..total {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((rng_state >> 33) as f64) / (u32::MAX as f64);

        if r < 0.70 {
            terrain[i] = TerrainType::Plains as u8;
        } else if r < 0.90 {
            terrain[i] = TerrainType::Hills as u8;
        } else {
            terrain[i] = TerrainType::Mountain as u8;
        }
    }

    terrain
}

fn find_valid_position(
    terrain: &[u8],
    board: usize,
    start_x: u8,
    start_y: u8,
    range: u8,
    seed: u64,
) -> (u8, u8) {
    let mut rng_state = seed;
    for _ in 0..100 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let dx = ((rng_state >> 33) % range as u64) as u8;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let dy = ((rng_state >> 33) % range as u64) as u8;

        let x = (start_x + dx).min(board as u8 - 1);
        let y = (start_y + dy).min(board as u8 - 1);
        let idx = y as usize * board + x as usize;

        if idx < terrain.len() && terrain[idx] != TerrainType::Mountain as u8 {
            return (x, y);
        }
    }
    (start_x, start_y)
}

fn find_adjacent_valid(terrain: &[u8], board: usize, pos: (u8, u8)) -> (u8, u8) {
    let deltas: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for (dx, dy) in deltas {
        let nx = pos.0 as i8 + dx;
        let ny = pos.1 as i8 + dy;
        if nx >= 0 && nx < board as i8 && ny >= 0 && ny < board as i8 {
            let idx = ny as usize * board + nx as usize;
            if idx < terrain.len() && terrain[idx] != TerrainType::Mountain as u8 {
                return (nx as u8, ny as u8);
            }
        }
    }
    pos
}

// ---------------------------------------------------------------------------
// Unit encoding/decoding (8 bytes per unit)
// ---------------------------------------------------------------------------
// Layout: [player, unit_type, x, y, health, moves_remaining, strength, id]

fn encode_unit(
    buf: &mut Vec<u8>,
    player: u8,
    unit_type: u8,
    x: u8,
    y: u8,
    health: u8,
    moves: u8,
    strength: u8,
    id: u8,
) {
    buf.extend_from_slice(&[player, unit_type, x, y, health, moves, strength, id]);
}

struct DecodedUnit {
    player: u8,
    unit_type: u8,
    x: u8,
    y: u8,
    health: u8,
    moves_remaining: u8,
    strength: u8,
    id: u8,
}

fn decode_units(data: &[u8]) -> Vec<DecodedUnit> {
    data.chunks_exact(8)
        .map(|chunk| DecodedUnit {
            player: chunk[0],
            unit_type: chunk[1],
            x: chunk[2],
            y: chunk[3],
            health: chunk[4],
            moves_remaining: chunk[5],
            strength: chunk[6],
            id: chunk[7],
        })
        .collect()
}

fn encode_units(units: &[DecodedUnit]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(units.len() * 8);
    for u in units {
        buf.extend_from_slice(&[
            u.player, u.unit_type, u.x, u.y,
            u.health, u.moves_remaining, u.strength, u.id,
        ]);
    }
    buf
}

// ---------------------------------------------------------------------------
// City encoding/decoding (8 bytes per city, simplified)
// ---------------------------------------------------------------------------
// Layout: [player, x, y, population, production_lo, production_hi, defense,
//          production_queue (0xFF=none), id]
// Actually 9 bytes, but we pad to keep it simple.

const CITY_RECORD_SIZE: usize = 9;

struct DecodedCity {
    player: u8,
    x: u8,
    y: u8,
    population: u8,
    production: u16,
    defense: u8,
    production_queue: u8, // 0xFF = none, else UnitType
    id: u8,
}

fn decode_cities(data: &[u8]) -> Vec<DecodedCity> {
    data.chunks_exact(CITY_RECORD_SIZE)
        .map(|c| DecodedCity {
            player: c[0],
            x: c[1],
            y: c[2],
            population: c[3],
            production: u16::from_le_bytes([c[4], c[5]]),
            defense: c[6],
            production_queue: c[7],
            id: c[8],
        })
        .collect()
}

fn encode_cities(cities: &[DecodedCity]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(cities.len() * CITY_RECORD_SIZE);
    for c in cities {
        let prod_bytes = c.production.to_le_bytes();
        buf.extend_from_slice(&[
            c.player, c.x, c.y, c.population,
            prod_bytes[0], prod_bytes[1],
            c.defense, c.production_queue, c.id,
        ]);
    }
    buf
}

// ---------------------------------------------------------------------------
// Action mask utilities
// ---------------------------------------------------------------------------

fn is_action_valid(mask: &[u8], action: u16) -> bool {
    let byte_idx = action as usize / 8;
    let bit_idx = action as usize % 8;
    if byte_idx >= mask.len() {
        return false;
    }
    (mask[byte_idx] >> bit_idx) & 1 == 1
}

fn set_action_valid(mask: &mut [u8], action: u16) {
    let byte_idx = action as usize / 8;
    let bit_idx = action as usize % 8;
    if byte_idx < mask.len() {
        mask[byte_idx] |= 1 << bit_idx;
    }
}

fn compute_action_mask(
    terrain: &[u8],
    units_data: &[u8],
    cities_data: &[u8],
    board_size: u8,
    current_player: u8,
) -> Vec<u8> {
    let mask_bytes = (ACTION_SPACE_SIZE as usize + 7) / 8; // 47 bytes
    let mut mask = vec![0u8; mask_bytes];

    // END_TURN always valid
    set_action_valid(&mut mask, END_TURN_OFFSET);

    let units = decode_units(units_data);
    let cities = decode_cities(cities_data);
    let bs = board_size as usize;

    // Build enemy position set
    let enemy_positions: Vec<(u8, u8)> = units.iter()
        .filter(|u| u.player != current_player)
        .map(|u| (u.x, u.y))
        .collect();

    // Build all unit position set (for collision checking)
    let all_unit_positions: Vec<(u8, u8)> = units.iter()
        .map(|u| (u.x, u.y))
        .collect();

    // Unit actions (only for current player's units)
    let mut own_idx: u16 = 0;
    for unit in &units {
        if own_idx >= MAX_UNITS_PER_PLAYER as u16 {
            break;
        }
        if unit.player != current_player {
            continue;
        }

        if unit.moves_remaining > 0 {
            // Movement: 8 directions
            for dir_idx in 0..NUM_DIRECTIONS as u16 {
                let (dx, dy) = DIRECTIONS[dir_idx as usize];
                let nx = unit.x as i8 + dx;
                let ny = unit.y as i8 + dy;

                if nx >= 0 && nx < board_size as i8 && ny >= 0 && ny < board_size as i8 {
                    let tidx = ny as usize * bs + nx as usize;
                    if tidx < terrain.len() && terrain[tidx] != TerrainType::Mountain as u8 {
                        // Check no unit collision
                        let target = (nx as u8, ny as u8);
                        if !all_unit_positions.contains(&target) {
                            set_action_valid(
                                &mut mask,
                                MOVE_OFFSET + own_idx * 8 + dir_idx,
                            );
                        }
                    }
                }
            }

            // Attacks: 8 directions toward enemy units
            for dir_idx in 0..NUM_DIRECTIONS as u16 {
                let (dx, dy) = DIRECTIONS[dir_idx as usize];
                let nx = unit.x as i8 + dx;
                let ny = unit.y as i8 + dy;

                if nx >= 0 && nx < board_size as i8 && ny >= 0 && ny < board_size as i8 {
                    let target = (nx as u8, ny as u8);
                    if enemy_positions.contains(&target) {
                        set_action_valid(
                            &mut mask,
                            ATTACK_OFFSET + own_idx * 8 + dir_idx,
                        );
                    }
                }
            }

            // Skip
            set_action_valid(&mut mask, SKIP_OFFSET + own_idx);

            // Found city (settlers only)
            if unit.unit_type == UnitType::Settler as u8 {
                let has_nearby_city = cities.iter().any(|c| {
                    let dx = (c.x as i16 - unit.x as i16).unsigned_abs();
                    let dy = (c.y as i16 - unit.y as i16).unsigned_abs();
                    dx <= 3 && dy <= 3
                });
                if !has_nearby_city {
                    set_action_valid(&mut mask, FOUND_CITY_OFFSET + own_idx);
                }
            }
        }

        own_idx += 1;
    }

    // City production
    let mut own_city_idx: u16 = 0;
    for city in &cities {
        if own_city_idx >= MAX_CITIES_PER_PLAYER as u16 {
            break;
        }
        if city.player != current_player {
            continue;
        }

        if city.production_queue == 0xFF {
            // No current production -- can produce any unit type
            for ut in 0..NUM_UNIT_TYPES as u16 {
                set_action_valid(
                    &mut mask,
                    PRODUCE_OFFSET + own_city_idx * NUM_UNIT_TYPES as u16 + ut,
                );
            }
        }

        own_city_idx += 1;
    }

    mask
}

// ---------------------------------------------------------------------------
// Action application: the core game rules engine
// ---------------------------------------------------------------------------

/// Apply an action to the game state. Returns (new_state, game_over, winner).
fn apply_action(
    mut state: GameState,
    action: u16,
    board_size: u8,
) -> (GameState, bool, Option<u8>) {
    let mut units = decode_units(&state.units);
    let mut cities = decode_cities(&state.cities);
    let current = state.current_player;
    let bs = board_size as usize;

    let mut game_over = false;
    let mut winner: Option<u8> = None;

    if action == END_TURN_OFFSET {
        // Process city production for current player
        process_city_production(&mut units, &mut cities, current, &state.terrain, bs, &mut state.next_unit_id);

        // Switch player
        state.current_player = 1 - current;

        // If back to player 0, increment turn and reset moves
        if state.current_player == 0 {
            state.turn_number += 1;
            for unit in &mut units {
                let moves = if (unit.unit_type as usize) < UNIT_MOVES.len() {
                    UNIT_MOVES[unit.unit_type as usize]
                } else {
                    2
                };
                unit.moves_remaining = moves;
            }
        }
    } else if action >= MOVE_OFFSET && action < ATTACK_OFFSET {
        // Movement
        let local = action - MOVE_OFFSET;
        let unit_idx = local / 8;
        let dir_idx = (local % 8) as usize;

        if let Some(unit) = get_own_unit_mut(&mut units, current, unit_idx as usize) {
            let (dx, dy) = DIRECTIONS[dir_idx];
            let nx = (unit.x as i8 + dx) as u8;
            let ny = (unit.y as i8 + dy) as u8;
            unit.x = nx;
            unit.y = ny;
            unit.moves_remaining = unit.moves_remaining.saturating_sub(1);
        }
    } else if action >= ATTACK_OFFSET && action < SKIP_OFFSET {
        // Attack
        let local = action - ATTACK_OFFSET;
        let unit_idx = local / 8;
        let dir_idx = (local % 8) as usize;

        let (dx, dy) = DIRECTIONS[dir_idx];

        // Get attacker stats
        let attacker_info = get_own_unit(&units, current, unit_idx as usize)
            .map(|u| (u.x, u.y, u.strength, u.health));

        if let Some((ax, ay, a_str, a_hp)) = attacker_info {
            let tx = (ax as i8 + dx) as u8;
            let ty = (ay as i8 + dy) as u8;

            // Find defender
            let defender_idx = units.iter().position(|u| {
                u.player != current && u.x == tx && u.y == ty
            });

            if let Some(def_idx) = defender_idx {
                let d_str = units[def_idx].strength;
                let d_hp = units[def_idx].health;

                // Combat resolution (same as Python)
                let a_eff = a_str as f64 * (a_hp as f64 / 100.0);
                let d_eff = d_str as f64 * (d_hp as f64 / 100.0);

                if a_eff + d_eff > 0.0 {
                    let ratio = a_eff / (a_eff + d_eff + 0.1);
                    let dmg_to_def = (30.0 * ratio + 10.0) as u8;
                    let dmg_to_att = (30.0 * (1.0 - ratio) + 5.0) as u8;

                    units[def_idx].health = units[def_idx].health.saturating_sub(dmg_to_def);

                    // Update attacker
                    if let Some(att) = get_own_unit_mut(&mut units, current, unit_idx as usize) {
                        att.health = att.health.saturating_sub(dmg_to_att);
                        att.moves_remaining = att.moves_remaining.saturating_sub(1);
                    }

                    // Remove dead units (defender first to avoid index shift issues)
                    units.retain(|u| u.health > 0);
                }
            }
        }
    } else if action >= SKIP_OFFSET && action < FOUND_CITY_OFFSET {
        // Skip
        let unit_idx = (action - SKIP_OFFSET) as usize;
        if let Some(unit) = get_own_unit_mut(&mut units, current, unit_idx) {
            unit.moves_remaining = 0;
        }
    } else if action >= FOUND_CITY_OFFSET && action < PRODUCE_OFFSET {
        // Found city
        let unit_idx = (action - FOUND_CITY_OFFSET) as usize;

        let unit_info = get_own_unit(&units, current, unit_idx)
            .filter(|u| u.unit_type == UnitType::Settler as u8)
            .map(|u| (u.x, u.y, u.id));

        if let Some((ux, uy, uid)) = unit_info {
            // Create city
            cities.push(DecodedCity {
                player: current,
                x: ux,
                y: uy,
                population: 1,
                production: 0,
                defense: 20,
                production_queue: 0xFF,
                id: state.next_city_id as u8,
            });
            state.next_city_id += 1;

            // Remove settler
            units.retain(|u| u.id != uid);
        }
    } else if action >= PRODUCE_OFFSET && action < ACTION_SPACE_SIZE {
        // City production
        let local = action - PRODUCE_OFFSET;
        let city_idx = (local / NUM_UNIT_TYPES as u16) as usize;
        let unit_type = (local % NUM_UNIT_TYPES as u16) as u8;

        if let Some(city) = get_own_city_mut(&mut cities, current, city_idx) {
            if city.production_queue == 0xFF {
                city.production_queue = unit_type;
            }
        }
    }

    // Check victory conditions
    let p0_cities = cities.iter().filter(|c| c.player == 0).count();
    let p1_cities = cities.iter().filter(|c| c.player == 1).count();
    let p0_units = units.iter().filter(|u| u.player == 0).count();
    let p1_units = units.iter().filter(|u| u.player == 1).count();

    if p0_cities == 0 && p0_units == 0 {
        game_over = true;
        winner = Some(1);
    } else if p1_cities == 0 && p1_units == 0 {
        game_over = true;
        winner = Some(0);
    }

    // Check turn limit
    if state.turn_number >= state.turn_number {
        // (We check against the game's max_turns in the caller; for now just
        // encode the state. The caller handles truncation.)
    }

    // Re-encode state
    state.units = encode_units(&units);
    state.cities = encode_cities(&cities);

    // Recompute action mask for next player
    state.action_mask = compute_action_mask(
        &state.terrain,
        &state.units,
        &state.cities,
        board_size,
        state.current_player,
    );

    // Recompute tensors for AI
    state.spatial_tensor = compute_spatial_tensor(
        &state.terrain,
        &state.units,
        &state.cities,
        board_size,
        state.current_player,
    );
    state.vector_features = compute_vector_features(
        &state.units,
        &state.cities,
        state.turn_number,
        state.current_player,
    );

    (state, game_over, winner)
}

// ---------------------------------------------------------------------------
// City production processing (called on end_turn)
// ---------------------------------------------------------------------------

fn process_city_production(
    units: &mut Vec<DecodedUnit>,
    cities: &mut Vec<DecodedCity>,
    player: u8,
    terrain: &[u8],
    board_size: usize,
    next_unit_id: &mut u16,
) {
    let mut new_units: Vec<DecodedUnit> = Vec::new();

    for city in cities.iter_mut() {
        if city.player != player {
            continue;
        }
        if city.production_queue == 0xFF {
            continue;
        }

        // Accumulate production
        city.production += (city.population as u16) + 2;

        let unit_type_idx = city.production_queue as usize;
        let cost = if unit_type_idx < UNIT_COSTS.len() {
            UNIT_COSTS[unit_type_idx]
        } else {
            50
        };

        if city.production >= cost {
            // Spawn unit adjacent to city
            let spawn_pos = find_adjacent_valid(terrain, board_size, (city.x, city.y));
            let strength = if unit_type_idx < UNIT_STRENGTH.len() {
                UNIT_STRENGTH[unit_type_idx]
            } else {
                20
            };

            new_units.push(DecodedUnit {
                player: city.player,
                unit_type: city.production_queue,
                x: spawn_pos.0,
                y: spawn_pos.1,
                health: 100,
                moves_remaining: 0, // Just spawned, no moves this turn
                strength,
                id: *next_unit_id as u8,
            });
            *next_unit_id += 1;

            city.production = 0;
            city.production_queue = 0xFF;
        }
    }

    units.extend(new_units);
}

// ---------------------------------------------------------------------------
// Unit/city accessors
// ---------------------------------------------------------------------------

fn get_own_unit(units: &[DecodedUnit], player: u8, own_idx: usize) -> Option<&DecodedUnit> {
    units.iter()
        .filter(|u| u.player == player)
        .nth(own_idx)
}

fn get_own_unit_mut(units: &mut [DecodedUnit], player: u8, own_idx: usize) -> Option<&mut DecodedUnit> {
    units.iter_mut()
        .filter(|u| u.player == player)
        .nth(own_idx)
}

fn get_own_city_mut(cities: &mut [DecodedCity], player: u8, own_idx: usize) -> Option<&mut DecodedCity> {
    cities.iter_mut()
        .filter(|c| c.player == player)
        .nth(own_idx)
}

// ---------------------------------------------------------------------------
// Spatial tensor computation (13 channels * 8 * 8 = 832 f32 values)
// ---------------------------------------------------------------------------
// Channels (matching Python MicroCivEncoder):
//   0-2: Terrain (plains, hills, mountain) one-hot
//   3-5: Own units (warrior, settler, archer)
//   6-8: Enemy units (warrior, settler, archer)
//   9:   Own cities
//   10:  Enemy cities
//   11:  Visibility (simplified: 1.0 everywhere for now)
//   12:  Constant bias (1.0)

fn compute_spatial_tensor(
    terrain: &[u8],
    units_data: &[u8],
    cities_data: &[u8],
    board_size: u8,
    current_player: u8,
) -> Vec<u8> {
    let bs = board_size as usize;
    let channels = SPATIAL_CHANNELS as usize;
    let total_floats = channels * bs * bs;
    let mut tensor = vec![0.0f32; total_floats];

    // Helper to set channel value
    let set = |t: &mut Vec<f32>, ch: usize, x: usize, y: usize, val: f32| {
        let idx = ch * bs * bs + y * bs + x;
        if idx < t.len() {
            t[idx] = val;
        }
    };

    // Terrain channels (0-2)
    for y in 0..bs {
        for x in 0..bs {
            let tidx = y * bs + x;
            if tidx < terrain.len() {
                let tt = terrain[tidx];
                if tt == TerrainType::Plains as u8 {
                    set(&mut tensor, 0, x, y, 1.0);
                } else if tt == TerrainType::Hills as u8 {
                    set(&mut tensor, 1, x, y, 1.0);
                } else if tt == TerrainType::Mountain as u8 {
                    set(&mut tensor, 2, x, y, 1.0);
                }
            }
        }
    }

    // Unit channels (3-8)
    let units = decode_units(units_data);
    for unit in &units {
        let x = unit.x as usize;
        let y = unit.y as usize;
        if x < bs && y < bs {
            let is_own = unit.player == current_player;
            let base_ch = if is_own { 3 } else { 6 };
            let type_offset = (unit.unit_type as usize).min(2);
            set(&mut tensor, base_ch + type_offset, x, y, unit.health as f32 / 100.0);
        }
    }

    // City channels (9-10)
    let cities = decode_cities(cities_data);
    for city in &cities {
        let x = city.x as usize;
        let y = city.y as usize;
        if x < bs && y < bs {
            let ch = if city.player == current_player { 9 } else { 10 };
            set(&mut tensor, ch, x, y, 1.0);
        }
    }

    // Visibility (channel 11) - simplified: all visible
    for y in 0..bs {
        for x in 0..bs {
            set(&mut tensor, 11, x, y, 1.0);
        }
    }

    // Constant bias (channel 12)
    for y in 0..bs {
        for x in 0..bs {
            set(&mut tensor, 12, x, y, 1.0);
        }
    }

    // Convert to little-endian bytes
    let mut bytes = Vec::with_capacity(total_floats * 4);
    for val in &tensor {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

// ---------------------------------------------------------------------------
// Vector features computation (64 f32 values)
// ---------------------------------------------------------------------------

fn compute_vector_features(
    units_data: &[u8],
    cities_data: &[u8],
    turn: u16,
    current_player: u8,
) -> Vec<u8> {
    let dim = VECTOR_DIM as usize;
    let mut features = vec![0.0f32; dim];

    let units = decode_units(units_data);
    let cities = decode_cities(cities_data);

    let own_units: Vec<_> = units.iter().filter(|u| u.player == current_player).collect();
    let enemy_units: Vec<_> = units.iter().filter(|u| u.player != current_player).collect();
    let own_cities: Vec<_> = cities.iter().filter(|c| c.player == current_player).collect();
    let enemy_cities: Vec<_> = cities.iter().filter(|c| c.player != current_player).collect();

    // Feature encoding (indices 0-63):
    // 0: turn / max_turns (normalized)
    features[0] = turn as f32 / DEFAULT_MAX_TURNS as f32;
    // 1: current player (0 or 1)
    features[1] = current_player as f32;
    // 2: own unit count / max
    features[2] = own_units.len() as f32 / MAX_UNITS_PER_PLAYER as f32;
    // 3: enemy unit count / max
    features[3] = enemy_units.len() as f32 / MAX_UNITS_PER_PLAYER as f32;
    // 4: own city count / max
    features[4] = own_cities.len() as f32 / MAX_CITIES_PER_PLAYER as f32;
    // 5: enemy city count / max
    features[5] = enemy_cities.len() as f32 / MAX_CITIES_PER_PLAYER as f32;
    // 6: own warriors
    features[6] = own_units.iter().filter(|u| u.unit_type == 0).count() as f32 / 10.0;
    // 7: own settlers
    features[7] = own_units.iter().filter(|u| u.unit_type == 1).count() as f32 / 5.0;
    // 8: own archers
    features[8] = own_units.iter().filter(|u| u.unit_type == 2).count() as f32 / 10.0;
    // 9: average own unit health
    features[9] = if own_units.is_empty() {
        0.0
    } else {
        own_units.iter().map(|u| u.health as f32).sum::<f32>() / (own_units.len() as f32 * 100.0)
    };
    // 10: average enemy unit health
    features[10] = if enemy_units.is_empty() {
        0.0
    } else {
        enemy_units.iter().map(|u| u.health as f32).sum::<f32>() / (enemy_units.len() as f32 * 100.0)
    };
    // 11-15: own city populations
    for (i, city) in own_cities.iter().enumerate().take(5) {
        features[11 + i] = city.population as f32 / 10.0;
    }
    // 16: total own military strength
    features[16] = own_units.iter()
        .map(|u| u.strength as f32 * u.health as f32 / 100.0)
        .sum::<f32>() / 200.0;
    // 17: total enemy military strength
    features[17] = enemy_units.iter()
        .map(|u| u.strength as f32 * u.health as f32 / 100.0)
        .sum::<f32>() / 200.0;
    // 18: military advantage (own - enemy, normalized)
    features[18] = ((features[16] - features[17]) + 1.0) / 2.0;

    // Remaining features (19-63) are zero-padded for future use

    // Convert to bytes
    let mut bytes = Vec::with_capacity(dim * 4);
    for val in &features {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

// ---------------------------------------------------------------------------
// Action description (for human-readable move history)
// ---------------------------------------------------------------------------

fn describe_action(action: u16) -> String {
    if action == END_TURN_OFFSET {
        return "End turn".to_string();
    }

    let dir_names = ["NW", "N", "NE", "W", "E", "SW", "S", "SE"];

    if action >= MOVE_OFFSET && action < ATTACK_OFFSET {
        let local = action - MOVE_OFFSET;
        let unit_idx = local / 8;
        let dir_idx = (local % 8) as usize;
        return format!("Unit {} moves {}", unit_idx, dir_names[dir_idx]);
    }

    if action >= ATTACK_OFFSET && action < SKIP_OFFSET {
        let local = action - ATTACK_OFFSET;
        let unit_idx = local / 8;
        let dir_idx = (local % 8) as usize;
        return format!("Unit {} attacks {}", unit_idx, dir_names[dir_idx]);
    }

    if action >= SKIP_OFFSET && action < FOUND_CITY_OFFSET {
        let unit_idx = action - SKIP_OFFSET;
        return format!("Unit {} skips", unit_idx);
    }

    if action >= FOUND_CITY_OFFSET && action < PRODUCE_OFFSET {
        let unit_idx = action - FOUND_CITY_OFFSET;
        return format!("Unit {} founds city", unit_idx);
    }

    if action >= PRODUCE_OFFSET && action < ACTION_SPACE_SIZE {
        let local = action - PRODUCE_OFFSET;
        let city_idx = local / NUM_UNIT_TYPES as u16;
        let unit_type = local % NUM_UNIT_TYPES as u16;
        let type_names = ["warrior", "settler", "archer"];
        let name = if (unit_type as usize) < type_names.len() {
            type_names[unit_type as usize]
        } else {
            "unknown"
        };
        return format!("City {} produces {}", city_idx, name);
    }

    format!("Unknown action {}", action)
}

// ---------------------------------------------------------------------------
// AI turn queueing
// ---------------------------------------------------------------------------

fn queue_ai_turn(
    ctx: &ReducerContext,
    game_id: u64,
    player_slot: u8,
    model_version: &str,
) -> Result<(), String> {
    let now = ctx.timestamp;

    let state = ctx.db.game_state().game_id().find(game_id)
        .ok_or_else(|| "Game state not found for AI turn.".to_string())?;

    // Check if request already exists
    let existing = ctx.db.ai_turn_request().iter()
        .any(|r| r.game_id == game_id);

    if existing {
        // Already queued
        return Ok(());
    }

    ctx.db.ai_turn_request().insert(AiTurnRequest {
        request_id: 0,
        game_id,
        player_slot,
        model_version: model_version.to_string(),
        spatial_tensor: state.spatial_tensor.clone(),
        vector_features: state.vector_features.clone(),
        action_mask: state.action_mask.clone(),
        requested_at: now,
    });

    log::info!(
        "AI turn requested for game {} player {} (model={})",
        game_id, player_slot, model_version
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// ELO rating update
// ---------------------------------------------------------------------------

fn update_elo(ctx: &ReducerContext, player0_id: u64, player1_id: u64, result: u8) {
    let p0 = ctx.db.player().player_id().find(player0_id);
    let p1 = ctx.db.player().player_id().find(player1_id);

    if let (Some(p0), Some(p1)) = (p0, p1) {
        let (s0, s1) = match result {
            0 => (1.0, 0.0), // Player 0 wins
            1 => (0.0, 1.0), // Player 1 wins
            _ => (0.5, 0.5), // Draw
        };

        let e0 = 1.0 / (1.0 + 10.0_f64.powf((p1.elo_rating - p0.elo_rating) as f64 / 400.0));
        let e1 = 1.0 - e0;

        let new_elo0 = p0.elo_rating + (ELO_K_FACTOR * (s0 - e0)) as i32;
        let new_elo1 = p1.elo_rating + (ELO_K_FACTOR * (s1 - e1)) as i32;

        let (w0, l0, d0) = match result {
            0 => (1, 0, 0),
            1 => (0, 1, 0),
            _ => (0, 0, 1),
        };

        ctx.db.player().player_id().update(Player {
            elo_rating: new_elo0,
            games_played: p0.games_played + 1,
            games_won: p0.games_won + w0,
            games_lost: p0.games_lost + l0,
            games_drawn: p0.games_drawn + d0,
            ..p0
        });

        ctx.db.player().player_id().update(Player {
            elo_rating: new_elo1,
            games_played: p1.games_played + 1,
            games_won: p1.games_won + (1 - w0 - d0),
            games_lost: p1.games_lost + (1 - l0 - d0),
            games_drawn: p1.games_drawn + d0,
            ..p1
        });
    }
}

// ---------------------------------------------------------------------------
// Matchmaking
// ---------------------------------------------------------------------------

fn try_matchmake(ctx: &ReducerContext) -> Result<(), String> {
    // Collect all queue entries sorted by queue time
    let mut entries: Vec<MatchmakingQueue> = ctx.db.matchmaking_queue().iter().collect();

    if entries.len() < 2 {
        return Ok(());
    }

    // Sort by ELO for proximity matching
    entries.sort_by_key(|e| e.elo_rating);

    // Greedy pairing: match adjacent ELO entries with compatible modes
    let mut matched = Vec::new();
    let mut i = 0;
    while i + 1 < entries.len() {
        let a = &entries[i];
        let b = &entries[i + 1];

        // ELO difference threshold (widen over time if needed)
        let elo_diff = (a.elo_rating - b.elo_rating).unsigned_abs();
        if elo_diff <= 200 && a.desired_mode == b.desired_mode {
            matched.push((a.clone(), b.clone()));
            i += 2;
        } else {
            i += 1;
        }
    }

    // Create games for matched pairs
    for (a, b) in &matched {
        // Use combined player IDs as seed for reproducibility
        let seed = (a.player_id.wrapping_mul(31)) ^ b.player_id;

        let game = ctx.db.game().insert(Game {
            game_id: 0,
            status: GameStatus::InProgress as u8,
            mode: a.desired_mode,
            player0_id: a.player_id,
            player1_id: b.player_id,
            current_player: 0,
            turn_number: 0,
            max_turns: DEFAULT_MAX_TURNS,
            winner: 255,
            result: 255,
            map_seed: seed,
            board_size: BOARD_SIZE,
            created_at: ctx.timestamp,
            started_at: ctx.timestamp,
            finished_at: Timestamp::UNIX_EPOCH,
            tournament_id: 0,
            tournament_round: 0,
        });

        initialize_game_state(ctx, game.game_id, seed)?;

        // Remove from queue
        ctx.db.matchmaking_queue().entry_id().delete(a.entry_id);
        ctx.db.matchmaking_queue().entry_id().delete(b.entry_id);

        log::info!(
            "Matchmaking: player {} (ELO {}) vs player {} (ELO {}) -> game {}",
            a.player_id, a.elo_rating, b.player_id, b.elo_rating, game.game_id
        );

        // Check if player 0 is AI
        if let Some(p0) = ctx.db.player().player_id().find(a.player_id) {
            if p0.is_ai {
                let _ = queue_ai_turn(ctx, game.game_id, 0, &p0.ai_model_version);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tournament bracket advancement
// ---------------------------------------------------------------------------

fn advance_tournament_bracket(
    ctx: &ReducerContext,
    tournament_id: u64,
    game_id: u64,
    winner_id: Option<u64>,
) {
    let winner_pid = match winner_id {
        Some(id) => id,
        None => return, // Draw in tournament -- would need tiebreaker logic
    };

    // Find the bracket entry for this game
    let bracket_entries: Vec<_> = ctx.db.tournament_bracket().iter()
        .filter(|b| b.tournament_id == tournament_id && b.game_id == game_id)
        .collect();

    if let Some(entry) = bracket_entries.first() {
        // Update winner
        ctx.db.tournament_bracket().bracket_id().update(TournamentBracket {
            winner_id: winner_pid,
            ..entry.clone()
        });

        // Mark loser as eliminated
        let loser_id = if winner_pid == entry.player0_id {
            entry.player1_id
        } else {
            entry.player0_id
        };

        let loser_regs: Vec<_> = ctx.db.tournament_registration().iter()
            .filter(|r| r.tournament_id == tournament_id && r.player_id == loser_id)
            .collect();
        for reg in loser_regs {
            ctx.db.tournament_registration().registration_id().update(
                TournamentRegistration {
                    eliminated: true,
                    ..reg
                }
            );
        }

        // Check if round is complete
        let current_round = entry.round;
        let round_brackets: Vec<_> = ctx.db.tournament_bracket().iter()
            .filter(|b| b.tournament_id == tournament_id && b.round == current_round)
            .collect();

        let all_decided = round_brackets.iter().all(|b| b.winner_id != 0);

        if all_decided {
            // Check if tournament is over
            let tournament = ctx.db.tournament().tournament_id().find(tournament_id);
            if let Some(t) = tournament {
                if current_round >= t.total_rounds {
                    // Tournament complete
                    ctx.db.tournament().tournament_id().update(Tournament {
                        status: TournamentStatus::Completed as u8,
                        finished_at: ctx.timestamp,
                        ..t
                    });
                    log::info!("Tournament {} completed! Winner: player {}", tournament_id, winner_pid);
                } else {
                    // Advance to next round
                    let next_round = current_round + 1;
                    let winners: Vec<u64> = round_brackets.iter()
                        .map(|b| b.winner_id)
                        .collect();

                    let matches_next = winners.len() / 2;
                    for i in 0..matches_next {
                        ctx.db.tournament_bracket().insert(TournamentBracket {
                            bracket_id: 0,
                            tournament_id,
                            round: next_round,
                            match_index: i as u16,
                            player0_id: winners[i * 2],
                            player1_id: winners[i * 2 + 1],
                            game_id: 0,
                            winner_id: 0,
                        });
                    }

                    ctx.db.tournament().tournament_id().update(Tournament {
                        current_round: next_round,
                        ..t
                    });

                    log::info!(
                        "Tournament {} advancing to round {} ({} matches)",
                        tournament_id, next_round, matches_next
                    );
                }
            }
        }
    }
}
