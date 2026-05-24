-- ============================================================================
-- Main.lua - Entry point for AlphaZero AI Bridge
-- ============================================================================
-- Coordinates all modules and handles game events to provide AI control.
-- Fixes: deduplicated events, proper polling handle, graceful shutdown, timeout.
-- ============================================================================

-- Load dependent modules via include() (required in AddUserInterfaces context)
include("Config")
include("Logger")
include("JsonEncoder")
include("FileIO")
include("StateExporter")
include("ActionExecutor")

AlphaZeroBridge = AlphaZeroBridge or {}


local Log = nil  -- Will be set after Logger initializes

-- Bridge state
local bridgeState = {
    initialized = false,
    aiPlayerId = nil,
    isMyTurn = false,
    waitingForCommand = false,
    lastHeartbeatTime = 0,
    actionsThisTurn = 0,
    lastError = nil,
    turnStartTime = 0,
}

-- Polling state (use explicit callback reference for reliable removal)
local isPolling = false
local pollCallback = nil  -- Will be set to AlphaZeroBridge.PollCommands after definition

-- Track registered events for graceful shutdown
local registeredEvents = {}

-- ----------------------------------------------------------------------------
-- Initialization
-- ----------------------------------------------------------------------------

-- Find the AI player (first human player we'll control)
local function findAIPlayer()
    local configuredId = AlphaZeroConfig and AlphaZeroConfig.Get("AI_PLAYER_ID")
    if configuredId then
        return configuredId
    end

    -- Find first human player
    if PlayerManager and PlayerManager.GetWasEverAliveCount then
        for i = 0, PlayerManager.GetWasEverAliveCount() - 1 do
            local player = Players and Players[i]
            if player and player:IsHuman() and player:IsAlive() then
                return i
            end
        end
    end

    return 0  -- Fallback to player 0
end

-- Initialize the bridge
function AlphaZeroBridge.Initialize()
    if bridgeState.initialized then
        return true
    end

    -- Initialize all modules
    if AlphaZeroConfig then
        AlphaZeroConfig.Init()
    end

    if AlphaZeroLogger then
        AlphaZeroLogger.Init()
        Log = AlphaZeroLogger.CreateLogger("Main")
    end

    if AlphaZeroJson then
        AlphaZeroJson.Init()
    end

    if AlphaZeroFileIO then
        AlphaZeroFileIO.Init()
    end

    if AlphaZeroStateExporter then
        AlphaZeroStateExporter.Init()
    end

    if AlphaZeroActionExecutor then
        AlphaZeroActionExecutor.Init()
    end

    -- Validate configuration
    if AlphaZeroConfig and AlphaZeroConfig.Validate then
        local valid, errors = AlphaZeroConfig.Validate()
        if not valid then
            if Log then
                for _, err in ipairs(errors) do
                    Log.Error("Config error: %s", err)
                end
            end
            return false
        end
    end

    -- Find AI player
    bridgeState.aiPlayerId = findAIPlayer()
    if Log then Log.Info("AI Player ID: %d", bridgeState.aiPlayerId) end

    -- Clear old files
    if AlphaZeroFileIO then
        AlphaZeroFileIO.ClearAll()
        AlphaZeroFileIO.WriteHeartbeat()
    end

    bridgeState.initialized = true
    if Log then Log.Info("AlphaZero Bridge initialized successfully") end

    return true
end

-- ----------------------------------------------------------------------------
-- Turn Management
-- ----------------------------------------------------------------------------

-- Called when the AI player's turn starts
function AlphaZeroBridge.OnTurnStart(playerId)
    if playerId ~= bridgeState.aiPlayerId then
        return
    end

    -- Re-entrance guard: prevent double-firing from multiple event sources
    if bridgeState.isMyTurn then
        return
    end

    if Log then Log.Info("Turn started for player %d", playerId) end

    bridgeState.isMyTurn = true
    bridgeState.actionsThisTurn = 0
    bridgeState.waitingForCommand = true
    bridgeState.turnStartTime = os.time()

    -- Export and write current state
    local success, err = AlphaZeroStateExporter.ExportAndWrite(playerId)
    if not success then
        if Log then Log.Error("Failed to export state: %s", tostring(err)) end
        bridgeState.lastError = err
    end

    -- Start polling for commands
    AlphaZeroBridge.StartPolling()
end

-- Called when the AI player's turn ends
function AlphaZeroBridge.OnTurnEnd(playerId)
    if playerId ~= bridgeState.aiPlayerId then
        return
    end

    if Log then Log.Info("Turn ended for player %d (actions: %d)", playerId, bridgeState.actionsThisTurn) end

    bridgeState.isMyTurn = false
    bridgeState.waitingForCommand = false

    -- Stop polling
    AlphaZeroBridge.StopPolling()
end

-- ----------------------------------------------------------------------------
-- Command Polling
-- ----------------------------------------------------------------------------

-- Start polling for commands
function AlphaZeroBridge.StartPolling()
    if isPolling then
        return  -- Already polling
    end

    if Events and Events.GameCoreEventPublishComplete then
        -- Ensure pollCallback is set
        if not pollCallback then
            pollCallback = AlphaZeroBridge.PollCommands
        end
        Events.GameCoreEventPublishComplete.Add(pollCallback)
        isPolling = true
    else
        if Log then Log.Warn("No timer mechanism available for polling") end
    end
end

-- Stop polling
function AlphaZeroBridge.StopPolling()
    if isPolling and Events and Events.GameCoreEventPublishComplete and pollCallback then
        Events.GameCoreEventPublishComplete.Remove(pollCallback)
    end
    isPolling = false
end

-- Poll for and process commands
function AlphaZeroBridge.PollCommands()
    if not bridgeState.isMyTurn or not bridgeState.waitingForCommand then
        return
    end

    -- Command timeout check
    local timeout = (AlphaZeroConfig and AlphaZeroConfig.Get("COMMAND_TIMEOUT")) or 30.0
    local elapsed = os.time() - bridgeState.turnStartTime
    if elapsed > timeout then
        if Log then Log.Warn("Command timeout after %ds, stopping poll", elapsed) end
        bridgeState.waitingForCommand = false
        return
    end

    -- Write heartbeat periodically
    local now = os.time()
    if now - bridgeState.lastHeartbeatTime >= 1 then
        if AlphaZeroFileIO then
            AlphaZeroFileIO.WriteHeartbeat()
        end
        bridgeState.lastHeartbeatTime = now
    end

    -- Read commands
    local commandMessage, err = AlphaZeroFileIO.ReadCommands()
    if err then
        if Log then Log.Error("Error reading commands: %s", err) end
        return
    end

    if not commandMessage then
        return  -- No new commands
    end

    -- Process commands
    AlphaZeroBridge.ProcessCommands(commandMessage)
end

-- Process a command message
function AlphaZeroBridge.ProcessCommands(commandMessage)
    if Log then Log.Info("Processing commands: msg_id=%s", tostring(commandMessage.msg_id)) end

    local results = AlphaZeroActionExecutor.ExecuteCommands(
        bridgeState.aiPlayerId,
        commandMessage
    )

    -- Count actions
    bridgeState.actionsThisTurn = bridgeState.actionsThisTurn + #(commandMessage.commands or {})

    -- Write acknowledgment
    local ackSuccess = AlphaZeroFileIO.WriteAck(
        commandMessage.msg_id or 0,
        results.success,
        results.success and nil or "EXECUTION_ERROR",
        results.success and nil or table.concat(results.errors, "; "),
        results.executed
    )

    if not ackSuccess then
        if Log then Log.Error("Failed to write ack") end
    end

    -- Check if turn ended
    local endTurnExecuted = false
    for _, exec in ipairs(results.executed) do
        if exec.action_type == "end_turn" and exec.success then
            endTurnExecuted = true
            break
        end
    end

    if endTurnExecuted then
        bridgeState.waitingForCommand = false
    else
        -- Export new state after actions
        AlphaZeroStateExporter.ExportAndWrite(bridgeState.aiPlayerId)
    end
end

-- ----------------------------------------------------------------------------
-- Event Handlers
-- ----------------------------------------------------------------------------

-- Player turn started event
local function onPlayerTurnActivated(playerId, isFirstTime)
    if bridgeState.initialized then
        AlphaZeroBridge.OnTurnStart(playerId)
    end
end

-- Player turn ended event
local function onPlayerTurnDeactivated(playerId)
    if bridgeState.initialized then
        AlphaZeroBridge.OnTurnEnd(playerId)
    end
end

-- Game load complete
local function onLoadGameViewStateDone()
    if Log then Log.Info("Game view state loaded") end
    AlphaZeroBridge.Initialize()
end

-- Local player turn begin (single-player fallback)
local function onLocalPlayerTurnBegin()
    if not Game then return end
    local localPlayer = Game.GetLocalPlayer()
    if localPlayer == bridgeState.aiPlayerId then
        AlphaZeroBridge.OnTurnStart(localPlayer)
    end
end

-- Local player turn end (single-player fallback)
local function onLocalPlayerTurnEnd()
    if not Game then return end
    local localPlayer = Game.GetLocalPlayer()
    if localPlayer == bridgeState.aiPlayerId then
        AlphaZeroBridge.OnTurnEnd(localPlayer)
    end
end

-- Register event handlers (deduplicated: no GameEvents duplicates)
local function registerEvents()
    -- Clear any previous tracking
    registeredEvents = {}

    if Events then
        -- Primary turn events (UI context)
        if Events.PlayerTurnActivated then
            Events.PlayerTurnActivated.Add(onPlayerTurnActivated)
            table.insert(registeredEvents, {event = Events.PlayerTurnActivated, handler = onPlayerTurnActivated})
        end

        if Events.PlayerTurnDeactivated then
            Events.PlayerTurnDeactivated.Add(onPlayerTurnDeactivated)
            table.insert(registeredEvents, {event = Events.PlayerTurnDeactivated, handler = onPlayerTurnDeactivated})
        end

        if Events.LoadGameViewStateDone then
            Events.LoadGameViewStateDone.Add(onLoadGameViewStateDone)
            table.insert(registeredEvents, {event = Events.LoadGameViewStateDone, handler = onLoadGameViewStateDone})
        end

        -- Single-player fallback events
        if Events.LocalPlayerTurnBegin then
            Events.LocalPlayerTurnBegin.Add(onLocalPlayerTurnBegin)
            table.insert(registeredEvents, {event = Events.LocalPlayerTurnBegin, handler = onLocalPlayerTurnBegin})
        end

        if Events.LocalPlayerTurnEnd then
            Events.LocalPlayerTurnEnd.Add(onLocalPlayerTurnEnd)
            table.insert(registeredEvents, {event = Events.LocalPlayerTurnEnd, handler = onLocalPlayerTurnEnd})
        end
    end

    -- REMOVED: GameEvents.PlayerTurnStarted and GameEvents.PlayerTurnStartComplete
    -- These fire in gameplay context alongside the UI events above, causing triple-fire.
    -- The re-entrance guard in OnTurnStart provides additional safety.
end

-- Unregister all event listeners (for graceful shutdown)
local function unregisterEvents()
    for _, entry in ipairs(registeredEvents) do
        if entry.event and entry.handler then
            local ok = pcall(function() entry.event.Remove(entry.handler) end)
            if not ok and Log then
                Log.Debug("Could not remove event handler during shutdown")
            end
        end
    end
    registeredEvents = {}
end

-- ----------------------------------------------------------------------------
-- Debug/Status API
-- ----------------------------------------------------------------------------

-- Get bridge status
function AlphaZeroBridge.GetStatus()
    return {
        initialized = bridgeState.initialized,
        aiPlayerId = bridgeState.aiPlayerId,
        isMyTurn = bridgeState.isMyTurn,
        waitingForCommand = bridgeState.waitingForCommand,
        actionsThisTurn = bridgeState.actionsThisTurn,
        isPolling = isPolling,
        lastError = bridgeState.lastError,
    }
end

-- Manually trigger state export (for debugging)
function AlphaZeroBridge.ExportState()
    if bridgeState.aiPlayerId then
        return AlphaZeroStateExporter.ExportAndWrite(bridgeState.aiPlayerId)
    end
    return false, "Not initialized"
end

-- Manually trigger command poll (for debugging)
function AlphaZeroBridge.ManualPoll()
    AlphaZeroBridge.PollCommands()
end

-- Reset bridge state (graceful shutdown)
function AlphaZeroBridge.Reset()
    -- Unregister all event listeners
    unregisterEvents()

    -- Stop polling
    AlphaZeroBridge.StopPolling()

    -- Reset state
    bridgeState.initialized = false
    bridgeState.aiPlayerId = nil
    bridgeState.isMyTurn = false
    bridgeState.waitingForCommand = false
    bridgeState.actionsThisTurn = 0
    bridgeState.turnStartTime = 0

    if AlphaZeroFileIO then
        AlphaZeroFileIO.ClearAll()
    end

    if Log then Log.Info("Bridge reset") end
end

-- ----------------------------------------------------------------------------
-- Entry Point
-- ----------------------------------------------------------------------------

-- Set up pollCallback reference
pollCallback = AlphaZeroBridge.PollCommands

-- Register events on load
registerEvents()

-- Try to initialize immediately if game is already loaded
if Game then
    AlphaZeroBridge.Initialize()
end

print("[AlphaZero] Main module loaded - Bridge ready")
print("[AlphaZero] Version: " .. (AlphaZeroConfig and AlphaZeroConfig.Get("VERSION") or "unknown"))
