-- ============================================================================
-- Main.lua - Entry point for AlphaZero AI Bridge
-- ============================================================================
-- Coordinates all modules and handles game events to provide AI control.
-- ============================================================================

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
}

-- Polling timer handle
local pollTimerHandle = nil

-- ----------------------------------------------------------------------------
-- Initialization
-- ----------------------------------------------------------------------------

-- Find the AI player (first human player we'll control)
local function findAIPlayer()
    local configuredId = AlphaZeroConfig.Get("AI_PLAYER_ID")
    if configuredId then
        return configuredId
    end

    -- Find first human player
    if PlayerManager then
        for i = 0, PlayerManager.GetWasEverAliveCount() - 1 do
            local player = Players[i]
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
    local valid, errors = AlphaZeroConfig.Validate()
    if not valid then
        if Log then
            for _, err in ipairs(errors) do
                Log.Error("Config error: %s", err)
            end
        end
        return false
    end

    -- Find AI player
    bridgeState.aiPlayerId = findAIPlayer()
    if Log then Log.Info("AI Player ID: %d", bridgeState.aiPlayerId) end

    -- Clear old files
    AlphaZeroFileIO.ClearAll()

    -- Write initial heartbeat
    AlphaZeroFileIO.WriteHeartbeat()

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

    if Log then Log.Info("Turn started for player %d", playerId) end

    bridgeState.isMyTurn = true
    bridgeState.actionsThisTurn = 0
    bridgeState.waitingForCommand = true

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
    if pollTimerHandle then
        return  -- Already polling
    end

    local interval = AlphaZeroConfig.Get("COMMAND_POLL_INTERVAL") or 0.01

    if Events and Events.GameCoreEventPublishComplete then
        -- Use game events for timing
        Events.GameCoreEventPublishComplete.Add(AlphaZeroBridge.PollCommands)
        pollTimerHandle = true
    else
        -- Fallback: would need timer implementation
        if Log then Log.Warn("No timer mechanism available") end
    end
end

-- Stop polling
function AlphaZeroBridge.StopPolling()
    if pollTimerHandle and Events then
        Events.GameCoreEventPublishComplete.Remove(AlphaZeroBridge.PollCommands)
    end
    pollTimerHandle = nil
end

-- Poll for and process commands
function AlphaZeroBridge.PollCommands()
    if not bridgeState.isMyTurn or not bridgeState.waitingForCommand then
        return
    end

    -- Write heartbeat periodically
    local now = os.time()
    if now - bridgeState.lastHeartbeatTime >= 1 then
        AlphaZeroFileIO.WriteHeartbeat()
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

-- Local player turn begin
local function onLocalPlayerTurnBegin()
    local localPlayer = Game.GetLocalPlayer()
    if localPlayer == bridgeState.aiPlayerId then
        AlphaZeroBridge.OnTurnStart(localPlayer)
    end
end

-- Local player turn end
local function onLocalPlayerTurnEnd()
    local localPlayer = Game.GetLocalPlayer()
    if localPlayer == bridgeState.aiPlayerId then
        AlphaZeroBridge.OnTurnEnd(localPlayer)
    end
end

-- Register event handlers
local function registerEvents()
    if Events then
        if Events.PlayerTurnActivated then
            Events.PlayerTurnActivated.Add(onPlayerTurnActivated)
        end

        if Events.PlayerTurnDeactivated then
            Events.PlayerTurnDeactivated.Add(onPlayerTurnDeactivated)
        end

        if Events.LoadGameViewStateDone then
            Events.LoadGameViewStateDone.Add(onLoadGameViewStateDone)
        end

        if Events.LocalPlayerTurnBegin then
            Events.LocalPlayerTurnBegin.Add(onLocalPlayerTurnBegin)
        end

        if Events.LocalPlayerTurnEnd then
            Events.LocalPlayerTurnEnd.Add(onLocalPlayerTurnEnd)
        end
    end

    if GameEvents then
        if GameEvents.PlayerTurnStarted then
            GameEvents.PlayerTurnStarted.Add(onPlayerTurnActivated)
        end

        if GameEvents.PlayerTurnStartComplete then
            GameEvents.PlayerTurnStartComplete.Add(onPlayerTurnActivated)
        end
    end
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

-- Reset bridge state
function AlphaZeroBridge.Reset()
    bridgeState.initialized = false
    bridgeState.aiPlayerId = nil
    bridgeState.isMyTurn = false
    bridgeState.waitingForCommand = false
    bridgeState.actionsThisTurn = 0

    AlphaZeroBridge.StopPolling()
    AlphaZeroFileIO.ClearAll()

    if Log then Log.Info("Bridge reset") end
end

-- ----------------------------------------------------------------------------
-- Entry Point
-- ----------------------------------------------------------------------------

-- Register events on load
registerEvents()

-- Try to initialize immediately if game is already loaded
if Game then
    AlphaZeroBridge.Initialize()
end

print("[AlphaZero] Main module loaded - Bridge ready")
print("[AlphaZero] Version: " .. (AlphaZeroConfig and AlphaZeroConfig.Get("VERSION") or "unknown"))
