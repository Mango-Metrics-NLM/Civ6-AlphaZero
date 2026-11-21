-- ============================================================================
-- Main.lua - Entry point for AlphaZero AI Bridge
-- ============================================================================
-- Coordinates all modules and handles game events to provide AI control.
-- Fixes: deduplicated events, proper polling handle, graceful shutdown, timeout.
-- ============================================================================

-- Load dependent modules via include() (required in AddUserInterfaces context).
-- Guarded so re-including this file (or loading alongside AlphaZeroMain.lua's
-- bundled copies) doesn't redefine each module's globals unnecessarily.
if not AlphaZeroConfig then include("Config") end
if not AlphaZeroLogger then include("Logger") end
if not AlphaZeroJson then include("JsonEncoder") end
if not AlphaZeroFileIO then include("FileIO") end
if not AlphaZeroStateExporter then include("StateExporter") end
if not AlphaZeroActionExecutor then include("ActionExecutor") end

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
    -- When the last command message was handled; the command window runs
    -- from the later of this and the turn's start.
    lastCommandTime = 0,
    -- Frames left before the state is exported after a command message; nil
    -- when no export is pending.
    exportAfterTicks = nil,
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

    -- Diagnostic: dump available APIs to Lua.log. Gated behind DEBUG log
    -- level -- this ran unconditionally on every init, spamming Lua.log and
    -- slowing the Python LogTailer. Only emit when LOG_LEVEL=DEBUG (numeric
    -- level 1; see the levels table in Config.lua's GetLogLevelNum()).
    if AlphaZeroConfig and AlphaZeroConfig.GetLogLevelNum and AlphaZeroConfig.GetLogLevelNum() <= 1 then
        local apis = {
            {"io", io},
            {"io.open", io and io.open},
            {"os", os},
            {"os.time", os and os.time},
            {"os.clock", os and os.clock},
            {"os.execute", os and os.execute},
            {"os.getenv", os and os.getenv},
            {"os.remove", os and os.remove},
            {"os.rename", os and os.rename},
            {"UI", UI},
            {"UI.GetAppSavegameDirectory", UI and UI.GetAppSavegameDirectory},
            {"UI.DataError", UI and UI.DataError},
            {"Modding", Modding},
            {"Modding.GetUserDataFolder", Modding and Modding.GetUserDataFolder},
            {"Game", Game},
            {"Players", Players},
            {"GameEvents", GameEvents},
            {"Events", Events},
            {"ContextPtr", ContextPtr},
            {"Network", Network},
            {"loadfile", loadfile},
            {"dofile", dofile},
            {"require", require},
            {"include", include},
        }
        print("[AlphaZero] === API AVAILABILITY DIAGNOSTIC ===")
        for _, entry in ipairs(apis) do
            local name, val = entry[1], entry[2]
            print(string.format("[AlphaZero] %s = %s", name, val and "AVAILABLE" or "nil"))
        end
        print("[AlphaZero] === END DIAGNOSTIC ===")
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

    -- Write the game's rule tables once per load, for the Python side to
    -- capture (GM-B4). The tables do not change during a game. Never fatal:
    -- the bridge works without the snapshot.
    if AlphaZeroGameInfo and AlphaZeroConfig and AlphaZeroConfig.IsEnabled("GAMEINFO_EXPORT") then
        local ok, written, err = pcall(AlphaZeroGameInfo.ExportAndWrite)
        if not ok or not written then
            if Log then
                Log.Warn("GameInfo snapshot not written: %s", tostring(ok and err or written))
            end
        end
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
    bridgeState.turnStartTime = (os and os.time) and os.time() or 0
    bridgeState.lastCommandTime = bridgeState.turnStartTime
    bridgeState.exportAfterTicks = nil

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

    -- The state after a command message is exported some frames later, once
    -- the game has carried the requests out. Until then no command is read:
    -- the next one should be chosen on the new state.
    if bridgeState.exportAfterTicks then
        bridgeState.exportAfterTicks = bridgeState.exportAfterTicks - 1
        if bridgeState.exportAfterTicks > 0 then
            return
        end
        bridgeState.exportAfterTicks = nil
        local exported, exportErr = AlphaZeroStateExporter.ExportAndWrite(bridgeState.aiPlayerId)
        if not exported then
            if Log then Log.Error("Failed to export state after a command: %s", tostring(exportErr)) end
            bridgeState.lastError = exportErr
        end
        return
    end

    -- Command timeout check: the window runs from the last command handled,
    -- so a turn of many commands is not cut off thirty seconds after it began.
    local timeout = (AlphaZeroConfig and AlphaZeroConfig.Get("COMMAND_TIMEOUT")) or 30.0
    local since = math.max(bridgeState.turnStartTime or 0, bridgeState.lastCommandTime or 0)
    local elapsed = ((os and os.time) and os.time() or 0) - since
    if elapsed > timeout then
        if Log then Log.Warn("Command timeout after %ds, stopping poll", elapsed) end
        bridgeState.waitingForCommand = false
        return
    end

    -- Write heartbeat periodically, every HEARTBEAT_INTERVAL seconds. A value
    -- that is not a positive number (a typo in the override file) falls back to
    -- one second: an error here would stop command polling.
    local heartbeatInterval = tonumber(AlphaZeroConfig and AlphaZeroConfig.Get("HEARTBEAT_INTERVAL"))
    if not heartbeatInterval or heartbeatInterval <= 0 then
        heartbeatInterval = 1.0
    end
    local now = (os and os.time) and os.time() or 0
    if now - bridgeState.lastHeartbeatTime >= heartbeatInterval then
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

-- Commands that can answer what blocks the end of a turn.
local UNBLOCKING_ACTIONS = {
    research_tech = true, research_civic = true, city_produce = true,
    unit_skip = true, unit_fortify = true, unit_sleep = true,
}

-- Process a command message
function AlphaZeroBridge.ProcessCommands(commandMessage)
    if Log then Log.Info("Processing commands: msg_id=%s", tostring(commandMessage.msg_id)) end

    -- A cap on commands per turn, so a loop that keeps sending a command the
    -- game ignores stops. Past it only END_TURN is carried out, with one
    -- exception: while the game refuses to end the turn (a research,
    -- production or unit-orders choice is open), the commands that can answer
    -- that choice are still carried out, or the cap itself would trap the
    -- turn. That allowance ends at twice the cap.
    local commands = commandMessage.commands or {}
    local cap = tonumber(AlphaZeroConfig and AlphaZeroConfig.Get("MAX_COMMANDS_PER_TURN")) or 200
    local onlyEndTurn, onlyUnblocking = #commands > 0, #commands > 0
    for _, command in ipairs(commands) do
        if command.action_type ~= "end_turn" then
            onlyEndTurn = false
            if not UNBLOCKING_ACTIONS[command.action_type] then onlyUnblocking = false end
        end
    end

    -- The count the turn would reach with this message: a batch may not carry
    -- it past a limit either.
    local projected = bridgeState.actionsThisTurn + #commands
    local overCap = projected > cap and not onlyEndTurn
    if overCap and onlyUnblocking and projected <= 2 * cap then
        local blocker = nil
        if AlphaZeroStateExporter and AlphaZeroStateExporter.GetEndTurnBlocker then
            local ok, found = pcall(AlphaZeroStateExporter.GetEndTurnBlocker, bridgeState.aiPlayerId)
            blocker = ok and found or nil
        end
        if blocker then overCap = false end
    end

    local results, errorCode
    if overCap then
        local message = string.format(
            "%d commands already handled this turn (the cap is %d); only end_turn is accepted",
            bridgeState.actionsThisTurn, cap)
        if Log then Log.Warn(message) end
        results = { msg_id = commandMessage.msg_id, success = false, executed = {}, errors = { message } }
        errorCode = "COMMAND_CAP_REACHED"
    else
        -- Only a message that is carried out keeps the command window open.
        bridgeState.lastCommandTime = (os and os.time) and os.time() or 0
        results = AlphaZeroActionExecutor.ExecuteCommands(bridgeState.aiPlayerId, commandMessage)
        errorCode = "EXECUTION_ERROR"
        bridgeState.actionsThisTurn = bridgeState.actionsThisTurn + #commands
    end

    -- Write acknowledgment
    local ackSuccess = AlphaZeroFileIO.WriteAck(
        commandMessage.msg_id or 0,
        results.success,
        results.success and nil or errorCode,
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
        -- Export the new state some frames from now, not in this one: the
        -- game has not carried the requests out yet.
        local delay = tonumber(AlphaZeroConfig and AlphaZeroConfig.Get("STATE_EXPORT_DELAY_TICKS")) or 2
        if delay < 1 then delay = 1 end
        bridgeState.exportAfterTicks = math.floor(delay)
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
