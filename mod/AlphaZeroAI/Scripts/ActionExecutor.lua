-- ============================================================================
-- ActionExecutor.lua - Command execution for AlphaZero AI Bridge
-- ============================================================================
-- Executes commands received from the Python agent.
-- Uses safe accessors and correct Civ6 API calls (RequestOperation pattern).
-- ============================================================================

AlphaZeroActionExecutor = AlphaZeroActionExecutor or {}

local Log = nil  -- Will be set after Logger loads

-- Action type constants (must match Python ActionType enum exactly).
-- The action-mask invariant test (tests/unit/test_bridge_action_invariant.py)
-- asserts at CI time that this table and Python ActionType cover the same
-- set of values; do NOT add to one side without the other.
local ACTION_TYPES = {
    END_TURN = "end_turn",
    UNIT_MOVE = "unit_move",
    UNIT_ATTACK = "unit_attack",
    UNIT_RANGED_ATTACK = "unit_ranged_attack",
    UNIT_FORTIFY = "unit_fortify",
    UNIT_SKIP = "unit_skip",
    UNIT_SLEEP = "unit_sleep",
    UNIT_PROMOTE = "unit_promote",
    UNIT_DELETE = "unit_delete",
    UNIT_FOUND_CITY = "unit_found_city",
    UNIT_BUILD = "unit_build",
    CITY_PRODUCE = "city_produce",
    CITY_BUY = "city_buy",
    CITY_CITIZEN = "city_citizen",
    RESEARCH_TECH = "research_tech",
    RESEARCH_CIVIC = "research_civic",
    DIPLOMACY = "diplomacy",
}

-- Error codes
local ERROR_CODES = {
    INVALID_COMMAND = "INVALID_COMMAND",
    UNIT_NOT_FOUND = "UNIT_NOT_FOUND",
    CITY_NOT_FOUND = "CITY_NOT_FOUND",
    INVALID_TARGET = "INVALID_TARGET",
    NOT_ENOUGH_MOVEMENT = "NOT_ENOUGH_MOVEMENT",
    EXECUTION_FAILED = "EXECUTION_FAILED",
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED",
    -- The game will not end the turn until a pending choice is made. Not a
    -- malformed command: the agent clears the blocker and asks again.
    END_TURN_BLOCKED = "END_TURN_BLOCKED",
    -- The game's own check said the unit or city cannot start this now.
    OPERATION_REFUSED = "OPERATION_REFUSED",
    -- The move or attack would put the player at war. The mod never does that
    -- as a side effect; declaring war is the diplomacy command's business.
    WOULD_START_WAR = "WOULD_START_WAR",
    -- A diplomacy command whose setting (ALLOW_MAKE_PEACE, ALLOW_DENOUNCE) is
    -- not turned on. Nothing was sent to the game.
    DIPLOMACY_TURNED_OFF = "DIPLOMACY_TURNED_OFF",
}

-- ----------------------------------------------------------------------------
-- Reusable Safe Accessors
-- ----------------------------------------------------------------------------
-- These functions validate the entire access chain before returning an object.
-- Returns (object, nil, nil) on success or (nil, errorCode, message) on failure.

local function safeGetUnit(playerId, unitId)
    if not Players then
        return nil, ERROR_CODES.EXECUTION_FAILED, "Players table not available"
    end
    local player = Players[playerId]
    if not player then
        return nil, ERROR_CODES.EXECUTION_FAILED, "Player not found: " .. tostring(playerId)
    end
    local units = player:GetUnits()
    if not units then
        return nil, ERROR_CODES.EXECUTION_FAILED, "Units not available for player " .. tostring(playerId)
    end
    local unit = units:FindID(unitId)
    if not unit then
        return nil, ERROR_CODES.UNIT_NOT_FOUND, "Unit not found: " .. tostring(unitId)
    end
    return unit, nil, nil
end

local function safeGetCity(playerId, cityId)
    if not Players then
        return nil, ERROR_CODES.EXECUTION_FAILED, "Players table not available"
    end
    local player = Players[playerId]
    if not player then
        return nil, ERROR_CODES.EXECUTION_FAILED, "Player not found: " .. tostring(playerId)
    end
    local cities = player:GetCities()
    if not cities then
        return nil, ERROR_CODES.EXECUTION_FAILED, "Cities not available for player " .. tostring(playerId)
    end
    local city = cities:FindID(cityId)
    if not city then
        return nil, ERROR_CODES.CITY_NOT_FOUND, "City not found: " .. tostring(cityId)
    end
    return city, nil, nil
end

-- ----------------------------------------------------------------------------
-- Action Execution Functions
-- ----------------------------------------------------------------------------

-- End the current player's turn.
-- The request is refused, and nothing is sent to the game, while a
-- notification blocks ending the turn (an unanswered research, production, or
-- unit-orders choice): calling ACTION_ENDTURN while a blocker is active
-- deactivates the player's turn without advancing the turn counter (LF-H1).
-- UI.CanEndTurn only guards forced end-turn fallback and returns true even when
-- blockers exist in the live UI, so NotificationManager / GetEndTurnBlocker
-- is the sole end-turn blocking authority.
local function executeEndTurn(playerId, command)
    if Log then Log.Info("Executing: END_TURN") end

    -- When the game cannot be asked what blocks the turn, the turn is treated
    -- as blocked: nothing is sent.
    local blocker, blockerMessage, unknown = nil, nil, true
    if AlphaZeroStateExporter and AlphaZeroStateExporter.GetEndTurnBlocker then
        blocker, blockerMessage, unknown = AlphaZeroStateExporter.GetEndTurnBlocker(playerId)
    end
    if unknown and not blocker then
        blocker = "ENDTURN_BLOCKING_NOT_ASKABLE"
        blockerMessage = "the game could not be asked what blocks the end of turn"
    end

    -- ACTION_ENDTURN ends the LOCAL player's turn, whoever was asked about
    -- above. If the bridge's player is someone else, the blocker check was
    -- made for the wrong player: refuse.
    local okLocal, localId = pcall(function() return Game.GetLocalPlayer() end)
    if okLocal and type(localId) == "number" and localId >= 0 and localId ~= playerId then
        local message = string.format(
            "End turn refused: the bridge plays player %s and the local player is %s",
            tostring(playerId), tostring(localId))
        if Log then Log.Warn(message) end
        return false, ERROR_CODES.EXECUTION_FAILED, message
    end

    if blocker then
        local message = "End turn blocked: " .. tostring(blocker)
        if blockerMessage then
            message = message .. " (" .. tostring(blockerMessage) .. ")"
        end
        if Log then Log.Warn(message) end
        return false, ERROR_CODES.END_TURN_BLOCKED, message
    end

    -- In Civ6, ending turn is handled through the UI
    if UI and UI.RequestAction and ActionTypes then
        UI.RequestAction(ActionTypes.ACTION_ENDTURN)
        return true, nil, "End turn requested"
    end

    return false, ERROR_CODES.EXECUTION_FAILED,
        "End turn unavailable: no UI.RequestAction in this Lua context"
end

-- ----------------------------------------------------------------------------
-- Unit actions, requested the way the game's own UI requests them
-- ----------------------------------------------------------------------------
-- The game's unit panel does not hard-code operations. It walks
-- GameInfo.UnitOperations and GameInfo.UnitCommands, uses each row's Hash,
-- asks the game whether the unit can start it, and only then requests it.
-- Every unit action here goes the same way, by the row's type name, so a name
-- this version of the game does not have answers with a refusal that names it
-- instead of a request the game ignores. Not verified in a real game: the
-- names for skip and sleep are the least certain.

-- True while AlphaZeroActionExecutor.Check runs: every check is made and
-- nothing is requested, so "what the exporter offers" and "what the executor
-- carries out" are one piece of code and cannot drift apart.
local dryRun = false

-- The Hash and kind ("operation" / "command") of the row named `typeName`.
-- Found rows are remembered: the game's tables do not change while it runs,
-- and the exporter asks for the same few names for every unit and tile.
local actionRows = {}
local function findActionRow(typeName)
    local known = actionRows[typeName]
    if known then
        return known.hash, known.kind
    end
    local sources = {
        { table = "UnitOperations", column = "OperationType", kind = "operation" },
        { table = "UnitCommands", column = "CommandType", kind = "command" },
    }
    for _, source in ipairs(sources) do
        local gameTable = GameInfo and GameInfo[source.table]
        if gameTable then
            local found = nil
            pcall(function()
                for row in gameTable() do
                    if row[source.column] == typeName then
                        found = row
                        break
                    end
                end
            end)
            if found and found.Hash ~= nil then
                actionRows[typeName] = { hash = found.Hash, kind = source.kind }
                return found.Hash, source.kind
            end
        end
    end
    return nil, nil
end

-- The game's reasons for a refusal, as one line, or nil.
local function failureReasons(results, key)
    if type(results) ~= "table" or key == nil then return nil end
    local reasons = results[key]
    if type(reasons) ~= "table" or #reasons == 0 then return nil end
    local parts = {}
    for _, reason in ipairs(reasons) do
        -- The game gives text keys; show the text where it can be looked up.
        local text = tostring(reason)
        if Locale and Locale.Lookup then
            local ok, shown = pcall(Locale.Lookup, reason)
            if ok and type(shown) == "string" and shown ~= "" then text = shown end
        end
        table.insert(parts, text)
    end
    return table.concat(parts, "; ")
end

-- Ask the game's strict check and, if it agrees, request the action.
-- `params` is nil for an action without a target. Returns true, or false with
-- an error code and message. A request is fire-and-forget: true means the
-- game was asked, not that anything has happened yet.
local function requestUnitAction(unit, typeName, params, label)
    if not UnitManager then
        return false, ERROR_CODES.EXECUTION_FAILED,
            label .. " unavailable: no UnitManager in this Lua context"
    end
    local hash, kind = findActionRow(typeName)
    if hash == nil then
        return false, ERROR_CODES.EXECUTION_FAILED,
            string.format("%s: the game has no operation or command named %s", label, typeName)
    end

    local canStart, request, reasonsKey
    if kind == "operation" then
        canStart, request = UnitManager.CanStartOperation, UnitManager.RequestOperation
        reasonsKey = UnitOperationResults and UnitOperationResults.FAILURE_REASONS
    else
        canStart, request = UnitManager.CanStartCommand, UnitManager.RequestCommand
        -- The game's unit panel reads a command's reasons under the
        -- operation results key.
        reasonsKey = (UnitOperationResults and UnitOperationResults.FAILURE_REASONS)
            or (UnitCommandResults and UnitCommandResults.FAILURE_REASONS)
    end
    if not canStart or not request then
        return false, ERROR_CODES.EXECUTION_FAILED,
            string.format("%s unavailable: UnitManager cannot check or request a unit %s", label, kind)
    end

    -- The strict form: with a target, the parameters; without one, `false`
    -- ("for real", where `true` is the panel's loose "could it ever").
    local ok, allowed, results
    if kind == "command" then
        ok, allowed, results = pcall(canStart, unit, hash, false, true)
    elseif params then
        -- `true` after the parameters asks for results (the reasons).
        ok, allowed, results = pcall(canStart, unit, hash, nil, params, true)
    else
        ok, allowed, results = pcall(canStart, unit, hash, nil, false,
            OperationResultsTypes and OperationResultsTypes.NO_TARGETS)
    end
    if not ok then
        return false, ERROR_CODES.EXECUTION_FAILED,
            label .. ": the game's check raised: " .. tostring(allowed)
    end
    if allowed ~= true then
        local reason = failureReasons(results, reasonsKey)
        return false, ERROR_CODES.OPERATION_REFUSED,
            label .. " refused by the game" .. (reason and (": " .. reason) or "")
    end

    if dryRun then
        return true, nil, nil
    end
    local requested, err
    if params then
        requested, err = pcall(request, unit, hash, params)
    else
        requested, err = pcall(request, unit, hash)
    end
    if not requested then
        return false, ERROR_CODES.EXECUTION_FAILED, label .. ": the request raised: " .. tostring(err)
    end
    return true, nil, nil
end

-- Whether the player can see the plot. True when the game cannot say.
local function plotIsVisible(playerId, x, y)
    local visible = true
    pcall(function()
        local visibility = PlayersVisibility and PlayersVisibility[playerId]
        if visibility and visibility.IsVisible then
            visible = visibility:IsVisible(x, y) == true
        end
    end)
    return visible
end

-- Refuse anything that would put the player at war: nil when it would not,
-- else an error code and message. When the game cannot be asked, the answer
-- is a refusal too: the mod does not attack on a guess.
local warAnswerReported = false

local function warRefusal(unit, x, y)
    if not CombatManager or not CombatManager.IsAttackChangeWarState then
        return ERROR_CODES.EXECUTION_FAILED,
            "Cannot ask the game whether this would start a war (no CombatManager)"
    end
    local ok, results = pcall(function()
        return CombatManager.IsAttackChangeWarState(unit:GetComponentID(), x, y)
    end)
    if not ok then
        return ERROR_CODES.EXECUTION_FAILED,
            "Asking the game whether this would start a war raised: " .. tostring(results)
    end
    if results == nil then
        return nil, nil
    end
    if type(results) ~= "table" then
        -- The game's UI reads a list of players. Anything else is an answer
        -- this code does not understand, and "not understood" is not "no war".
        -- Said once per session with print, because the offer check runs with
        -- the log off: otherwise an agent that is offered no move says nothing.
        if not warAnswerReported then
            warAnswerReported = true
            print("[AlphaZero] WARN: IsAttackChangeWarState answered " .. type(results)
                .. " (" .. tostring(results) .. "); moves and attacks are refused until this is understood")
        end
        return ERROR_CODES.EXECUTION_FAILED,
            "The game's answer on whether this would start a war could not be read: "
            .. tostring(results)
    end
    local _, first = next(results)
    if first ~= nil then
        return ERROR_CODES.WOULD_START_WAR,
            string.format("Refused: (%d,%d) would start a war with player %s", x, y,
                tostring(results[1] or first))
    end
    return nil, nil
end

local function targetParams(target, modifierName)
    local params = {}
    params[UnitOperationTypes.PARAM_X] = target.x
    params[UnitOperationTypes.PARAM_Y] = target.y
    if modifierName and UnitOperationTypes.PARAM_MODIFIERS and UnitOperationMoveModifiers then
        params[UnitOperationTypes.PARAM_MODIFIERS] = UnitOperationMoveModifiers[modifierName]
    end
    return params
end

-- Whether the player sees something of another player's on the plot: a unit
-- the game says this player sees (its tile and the unit itself, so not a
-- stealth unit), or a city on a tile seen now. An occupant the player cannot
-- see is no target: offering an attack on it would give it away. When the
-- game cannot be asked about visibility, nothing is seen.
local function hasForeignOccupant(playerId, plot, x, y)
    local visibility = nil
    pcall(function() visibility = PlayersVisibility and PlayersVisibility[playerId] end)
    local plotSeen = false
    pcall(function() plotSeen = visibility:IsVisible(x, y) == true end)
    if not plotSeen then return false end

    local function seen(unit)
        local ok, visible = pcall(function() return visibility:IsUnitVisible(unit) end)
        return ok and visible == true
    end

    local found = false
    pcall(function()
        local units = Units and Units.GetUnitsInPlot and Units.GetUnitsInPlot(plot)
        if type(units) ~= "table" then return end
        if units.Members then
            for _, unit in units:Members() do
                if unit:GetOwner() ~= playerId and seen(unit) then found = true end
            end
        else
            for _, unit in ipairs(units) do
                if unit:GetOwner() ~= playerId and seen(unit) then found = true end
            end
        end
    end)
    if found then return true end
    pcall(function()
        local city = Cities and Cities.GetCityInPlot and Cities.GetCityInPlot(x, y)
        if city and city:GetOwner() ~= playerId then found = true end
    end)
    return found
end

-- Move a unit. The request carries no attack modifier, so a move onto another
-- player's unit is refused by the game's check and can never become an attack.
local function executeUnitMove(playerId, command)
    local unitId = command.unit_id
    local target = command.target

    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end
    if not target or not target.x or not target.y then
        return false, ERROR_CODES.INVALID_TARGET, "Missing or invalid target"
    end

    if Log then Log.Info("Executing: UNIT_MOVE unit=%d to (%d,%d)", unitId, target.x, target.y) end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    local movesRemaining = unit:GetMovementMovesRemaining() or 0
    if movesRemaining <= 0 then
        return false, ERROR_CODES.NOT_ENOUGH_MOVEMENT, "Unit has no movement remaining"
    end

    local plot = Map and Map.GetPlot(target.x, target.y)
    if not plot then
        return false, ERROR_CODES.INVALID_TARGET, "Invalid target position"
    end
    if plot:IsImpassable() then
        return false, ERROR_CODES.INVALID_TARGET, "Target position is impassable"
    end
    if not UnitOperationTypes then
        return false, ERROR_CODES.EXECUTION_FAILED, "Move unavailable: no UnitOperationTypes in this Lua context"
    end

    -- As the game's own move routine does, ask about war only for a plot the
    -- player can see: a move into the fog is a move toward it, not onto it.
    -- A refusal must not describe something the player cannot see: when
    -- nothing of anybody else's is in sight on the plot, the answer says only
    -- that the move was refused, not that the tile is held or by whom.
    local unseen = not hasForeignOccupant(playerId, plot, target.x, target.y)
    local function refusedBlind()
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("Move to (%d,%d) refused", target.x, target.y)
    end
    if plotIsVisible(playerId, target.x, target.y) then
        local warCode, warMessage = warRefusal(unit, target.x, target.y)
        if warCode then
            if unseen and warCode == ERROR_CODES.WOULD_START_WAR then return refusedBlind() end
            return false, warCode, warMessage
        end
    end

    local ok, code, message = requestUnitAction(
        unit, "UNITOPERATION_MOVE_TO", targetParams(target, "NONE"), "Move")
    if not ok then
        if unseen and code == ERROR_CODES.OPERATION_REFUSED then return refusedBlind() end
        return false, code, message
    end
    return true, nil, string.format("Move of unit %d to (%d,%d) requested", unitId, target.x, target.y)
end

-- Shared by the melee and the ranged attack.
local function executeAttack(playerId, command, typeName, modifierName, label)
    local unitId = command.unit_id
    local target = command.target

    if not unitId or not target or not target.x or not target.y then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id or target"
    end

    if Log then
        Log.Info("Executing: %s unit=%d at (%d,%d)", label, unitId, target.x, target.y)
    end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    if (unit:GetAttacksRemaining() or 0) <= 0 then
        return false, ERROR_CODES.NOT_ENOUGH_MOVEMENT, "Unit has no attacks remaining"
    end

    local plot = Map and Map.GetPlot(target.x, target.y)
    if not plot then
        return false, ERROR_CODES.INVALID_TARGET, "Invalid target position"
    end
    if not hasForeignOccupant(playerId, plot, target.x, target.y) then
        return false, ERROR_CODES.INVALID_TARGET,
            string.format("Nothing to attack at (%d,%d)", target.x, target.y)
    end
    if not UnitOperationTypes then
        return false, ERROR_CODES.EXECUTION_FAILED,
            label .. " unavailable: no UnitOperationTypes in this Lua context"
    end

    local warCode, warMessage = warRefusal(unit, target.x, target.y)
    if warCode then
        return false, warCode, warMessage
    end

    local ok, code, message = requestUnitAction(unit, typeName, targetParams(target, modifierName), label)
    if not ok then
        return false, code, message
    end
    return true, nil, string.format("%s at (%d,%d) requested", label, target.x, target.y)
end

-- Melee attack: the move operation with the attack modifier, which is how the
-- game's own move routine engages a unit on the destination tile.
local function executeUnitAttack(playerId, command)
    return executeAttack(playerId, command, "UNITOPERATION_MOVE_TO", "ATTACK", "Attack")
end

-- Ranged attack: its own operation (archers, catapults, bombards).
local function executeUnitRangedAttack(playerId, command)
    return executeAttack(playerId, command, "UNITOPERATION_RANGE_ATTACK", nil, "Ranged attack")
end

-- A unit action without a target.
local function executeSimpleUnitAction(playerId, command, typeName, label)
    local unitId = command.unit_id
    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end

    if Log then Log.Info("Executing: %s unit=%d", label, unitId) end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    local ok, code, message = requestUnitAction(unit, typeName, nil, label)
    if not ok then
        return false, code, message
    end
    return true, nil, string.format("%s of unit %d requested", label, unitId)
end

local function executeUnitFortify(playerId, command)
    return executeSimpleUnitAction(playerId, command, "UNITOPERATION_FORTIFY", "Fortify")
end

local function executeUnitSkip(playerId, command)
    return executeSimpleUnitAction(playerId, command, "UNITOPERATION_SKIP_TURN", "Skip")
end

local function executeUnitSleep(playerId, command)
    return executeSimpleUnitAction(playerId, command, "UNITOPERATION_SLEEP", "Sleep")
end

local function executeFoundCity(playerId, command)
    return executeSimpleUnitAction(playerId, command, "UNITOPERATION_FOUND_CITY", "Found city")
end

-- Build improvement
local function executeUnitBuild(playerId, command)
    local unitId = command.unit_id
    local itemId = command.item_id

    if not unitId or not itemId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id or item_id"
    end

    if Log then Log.Info("Executing: UNIT_BUILD unit=%d item=%s", unitId, tostring(itemId)) end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    if UnitManager and UnitManager.RequestOperation and UnitOperationTypes then
        local params = {}
        params[UnitOperationTypes.PARAM_IMPROVEMENT_TYPE] = itemId
        UnitManager.RequestOperation(unit, UnitOperationTypes.BUILD_IMPROVEMENT, params)
        return true, nil, "Build started"
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Build execution failed"
end

-- Production. The game's own production panel sends the row's Hash under the
-- parameter of its kind, with the exclusive insert mode (replace, do not
-- queue). Districts and wonders also need a plot, which this command does not
-- carry, so they are refused.
local PRODUCTION_KINDS = {
    unit = { table = "Units", column = "UnitType", param = "PARAM_UNIT_TYPE" },
    building = { table = "Buildings", column = "BuildingType", param = "PARAM_BUILDING_TYPE" },
    district = { table = "Districts", column = "DistrictType", param = "PARAM_DISTRICT_TYPE", placed = true },
}
local PRODUCTION_ORDER = { "unit", "building", "district" }

local function findProductionRow(kind, matches)
    local gameTable = GameInfo and GameInfo[kind.table]
    if not gameTable then return nil end
    local found = nil
    pcall(function()
        for row in gameTable() do
            if matches(row) then
                found = row
                break
            end
        end
    end)
    return found
end

-- The row and kind for an item given by type name (looked up in every kind)
-- or by row index (in the kind `itemType` names; units when it names none).
-- Found rows are remembered (see findActionRow): without this, listing what
-- a city can build walked the unit table once per candidate item.
local productionRows = {}
local function resolveProductionItem(item, itemType)
    if type(item) == "string" then
        local known = productionRows[item]
        if known then return known.row, known.kind end
        for _, name in ipairs(PRODUCTION_ORDER) do
            local kind = PRODUCTION_KINDS[name]
            local row = findProductionRow(kind, function(r) return r[kind.column] == item end)
            if row then
                productionRows[item] = { row = row, kind = kind }
                return row, kind
            end
        end
        return nil, nil
    end
    local kind = PRODUCTION_KINDS[itemType or "unit"]
    if not kind then return nil, nil end
    local row = findProductionRow(kind, function(r) return r.Index == item end)
    return row, kind
end

local function executeCityProduce(playerId, command)
    local cityId = command.city_id
    local itemId = command.item_id
    local itemType = command.item_type

    if not cityId or itemId == nil then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing city_id or item_id"
    end

    if Log then
        Log.Info("Executing: CITY_PRODUCE city=%d item=%s type=%s", cityId, tostring(itemId), tostring(itemType))
    end

    local city, errCode, errMsg = safeGetCity(playerId, cityId)
    if not city then
        return false, errCode, errMsg
    end

    if itemType == "wonder" then
        return false, ERROR_CODES.NOT_IMPLEMENTED,
            "A wonder needs a plot; placement is not supported: " .. tostring(itemId)
    end

    local row, kind = resolveProductionItem(itemId, itemType)
    if not row then
        return false, ERROR_CODES.INVALID_COMMAND, "Unknown production item: " .. tostring(itemId)
    end
    local typeName = tostring(row[kind.column])
    -- A district always needs a plot; a building does when its row says so
    -- (wonders), as the game's production panel checks before it builds one.
    if kind.placed or row.RequiresPlacement == true then
        return false, ERROR_CODES.NOT_IMPLEMENTED,
            typeName .. " needs a plot; placement is not supported"
    end
    -- The game's production panel never offers to build these.
    if row.MustPurchase == true then
        return false, ERROR_CODES.OPERATION_REFUSED,
            typeName .. " can only be purchased, not built"
    end
    if row.Hash == nil then
        return false, ERROR_CODES.EXECUTION_FAILED, typeName .. " has no hash in this Lua context"
    end

    if not CityManager or not CityManager.RequestOperation or not CityOperationTypes then
        return false, ERROR_CODES.EXECUTION_FAILED,
            "Production unavailable: no CityManager in this Lua context"
    end

    -- Ask both CanProduce(hash, true) (unlock/prereq eligibility) and
    -- CanProduce(hash, false, true) (exclusion/start-now check), matching Civ6's
    -- ProductionPanel.lua. Checking only (hash, false, true) returns true for
    -- unresearched buildings (e.g. BUILDING_SEWER) and clears the city's queue.
    local eligible = nil
    local allowed = nil
    pcall(function()
        local buildQueue = city:GetBuildQueue()
        if buildQueue and buildQueue.CanProduce then
            eligible = buildQueue:CanProduce(row.Hash, true)
            if eligible == true then
                allowed = buildQueue:CanProduce(row.Hash, false, true)
            end
        end
    end)
    -- Anything but a clear yes from both checks is a refusal, a game that
    -- cannot be asked included: a request without it would be acknowledged as
    -- a success, and these checks also decide what the exporter offers.
    local couldNotAsk = (eligible == nil) or (eligible == true and allowed == nil)
    if eligible ~= true or allowed ~= true then
        if Log and Log.Debug then
            Log.Debug(string.format(
                "City %d CanProduce(%s): eligible=%s allowed=%s",
                cityId, tostring(typeName), tostring(eligible), tostring(allowed)
            ))
        end
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("City %d cannot start %s now%s", cityId, typeName,
                couldNotAsk and " (the game could not be asked)" or "")
    end

    local params = {}
    params[CityOperationTypes[kind.param]] = row.Hash
    params[CityOperationTypes.PARAM_INSERT_MODE] = CityOperationTypes.VALUE_EXCLUSIVE
    if not dryRun then
        local requested, err = pcall(CityManager.RequestOperation, city, CityOperationTypes.BUILD, params)
        if not requested then
            return false, ERROR_CODES.EXECUTION_FAILED, "Production request raised: " .. tostring(err)
        end
    end
    return true, nil, string.format("City %d: production of %s requested", cityId, typeName)
end

-- ----------------------------------------------------------------------------
-- Research (technologies and civics)
-- ----------------------------------------------------------------------------

-- Resolve a research item to its GameInfo row. The item is a row index (what
-- the Python action space sends) or a type name such as "TECH_MINING".
-- Returns index, type name and hash; all nil when the GameInfo table is there
-- and has no such row. Without the table an index is passed through unchecked.
local function resolveResearchItem(tableName, typeField, item)
    local gameTable = GameInfo and GameInfo[tableName]
    if not gameTable then
        if type(item) == "number" then
            return item, tostring(item), nil
        end
        return nil, nil, nil
    end

    local index, typeName, hash = nil, nil, nil
    pcall(function()
        for row in gameTable() do
            if row.Index == item or row[typeField] == item then
                index, typeName, hash = row.Index, row[typeField], row.Hash
                break
            end
        end
    end)
    return index, typeName, hash
end

-- Call a method on a game object; nil when it is missing or raises.
local function callIfPresent(object, methodName, ...)
    if object == nil then return nil end
    local okMethod, method = pcall(function() return object[methodName] end)
    if not okMethod or not method then return nil end
    local okCall, result = pcall(method, object, ...)
    if okCall then return result end
    return nil
end

-- Shared by technologies and civics. `kind` names the differences:
--   table/typeField   GameInfo table and its type-name column
--   getter            player method returning the progress object
--   has/can           its "already has" and "can choose now" methods (by index)
--   operation/param   PlayerOperations keys for the UI request
--   setter            GameCore method on the progress object (fallback)
local function executeResearch(playerId, command, kind)
    local item = command.item_id
    if item == nil then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing item_id (" .. kind.label .. ")"
    end

    if Log then Log.Info("Executing: %s item=%s", kind.logName, tostring(item)) end

    if not Players then
        return false, ERROR_CODES.EXECUTION_FAILED, "Players table not available"
    end
    local player = Players[playerId]
    if not player then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player not found"
    end

    local index, typeName, hash = resolveResearchItem(kind.table, kind.typeField, item)
    if index == nil then
        return false, ERROR_CODES.INVALID_TARGET,
            "Unknown " .. kind.label .. ": " .. tostring(item)
    end

    -- Refuse what the game would drop without a word. Each check is skipped
    -- when its method is missing in this Lua context.
    local progressObject = callIfPresent(player, kind.getter)
    if callIfPresent(progressObject, kind.has, index) == true then
        return false, ERROR_CODES.INVALID_TARGET,
            tostring(typeName) .. " is already researched"
    end
    if callIfPresent(progressObject, kind.can, index) == false then
        return false, ERROR_CODES.INVALID_TARGET,
            tostring(typeName) .. " cannot be chosen now"
    end

    -- Primary path: ask the game as its own research and civics choosers do:
    -- the type parameter is the row's Hash (not its Index), with an exclusive
    -- insert mode so the choice replaces the queue. The request is
    -- fire-and-forget: the game applies it after this call returns and drops
    -- it without an error if it does not like it, so success here means
    -- "requested", not "done". The next exported state shows what the game
    -- did. No notification is touched: the game clears its own "choose
    -- research" blocker when the choice takes effect.
    local requested = false
    if hash ~= nil and UI and UI.RequestPlayerOperation and PlayerOperations
       and PlayerOperations[kind.operation] and PlayerOperations[kind.param] then
        local params = {}
        params[PlayerOperations[kind.param]] = hash
        if PlayerOperations.PARAM_INSERT_MODE and PlayerOperations.VALUE_EXCLUSIVE then
            params[PlayerOperations.PARAM_INSERT_MODE] = PlayerOperations.VALUE_EXCLUSIVE
        end
        requested = pcall(function()
            UI.RequestPlayerOperation(playerId, PlayerOperations[kind.operation], params)
        end)
    end
    if requested then
        return true, nil, kind.requestedMessage .. tostring(typeName)
    end

    -- Fallback: the GameCore setter, which takes the row index and exists
    -- only in gameplay contexts.
    if not progressObject then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player " .. kind.label .. " data not available"
    end
    local okSetter, setter = pcall(function() return progressObject[kind.setter] end)
    if not okSetter or not setter then
        return false, ERROR_CODES.EXECUTION_FAILED,
            "No way to set the " .. kind.label .. " in this Lua context"
    end
    local okSet, setErr = pcall(setter, progressObject, index)
    if not okSet then
        return false, ERROR_CODES.EXECUTION_FAILED,
            kind.setter .. " failed: " .. tostring(setErr)
    end
    return true, nil, kind.setMessage .. tostring(typeName)
end

local RESEARCH_TECH_KIND = {
    label = "tech",
    logName = "RESEARCH_TECH",
    table = "Technologies",
    typeField = "TechnologyType",
    getter = "GetTechs",
    has = "HasTech",
    can = "CanResearch",
    operation = "RESEARCH",
    param = "PARAM_TECH_TYPE",
    setter = "SetResearchingTech",
    requestedMessage = "Research requested: ",
    setMessage = "Researching tech: ",
}

local RESEARCH_CIVIC_KIND = {
    label = "civic",
    logName = "RESEARCH_CIVIC",
    table = "Civics",
    typeField = "CivicType",
    getter = "GetCulture",
    has = "HasCivic",
    can = "CanProgress",
    operation = "PROGRESS_CIVIC",
    param = "PARAM_CIVIC_TYPE",
    setter = "SetProgressingCivic",
    requestedMessage = "Civic requested: ",
    setMessage = "Progressing civic: ",
}

-- Research a technology
local function executeResearchTech(playerId, command)
    return executeResearch(playerId, command, RESEARCH_TECH_KIND)
end

-- Research a civic
local function executeResearchCivic(playerId, command)
    return executeResearch(playerId, command, RESEARCH_CIVIC_KIND)
end

-- ----------------------------------------------------------------------------
-- v1.8.0 W2.1 step 4 — newly-implemented handlers
-- ----------------------------------------------------------------------------

-- Buy production (rush-buy a unit or building with gold).
-- Mirrors executeCityProduce's API + dispatch tables, but calls PURCHASE
-- instead of BUILD.
local PURCHASE_PARAM_KEYS = {
    unit = "PARAM_UNIT_TYPE",
    building = "PARAM_BUILDING_TYPE",
    wonder = "PARAM_BUILDING_TYPE",
    district = "PARAM_DISTRICT_TYPE",
}

local function executeCityBuy(playerId, command)
    local cityId = command.city_id
    local itemId = command.item_id
    local itemType = command.item_type or "unit"

    if not cityId or not itemId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing city_id or item_id"
    end

    if Log then Log.Info("Executing: CITY_BUY city=%d item=%s type=%s",
        cityId, tostring(itemId), tostring(itemType)) end

    local city, errCode, errMsg = safeGetCity(playerId, cityId)
    if not city then
        return false, errCode, errMsg
    end

    -- Primary method: CityManager.RequestOperation(city, CityOperationTypes.PURCHASE, params).
    if CityManager and CityManager.RequestOperation and CityOperationTypes then
        local paramKey = PURCHASE_PARAM_KEYS[itemType] or PURCHASE_PARAM_KEYS["unit"]
        if CityOperationTypes[paramKey] and CityOperationTypes.PURCHASE then
            local params = {}
            params[CityOperationTypes[paramKey]] = itemId
            CityManager.RequestOperation(city, CityOperationTypes.PURCHASE, params)
            return true, nil, string.format("City %d purchasing %s (%s)",
                cityId, tostring(itemId), tostring(itemType))
        end
    end

    -- Fallback: buildQueue:PurchaseProduction(type, item) on older Civ6 mod contexts.
    local buildQueue = city:GetBuildQueue()
    if buildQueue and buildQueue.PurchaseProduction then
        local ok, result = pcall(function()
            return buildQueue:PurchaseProduction(itemType, itemId)
        end)
        if ok and result then
            return true, nil, string.format("City %d purchased %s (fallback)",
                cityId, tostring(itemId))
        end
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Purchase failed"
end

-- Lock or unlock a citizen on a specific plot.
-- command.item_id = plot index; command.metadata.lock = boolean.
--
-- CAVEAT (gap-analysis C2 on commit 3572832):
-- `SetCitizenLocked` is a void Civ6 API. The `ok` returned by `pcall`
-- only indicates that no Lua exception was thrown — it does NOT confirm
-- the lock state was actually applied (e.g. when the plot is outside the
-- city's working radius, or the citizen is already in the requested
-- state). The handler returns true so the agent loop sees `ack.success=
-- true` even if the in-game change was silently rejected. Future hardening
-- could query `citizens:GetCitizenIsWorking(plotIndex)` after the call to
-- compare against `lock`, but that API surface varies across Civ6 patches.
-- For v1.8.0 we accept the void-call contract and rely on action-mask
-- legality filtering to keep the agent from issuing nonsense plots.
local function executeCityCitizen(playerId, command)
    local cityId = command.city_id
    local plotIndex = command.item_id
    local lock = false
    if command.metadata and command.metadata.lock ~= nil then
        lock = command.metadata.lock
    end

    if not cityId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing city_id"
    end
    if plotIndex == nil then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing item_id (plot index)"
    end

    if Log then Log.Info("Executing: CITY_CITIZEN city=%d plot=%d lock=%s",
        cityId, plotIndex, tostring(lock)) end

    local city, errCode, errMsg = safeGetCity(playerId, cityId)
    if not city then
        return false, errCode, errMsg
    end

    local citizens = city.GetCitizens and city:GetCitizens()
    if not citizens then
        return false, ERROR_CODES.EXECUTION_FAILED, "Citizens not available"
    end

    if citizens.SetCitizenLocked then
        -- `ok` = pcall status only (see caveat above). Returning true here
        -- means "no exception", not "lock applied".
        local ok = pcall(function() citizens:SetCitizenLocked(plotIndex, lock) end)
        if ok then
            return true, nil, string.format("Plot %d lock=%s on city %d (void API, see caveat)",
                plotIndex, tostring(lock), cityId)
        end
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Citizen lock failed"
end

-- Promote a unit. command.item_id carries the promotion id (0 = first available).
local function executeUnitPromote(playerId, command)
    local unitId = command.unit_id
    local promotionId = command.item_id  -- may be 0 for first-available
    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end

    if Log then Log.Info("Executing: UNIT_PROMOTE unit=%d promotion=%s",
        unitId, tostring(promotionId)) end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    if unit.CanPromote and not unit:CanPromote() then
        return false, ERROR_CODES.EXECUTION_FAILED, "Unit cannot promote right now"
    end

    if UnitManager and UnitManager.RequestOperation and UnitOperationTypes
        and UnitOperationTypes.PROMOTE then
        local params = nil
        if promotionId and promotionId > 0
            and UnitOperationTypes.PARAM_PROMOTION_TYPE then
            params = {}
            params[UnitOperationTypes.PARAM_PROMOTION_TYPE] = promotionId
        end
        UnitManager.RequestOperation(unit, UnitOperationTypes.PROMOTE, params)
        return true, nil, "Promotion requested"
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Promote operation unavailable"
end

-- True only when the setting is the boolean true: a string "true", a 1 or an
-- unset setting leaves the command turned off.
local function _configFlagEnabled(name)
    return AlphaZeroConfig ~= nil and AlphaZeroConfig.Get(name) == true
end

-- Delete (disband) a unit.
local function executeUnitDelete(playerId, command)
    if not _configFlagEnabled("ALLOW_UNIT_DELETE") then
        if Log then Log.Warn("Refused UNIT_DELETE: ALLOW_UNIT_DELETE is turned off") end
        return false, ERROR_CODES.OPERATION_REFUSED,
            "Refused: deleting units is turned off (ALLOW_UNIT_DELETE)"
    end

    local unitId = command.unit_id
    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end

    if Log then Log.Info("Executing: UNIT_DELETE unit=%d", unitId) end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    if UnitManager and UnitManager.RequestCommand and UnitCommandTypes
        and UnitCommandTypes.DELETE then
        UnitManager.RequestCommand(unit, UnitCommandTypes.DELETE)
        return true, nil, "Unit deleted"
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Delete command unavailable"
end

-- ----------------------------------------------------------------------------
-- v1.8.0 W2.1 step 5 — diplomacy dispatcher
-- ----------------------------------------------------------------------------

-- Sub-action ID -> name mapping. Mirrors DiplomacyActionType in
-- alpha_zero/core/civ6_actions.py:79-90 (IntEnum values are part of the
-- wire contract; do NOT reorder). 9 total sub-actions; 3 implemented this
-- sprint per BRIDGE_IMPLEMENTED_DIPLOMACY, 6 round-trip to NOT_IMPLEMENTED.
local DIPLOMACY_SUB_ACTION_NAMES = {
    [0] = "DECLARE_WAR",
    [1] = "MAKE_PEACE",
    [2] = "OPEN_BORDERS",
    [3] = "ALLIANCE",
    [4] = "TRADE_DEAL",
    [5] = "DENOUNCE",
    [6] = "DELEGATE",
    [7] = "EMBASSY",
    [8] = "JOINT_WAR",
}

-- Per-sub-action dispatch table. Implemented = function, Deferred = false.
local DIPLOMACY_SUB_ACTION_HANDLERS

local function _diplomacyRequest(playerId, targetPlayerId, civ6ActionName, subActionName)
    -- Resolve the Civ6 Lua diplomacy API and request the action against the
    -- target player. The Civ6 API surface varies slightly across game patches;
    -- this code uses the most-stable path: Players[pid]:GetDiplomacy():DoAction
    -- when available, else falls back to UI.RequestDiplomacyMove. Both are
    -- documented entry points.
    if not Players or not Players[playerId] then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player not found: " .. tostring(playerId)
    end

    local player = Players[playerId]
    local diplomacy = player.GetDiplomacy and player:GetDiplomacy() or nil
    if not diplomacy then
        return false, ERROR_CODES.EXECUTION_FAILED, "Diplomacy table unavailable"
    end

    -- The error of the last path that was tried and raised, so a failure says
    -- why instead of claiming there was nothing to call.
    local lastError = nil

    -- Primary path: GetDiplomacy():DoAction(...). Civ6 expects a string or
    -- int action constant from DiplomaticActions table; pass civ6ActionName.
    if diplomacy.DoAction then
        local ok, err = pcall(function() diplomacy:DoAction(civ6ActionName, targetPlayerId) end)
        if ok then
            return true, nil, string.format("%s against player %d", subActionName, targetPlayerId)
        end
        lastError = err
        if Log then Log.Warn("DoAction failed for %s: %s", subActionName, tostring(err)) end
    end

    -- Fallback: UI.RequestDiplomacyMove for older mod contexts.
    if UI and UI.RequestDiplomacyMove then
        local ok, err = pcall(function() UI.RequestDiplomacyMove(civ6ActionName, targetPlayerId) end)
        if ok then
            return true, nil, string.format("%s requested via UI (player %d)",
                subActionName, targetPlayerId)
        end
        lastError = err
        if Log then Log.Warn("RequestDiplomacyMove failed for %s: %s", subActionName, tostring(err)) end
    end

    if lastError then
        return false, ERROR_CODES.EXECUTION_FAILED,
            string.format("Diplomacy action %s failed: %s", subActionName, tostring(lastError))
    end

    return false, ERROR_CODES.EXECUTION_FAILED,
        string.format("No diplomacy API available for %s", subActionName)
end

-- True only when the setting is the boolean true: a string "true", a 1 or an
-- unset setting leaves the command turned off.
local function _diplomacySettingOn(name)
    return _configFlagEnabled(name)
end

-- Ask the acting player's diplomacy object one yes/no question about the
-- target. Fails closed: where the game cannot be asked (no diplomacy object,
-- no such method, the call raises, the answer is not a boolean) the answer is
-- (nil, message) and the caller refuses the command.
local function _askDiplomacy(playerId, method, targetPlayerId)
    local function cannot(why)
        return nil, string.format("Cannot ask the game %s(%s): %s",
            method, tostring(targetPlayerId), why)
    end
    local player = Players and Players[playerId]
    if not player or type(player.GetDiplomacy) ~= "function" then
        return cannot("no diplomacy for player " .. tostring(playerId))
    end
    local ok, diplomacy = pcall(player.GetDiplomacy, player)
    if not ok then return cannot("GetDiplomacy raised: " .. tostring(diplomacy)) end
    if type(diplomacy) ~= "table" and type(diplomacy) ~= "userdata" then
        return cannot("no diplomacy object")
    end
    local okMethod, fn = pcall(function() return diplomacy[method] end)
    if not okMethod or type(fn) ~= "function" then
        return cannot("the method is not available")
    end
    local okCall, answer = pcall(fn, diplomacy, targetPlayerId)
    if not okCall then return cannot("the call raised: " .. tostring(answer)) end
    if type(answer) ~= "boolean" then
        return cannot("the answer was not a boolean (" .. tostring(answer) .. ")")
    end
    return answer, nil
end

-- Peace and denouncing are for major civilizations only (not barbarians,
-- city-states or free cities). Asks Players[target]:IsMajor() fail-closed;
-- returns nil when the target may be acted on, else (code, message).
local function _refuseUnlessMajor(targetPlayerId)
    local function cannot(why)
        return ERROR_CODES.EXECUTION_FAILED, string.format(
            "Cannot ask the game IsMajor(%s): %s", tostring(targetPlayerId), why)
    end
    local target = Players and Players[targetPlayerId]
    if target == nil then return cannot("no such player") end
    local okMethod, fn = pcall(function() return target.IsMajor end)
    if not okMethod or type(fn) ~= "function" then
        return cannot("the method is not available")
    end
    local okCall, answer = pcall(fn, target)
    if not okCall then return cannot("the call raised: " .. tostring(answer)) end
    if type(answer) ~= "boolean" then
        return cannot("the answer was not a boolean (" .. tostring(answer) .. ")")
    end
    if not answer then
        return ERROR_CODES.OPERATION_REFUSED, string.format(
            "Refused: player %s is not a major civilization", tostring(targetPlayerId))
    end
    return nil
end

local function diplomacyDeclareWar(playerId, targetPlayerId)
    -- Refused unless the setting allows it: nothing in this mod may start a
    -- war by default (the same answer a move or an attack that would gets).
    if not _diplomacySettingOn("ALLOW_DECLARE_WAR") then
        return false, ERROR_CODES.WOULD_START_WAR,
            string.format("Refused: declaring war on player %s is turned off (ALLOW_DECLARE_WAR)",
                tostring(targetPlayerId))
    end
    return _diplomacyRequest(playerId, targetPlayerId, "DIPLOACTION_DECLARE_WAR", "DECLARE_WAR")
end

local function diplomacyMakePeace(playerId, targetPlayerId)
    -- Refused unless the setting allows it, and then only with a major
    -- civilization the game itself reports the pair at war with.
    if not _diplomacySettingOn("ALLOW_MAKE_PEACE") then
        return false, ERROR_CODES.DIPLOMACY_TURNED_OFF,
            string.format("Refused: making peace with player %s is turned off (ALLOW_MAKE_PEACE)",
                tostring(targetPlayerId))
    end
    local majorCode, majorWhy = _refuseUnlessMajor(targetPlayerId)
    if majorCode then return false, majorCode, majorWhy end
    local atWar, why = _askDiplomacy(playerId, "IsAtWarWith", targetPlayerId)
    if atWar == nil then return false, ERROR_CODES.EXECUTION_FAILED, why end
    if not atWar then
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("Refused: not at war with player %s, so there is no peace to make",
                tostring(targetPlayerId))
    end
    return _diplomacyRequest(playerId, targetPlayerId, "DIPLOACTION_MAKE_PEACE", "MAKE_PEACE")
end

local function diplomacyDenounce(playerId, targetPlayerId)
    -- Refused unless the setting allows it, and then only a major civilization
    -- the game reports as met and not at war.
    if not _diplomacySettingOn("ALLOW_DENOUNCE") then
        return false, ERROR_CODES.DIPLOMACY_TURNED_OFF,
            string.format("Refused: denouncing player %s is turned off (ALLOW_DENOUNCE)",
                tostring(targetPlayerId))
    end
    local majorCode, majorWhy = _refuseUnlessMajor(targetPlayerId)
    if majorCode then return false, majorCode, majorWhy end
    local met, whyMet = _askDiplomacy(playerId, "HasMet", targetPlayerId)
    if met == nil then return false, ERROR_CODES.EXECUTION_FAILED, whyMet end
    if not met then
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("Refused: player %s has not been met", tostring(targetPlayerId))
    end
    local atWar, whyWar = _askDiplomacy(playerId, "IsAtWarWith", targetPlayerId)
    if atWar == nil then return false, ERROR_CODES.EXECUTION_FAILED, whyWar end
    if atWar then
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("Refused: at war with player %s; denouncing is for a player at peace",
                tostring(targetPlayerId))
    end
    return _diplomacyRequest(playerId, targetPlayerId, "DIPLOACTION_DENOUNCE", "DENOUNCE")
end

DIPLOMACY_SUB_ACTION_HANDLERS = {
    [0] = diplomacyDeclareWar,    -- DECLARE_WAR
    [1] = diplomacyMakePeace,     -- MAKE_PEACE
    [2] = false,                  -- OPEN_BORDERS  (deferred)
    [3] = false,                  -- ALLIANCE      (deferred)
    [4] = false,                  -- TRADE_DEAL    (deferred)
    [5] = diplomacyDenounce,      -- DENOUNCE
    [6] = false,                  -- DELEGATE      (deferred)
    [7] = false,                  -- EMBASSY       (deferred)
    [8] = false,                  -- JOINT_WAR     (deferred)
}

-- Diplomacy dispatcher. Reads command.metadata.{sub_action, target_player_id}.
local function executeDiplomacy(playerId, command)
    local metadata = command.metadata or {}
    local subActionId = metadata.sub_action
    local targetPlayerId = metadata.target_player_id

    if subActionId == nil then
        return false, ERROR_CODES.INVALID_COMMAND, "DIPLOMACY missing metadata.sub_action"
    end
    if targetPlayerId == nil then
        return false, ERROR_CODES.INVALID_COMMAND, "DIPLOMACY missing metadata.target_player_id"
    end

    if targetPlayerId == playerId then
        -- No diplomacy is aimed at oneself, whatever the sub-action (declaring
        -- war and the deferred ones included); checked first, nothing is sent.
        return false, ERROR_CODES.INVALID_TARGET,
            string.format("DIPLOMACY target player %s is the acting player", tostring(targetPlayerId))
    end

    local subActionName = DIPLOMACY_SUB_ACTION_NAMES[subActionId] or "UNKNOWN"
    if Log then
        Log.Info("Executing: DIPLOMACY sub_action=%d (%s) target_player=%d",
            subActionId, subActionName, targetPlayerId)
    end

    local handler = DIPLOMACY_SUB_ACTION_HANDLERS[subActionId]
    if not handler then
        -- Unknown sub-action id (out-of-range or new enum member added to
        -- Python without a matching Lua handler).
        local msg = string.format(
            "Diplomacy sub-action id %d not recognised (Python/Lua enum drift?)",
            subActionId)
        if Log then Log.Warn(msg) end
        return false, ERROR_CODES.NOT_IMPLEMENTED, msg
    end
    if handler == false then
        -- Recognised but deferred this sprint. Per-sub-action log line per
        -- the v1.8.0 plan's "log deferred sub-actions" requirement so future
        -- prioritisation can see which sub-actions the agent actually picks.
        local msg = string.format("Diplomacy sub-action '%s' deferred to v1.9.0",
            subActionName)
        if Log then Log.Info(msg) end
        return false, ERROR_CODES.NOT_IMPLEMENTED, msg
    end

    return handler(playerId, targetPlayerId)
end

-- Stub handler for unimplemented actions
local function executeNotImplemented(playerId, command)
    local actionType = command and command.action_type or "unknown"
    local msg = string.format("Action type '%s' is not yet implemented", tostring(actionType))
    if Log then Log.Warn(msg) end
    return false, ERROR_CODES.NOT_IMPLEMENTED, msg
end

-- Action dispatch table.
-- All 17 ActionType members have real handlers as of v1.8.0 W2.1 steps 4 + 5
-- (CITY_BUY, CITY_CITIZEN, UNIT_PROMOTE, UNIT_DELETE, DIPLOMACY) and the
-- W2.1 hardening pass (UNIT_RANGED_ATTACK = executeUnitRangedAttack, distinct
-- from melee UNIT_ATTACK). The action-mask invariant test enforces zero stubs.
local ACTION_HANDLERS = {
    [ACTION_TYPES.END_TURN] = executeEndTurn,
    [ACTION_TYPES.UNIT_MOVE] = executeUnitMove,
    [ACTION_TYPES.UNIT_ATTACK] = executeUnitAttack,
    [ACTION_TYPES.UNIT_RANGED_ATTACK] = executeUnitRangedAttack,
    [ACTION_TYPES.UNIT_FORTIFY] = executeUnitFortify,
    [ACTION_TYPES.UNIT_SKIP] = executeUnitSkip,
    [ACTION_TYPES.UNIT_SLEEP] = executeUnitSleep,
    [ACTION_TYPES.UNIT_FOUND_CITY] = executeFoundCity,
    [ACTION_TYPES.UNIT_BUILD] = executeUnitBuild,
    [ACTION_TYPES.UNIT_PROMOTE] = executeUnitPromote,
    [ACTION_TYPES.UNIT_DELETE] = executeUnitDelete,
    [ACTION_TYPES.CITY_PRODUCE] = executeCityProduce,
    [ACTION_TYPES.CITY_BUY] = executeCityBuy,
    [ACTION_TYPES.CITY_CITIZEN] = executeCityCitizen,
    [ACTION_TYPES.RESEARCH_TECH] = executeResearchTech,
    [ACTION_TYPES.RESEARCH_CIVIC] = executeResearchCivic,
    [ACTION_TYPES.DIPLOMACY] = executeDiplomacy,  -- W2.1 step 5 (3 sub-actions live)
}

-- ----------------------------------------------------------------------------
-- Public API
-- ----------------------------------------------------------------------------

-- Initialize module
function AlphaZeroActionExecutor.Init()
    if AlphaZeroLogger then
        Log = AlphaZeroLogger.CreateLogger("ActionExecutor")
    end
end

-- Execute a single command
function AlphaZeroActionExecutor.Execute(playerId, command)
    if not command then
        return false, ERROR_CODES.INVALID_COMMAND, "No command provided"
    end

    local actionType = command.action_type
    if not actionType then
        return false, ERROR_CODES.INVALID_COMMAND, "No action_type in command"
    end

    local handler = ACTION_HANDLERS[actionType]
    if not handler then
        if Log then Log.Warn("Unknown action type: %s", tostring(actionType)) end
        return false, ERROR_CODES.INVALID_COMMAND, "Unknown action type: " .. tostring(actionType)
    end

    -- Validate before execution if enabled
    if AlphaZeroConfig and AlphaZeroConfig.Get("VALIDATE_COMMANDS") then
        local valid, validErr = AlphaZeroActionExecutor.Validate(playerId, command)
        if not valid then
            return false, ERROR_CODES.INVALID_COMMAND, validErr
        end
    end

    -- Execute with error protection
    local ok, success, errCode, message = pcall(handler, playerId, command)
    if not ok then
        if Log then Log.Error("Exception in handler: %s", tostring(success)) end
        return false, ERROR_CODES.EXECUTION_FAILED, "Exception: " .. tostring(success)
    end

    return success, errCode, message
end

-- The commands Check can answer for: the ones whose handlers ask the game
-- before they request.
local CHECKABLE = {
    [ACTION_TYPES.UNIT_MOVE] = true, [ACTION_TYPES.UNIT_ATTACK] = true,
    [ACTION_TYPES.UNIT_RANGED_ATTACK] = true, [ACTION_TYPES.UNIT_SKIP] = true,
    [ACTION_TYPES.UNIT_FORTIFY] = true, [ACTION_TYPES.UNIT_SLEEP] = true,
    [ACTION_TYPES.UNIT_FOUND_CITY] = true, [ACTION_TYPES.CITY_PRODUCE] = true,
}

-- What Execute would answer for this command, without requesting anything.
-- The exporter offers an action exactly when this says yes.
function AlphaZeroActionExecutor.Check(playerId, command)
    if not command or not command.action_type then
        return AlphaZeroActionExecutor.Execute(playerId, command)  -- refused there, nothing runs
    end
    if not CHECKABLE[command.action_type] then
        return false, ERROR_CODES.NOT_IMPLEMENTED,
            "No check for action type: " .. tostring(command and command.action_type)
    end
    local savedLog, savedDryRun = Log, dryRun
    dryRun, Log = true, nil  -- a check per candidate would flood the log
    local ok, success, errCode, message = pcall(AlphaZeroActionExecutor.Execute, playerId, command)
    dryRun, Log = savedDryRun, savedLog  -- a check inside a check stays a check
    if not ok then
        return false, ERROR_CODES.EXECUTION_FAILED, "Exception: " .. tostring(success)
    end
    return success, errCode, message
end

-- Execute multiple commands from a CommandMessage
function AlphaZeroActionExecutor.ExecuteCommands(playerId, commandMessage)
    local results = {
        msg_id = commandMessage.msg_id,
        success = true,
        executed = {},
        errors = {},
    }

    local commands = commandMessage.commands or {}
    local maxActions = (AlphaZeroConfig and AlphaZeroConfig.Get("MAX_ACTIONS_PER_TURN")) or 1000

    for i, command in ipairs(commands) do
        if i > maxActions then
            if Log then Log.Warn("Max actions per turn exceeded (%d)", maxActions) end
            results.success = false
            table.insert(results.errors, "Max actions exceeded")
            break
        end

        local success, errCode, message = AlphaZeroActionExecutor.Execute(playerId, command)

        table.insert(results.executed, {
            action_type = command.action_type,
            success = success,
            error_code = errCode,
            message = message,
        })

        if not success then
            results.success = false
            table.insert(results.errors, message)
        end
    end

    return results
end

-- Validate a command without executing (uses safe accessors)
function AlphaZeroActionExecutor.Validate(playerId, command)
    if not command or not command.action_type then
        return false, "Missing command or action_type"
    end

    local actionType = command.action_type

    -- Type-specific validation using safe accessors
    if actionType == ACTION_TYPES.UNIT_MOVE or
       actionType == ACTION_TYPES.UNIT_ATTACK or
       actionType == ACTION_TYPES.UNIT_FORTIFY or
       actionType == ACTION_TYPES.UNIT_SKIP or
       actionType == ACTION_TYPES.UNIT_SLEEP or
       actionType == ACTION_TYPES.UNIT_FOUND_CITY or
       actionType == ACTION_TYPES.UNIT_BUILD then

        if not command.unit_id then
            return false, "Missing unit_id for unit action"
        end

        local unit, errCode, errMsg = safeGetUnit(playerId, command.unit_id)
        if not unit then
            return false, errMsg
        end
    end

    if actionType == ACTION_TYPES.CITY_PRODUCE or
       actionType == ACTION_TYPES.CITY_BUY or
       actionType == ACTION_TYPES.CITY_CITIZEN then
        if not command.city_id then
            return false, "Missing city_id for city action"
        end

        local city, errCode, errMsg = safeGetCity(playerId, command.city_id)
        if not city then
            return false, errMsg
        end
    end

    if actionType == ACTION_TYPES.RESEARCH_TECH or
       actionType == ACTION_TYPES.RESEARCH_CIVIC then
        if not command.item_id then
            return false, "Missing item_id for research action"
        end
    end

    return true, nil
end

-- Get available action types
function AlphaZeroActionExecutor.GetActionTypes()
    return ACTION_TYPES
end

-- Get safe accessor functions (for reuse by other modules)
function AlphaZeroActionExecutor.SafeGetUnit(playerId, unitId)
    return safeGetUnit(playerId, unitId)
end

function AlphaZeroActionExecutor.SafeGetCity(playerId, cityId)
    return safeGetCity(playerId, cityId)
end

-- Initialize on load
AlphaZeroActionExecutor.Init()

print("[AlphaZero] ActionExecutor module loaded")
