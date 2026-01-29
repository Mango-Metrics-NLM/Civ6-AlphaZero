-- ============================================================================
-- ActionExecutor.lua - Command execution for AlphaZero AI Bridge
-- ============================================================================
-- Executes commands received from the Python agent.
-- ============================================================================

AlphaZeroActionExecutor = AlphaZeroActionExecutor or {}

local Log = nil  -- Will be set after Logger loads

-- Action type constants (must match Python ActionType enum)
local ACTION_TYPES = {
    END_TURN = "end_turn",
    UNIT_MOVE = "unit_move",
    UNIT_ATTACK = "unit_attack",
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
}

-- ----------------------------------------------------------------------------
-- Action Execution Functions
-- ----------------------------------------------------------------------------

-- End the current player's turn
local function executeEndTurn(playerId, command)
    if Log then Log.Info("Executing: END_TURN") end

    -- In Civ6, ending turn is handled through the UI
    -- We need to use the proper API
    if UI and UI.RequestAction then
        UI.RequestAction(ActionTypes.ACTION_ENDTURN)
        return true, nil, "Turn ended"
    end

    return true, nil, "End turn requested"
end

-- Move a unit to target position
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

    local player = Players[playerId]
    if not player then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player not found"
    end

    local unit = player:GetUnits():FindID(unitId)
    if not unit then
        return false, ERROR_CODES.UNIT_NOT_FOUND, "Unit not found: " .. tostring(unitId)
    end

    -- Check movement remaining
    local movesRemaining = unit:GetMovementMovesRemaining() or 0
    if movesRemaining <= 0 then
        return false, ERROR_CODES.NOT_ENOUGH_MOVEMENT, "Unit has no movement remaining"
    end

    -- Get target plot
    local plot = Map and Map.GetPlot(target.x, target.y)
    if not plot then
        return false, ERROR_CODES.INVALID_TARGET, "Invalid target position"
    end

    -- Check if target is passable
    if plot:IsImpassable() then
        return false, ERROR_CODES.INVALID_TARGET, "Target position is impassable"
    end

    -- Execute move using UnitManager
    if UnitManager then
        local result = UnitManager.MoveUnit(unit, target.x, target.y)
        if result then
            return true, nil, string.format("Unit %d moved to (%d,%d)", unitId, target.x, target.y)
        end
    end

    -- Fallback: try direct coordinate set (less reliable)
    -- This might not work in all cases but provides a backup
    if unit.SetPosition then
        unit:SetPosition(target.x, target.y)
        return true, nil, string.format("Unit %d position set to (%d,%d)", unitId, target.x, target.y)
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Move execution failed"
end

-- Attack a target position with a unit
local function executeUnitAttack(playerId, command)
    local unitId = command.unit_id
    local target = command.target

    if not unitId or not target then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id or target"
    end

    if Log then Log.Info("Executing: UNIT_ATTACK unit=%d at (%d,%d)", unitId, target.x, target.y) end

    local player = Players[playerId]
    if not player then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player not found"
    end

    local unit = player:GetUnits():FindID(unitId)
    if not unit then
        return false, ERROR_CODES.UNIT_NOT_FOUND, "Unit not found"
    end

    -- Check attacks remaining
    if (unit:GetAttacksRemaining() or 0) <= 0 then
        return false, ERROR_CODES.NOT_ENOUGH_MOVEMENT, "Unit has no attacks remaining"
    end

    -- Execute attack
    if UnitManager and UnitManager.RequestOperation then
        local params = {}
        params[UnitOperationTypes.PARAM_X] = target.x
        params[UnitOperationTypes.PARAM_Y] = target.y
        UnitManager.RequestOperation(unit, UnitOperationTypes.RANGE_ATTACK, params)
        return true, nil, string.format("Attack executed at (%d,%d)", target.x, target.y)
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Attack execution failed"
end

-- Fortify a unit
local function executeUnitFortify(playerId, command)
    local unitId = command.unit_id

    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end

    if Log then Log.Info("Executing: UNIT_FORTIFY unit=%d", unitId) end

    local player = Players[playerId]
    if not player then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player not found"
    end

    local unit = player:GetUnits():FindID(unitId)
    if not unit then
        return false, ERROR_CODES.UNIT_NOT_FOUND, "Unit not found"
    end

    if UnitManager and UnitManager.RequestCommand then
        UnitManager.RequestCommand(unit, UnitCommandTypes.FORTIFY)
        return true, nil, "Unit fortified"
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Fortify execution failed"
end

-- Skip unit turn
local function executeUnitSkip(playerId, command)
    local unitId = command.unit_id

    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end

    if Log then Log.Info("Executing: UNIT_SKIP unit=%d", unitId) end

    local player = Players[playerId]
    local unit = player and player:GetUnits():FindID(unitId)

    if not unit then
        return false, ERROR_CODES.UNIT_NOT_FOUND, "Unit not found"
    end

    if UnitManager and UnitManager.RequestCommand then
        UnitManager.RequestCommand(unit, UnitCommandTypes.WAIT)
        return true, nil, "Unit skipped"
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Skip execution failed"
end

-- Found a city with a settler
local function executeFoundCity(playerId, command)
    local unitId = command.unit_id

    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end

    if Log then Log.Info("Executing: UNIT_FOUND_CITY unit=%d", unitId) end

    local player = Players[playerId]
    local unit = player and player:GetUnits():FindID(unitId)

    if not unit then
        return false, ERROR_CODES.UNIT_NOT_FOUND, "Unit not found"
    end

    if UnitManager and UnitManager.RequestOperation then
        UnitManager.RequestOperation(unit, UnitOperationTypes.FOUND_CITY)
        return true, nil, "City founded"
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Found city execution failed"
end

-- Build improvement
local function executeUnitBuild(playerId, command)
    local unitId = command.unit_id
    local itemId = command.item_id

    if not unitId or not itemId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id or item_id"
    end

    if Log then Log.Info("Executing: UNIT_BUILD unit=%d item=%s", unitId, tostring(itemId)) end

    local player = Players[playerId]
    local unit = player and player:GetUnits():FindID(unitId)

    if not unit then
        return false, ERROR_CODES.UNIT_NOT_FOUND, "Unit not found"
    end

    if UnitManager and UnitManager.RequestOperation then
        local params = {}
        params[UnitOperationTypes.PARAM_IMPROVEMENT_TYPE] = itemId
        UnitManager.RequestOperation(unit, UnitOperationTypes.BUILD_IMPROVEMENT, params)
        return true, nil, "Build started"
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Build execution failed"
end

-- Set city production
local function executeCityProduce(playerId, command)
    local cityId = command.city_id
    local itemId = command.item_id
    local itemType = command.item_type  -- "unit", "building", "wonder", "district"

    if not cityId or not itemId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing city_id or item_id"
    end

    if Log then Log.Info("Executing: CITY_PRODUCE city=%d item=%s type=%s", cityId, tostring(itemId), tostring(itemType)) end

    local player = Players[playerId]
    if not player then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player not found"
    end

    local city = player:GetCities():FindID(cityId)
    if not city then
        return false, ERROR_CODES.CITY_NOT_FOUND, "City not found"
    end

    local buildQueue = city:GetBuildQueue()
    if not buildQueue then
        return false, ERROR_CODES.EXECUTION_FAILED, "City has no build queue"
    end

    -- Add to production queue based on type
    local success = false
    if itemType == "unit" or not itemType then
        -- Try as unit first
        if buildQueue.AddUnitType then
            success = buildQueue:AddUnitType(itemId)
        end
    elseif itemType == "building" then
        if buildQueue.AddBuildingType then
            success = buildQueue:AddBuildingType(itemId)
        end
    elseif itemType == "district" then
        if buildQueue.AddDistrictType then
            success = buildQueue:AddDistrictType(itemId)
        end
    elseif itemType == "wonder" then
        if buildQueue.AddWonderType then
            success = buildQueue:AddWonderType(itemId)
        end
    end

    if success then
        return true, nil, string.format("City %d producing %s", cityId, tostring(itemId))
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Production set failed"
end

-- Research a technology
local function executeResearchTech(playerId, command)
    local techId = command.item_id

    if not techId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing item_id (tech)"
    end

    if Log then Log.Info("Executing: RESEARCH_TECH tech=%s", tostring(techId)) end

    local player = Players[playerId]
    if not player then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player not found"
    end

    local playerTech = player:GetTechs()
    if not playerTech then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player tech not available"
    end

    if playerTech.SetResearchingTech then
        playerTech:SetResearchingTech(techId)
        return true, nil, "Researching tech: " .. tostring(techId)
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Tech research failed"
end

-- Research a civic
local function executeResearchCivic(playerId, command)
    local civicId = command.item_id

    if not civicId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing item_id (civic)"
    end

    if Log then Log.Info("Executing: RESEARCH_CIVIC civic=%s", tostring(civicId)) end

    local player = Players[playerId]
    if not player then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player not found"
    end

    local playerCulture = player:GetCulture()
    if not playerCulture then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player culture not available"
    end

    if playerCulture.SetProgressingCivic then
        playerCulture:SetProgressingCivic(civicId)
        return true, nil, "Researching civic: " .. tostring(civicId)
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Civic research failed"
end

-- Action dispatch table
local ACTION_HANDLERS = {
    [ACTION_TYPES.END_TURN] = executeEndTurn,
    [ACTION_TYPES.UNIT_MOVE] = executeUnitMove,
    [ACTION_TYPES.UNIT_ATTACK] = executeUnitAttack,
    [ACTION_TYPES.UNIT_FORTIFY] = executeUnitFortify,
    [ACTION_TYPES.UNIT_SKIP] = executeUnitSkip,
    [ACTION_TYPES.UNIT_SLEEP] = executeUnitSkip,  -- Alias
    [ACTION_TYPES.UNIT_FOUND_CITY] = executeFoundCity,
    [ACTION_TYPES.UNIT_BUILD] = executeUnitBuild,
    [ACTION_TYPES.CITY_PRODUCE] = executeCityProduce,
    [ACTION_TYPES.RESEARCH_TECH] = executeResearchTech,
    [ACTION_TYPES.RESEARCH_CIVIC] = executeResearchCivic,
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
    if AlphaZeroConfig.Get("VALIDATE_COMMANDS") then
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

-- Execute multiple commands from a CommandMessage
function AlphaZeroActionExecutor.ExecuteCommands(playerId, commandMessage)
    local results = {
        msg_id = commandMessage.msg_id,
        success = true,
        executed = {},
        errors = {},
    }

    local commands = commandMessage.commands or {}
    local maxActions = AlphaZeroConfig.Get("MAX_ACTIONS_PER_TURN") or 1000

    for i, command in ipairs(commands) do
        if i > maxActions then
            if Log then Log.Warn("Max actions per turn exceeded") end
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

-- Validate a command without executing
function AlphaZeroActionExecutor.Validate(playerId, command)
    if not command or not command.action_type then
        return false, "Missing command or action_type"
    end

    local actionType = command.action_type

    -- Type-specific validation
    if actionType == ACTION_TYPES.UNIT_MOVE or
       actionType == ACTION_TYPES.UNIT_ATTACK or
       actionType == ACTION_TYPES.UNIT_FORTIFY or
       actionType == ACTION_TYPES.UNIT_SKIP then

        if not command.unit_id then
            return false, "Missing unit_id for unit action"
        end

        local player = Players[playerId]
        if player then
            local unit = player:GetUnits():FindID(command.unit_id)
            if not unit then
                return false, "Unit not found: " .. tostring(command.unit_id)
            end
        end
    end

    if actionType == ACTION_TYPES.CITY_PRODUCE then
        if not command.city_id then
            return false, "Missing city_id for city action"
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

-- Initialize on load
AlphaZeroActionExecutor.Init()

print("[AlphaZero] ActionExecutor module loaded")
