-- ============================================================================
-- ActionExecutor.lua - Command execution for AlphaZero AI Bridge
-- ============================================================================
-- Executes commands received from the Python agent.
-- Uses safe accessors and correct Civ6 API calls (RequestOperation pattern).
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
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED",
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

-- End the current player's turn
local function executeEndTurn(playerId, command)
    if Log then Log.Info("Executing: END_TURN") end

    -- In Civ6, ending turn is handled through the UI
    if UI and UI.RequestAction and ActionTypes then
        UI.RequestAction(ActionTypes.ACTION_ENDTURN)
        return true, nil, "Turn ended"
    end

    return true, nil, "End turn requested"
end

-- Move a unit to target position using RequestOperation (correct Civ6 API)
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

    -- Execute move using UnitManager.RequestOperation (correct Civ6 API)
    if UnitManager and UnitManager.RequestOperation and UnitOperationTypes then
        local params = {}
        params[UnitOperationTypes.PARAM_X] = target.x
        params[UnitOperationTypes.PARAM_Y] = target.y
        UnitManager.RequestOperation(unit, UnitOperationTypes.MOVETO, params)
        return true, nil, string.format("Unit %d moving to (%d,%d)", unitId, target.x, target.y)
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Move execution failed: UnitManager.RequestOperation not available"
end

-- Attack a target position with a unit
local function executeUnitAttack(playerId, command)
    local unitId = command.unit_id
    local target = command.target

    if not unitId or not target then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id or target"
    end

    if Log then Log.Info("Executing: UNIT_ATTACK unit=%d at (%d,%d)", unitId, target.x, target.y) end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    -- Check attacks remaining
    if (unit:GetAttacksRemaining() or 0) <= 0 then
        return false, ERROR_CODES.NOT_ENOUGH_MOVEMENT, "Unit has no attacks remaining"
    end

    -- Execute attack using RequestOperation
    if UnitManager and UnitManager.RequestOperation and UnitOperationTypes then
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

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    if UnitManager and UnitManager.RequestCommand and UnitCommandTypes then
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

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    if UnitManager and UnitManager.RequestCommand and UnitCommandTypes then
        UnitManager.RequestCommand(unit, UnitCommandTypes.WAIT)
        return true, nil, "Unit skipped"
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Skip execution failed"
end

-- Sleep unit (distinct from skip)
local function executeUnitSleep(playerId, command)
    local unitId = command.unit_id

    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end

    if Log then Log.Info("Executing: UNIT_SLEEP unit=%d", unitId) end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    if UnitManager and UnitManager.RequestCommand and UnitCommandTypes then
        UnitManager.RequestCommand(unit, UnitCommandTypes.SLEEP)
        return true, nil, "Unit sleeping"
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Sleep execution failed"
end

-- Found a city with a settler
local function executeFoundCity(playerId, command)
    local unitId = command.unit_id

    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end

    if Log then Log.Info("Executing: UNIT_FOUND_CITY unit=%d", unitId) end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    if UnitManager and UnitManager.RequestOperation and UnitOperationTypes then
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

-- Item type dispatch tables for production commands.
-- Maps item_type -> CityOperationTypes param key (for CityManager API)
local PRODUCTION_PARAM_KEYS = {
    unit = "PARAM_UNIT_TYPE",
    building = "PARAM_BUILDING_TYPE",
    wonder = "PARAM_BUILDING_TYPE",  -- Wonders use building param
    district = "PARAM_DISTRICT_TYPE",
}

-- Maps item_type -> buildQueue method name (for fallback API)
local PRODUCTION_QUEUE_METHODS = {
    unit = "AddUnitType",
    building = "AddBuildingType",
    wonder = "AddWonderType",
    district = "AddDistrictType",
}

-- Set city production using CityManager.RequestOperation (correct Civ6 API)
local function executeCityProduce(playerId, command)
    local cityId = command.city_id
    local itemId = command.item_id
    local itemType = command.item_type or "unit"  -- default to unit

    if not cityId or not itemId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing city_id or item_id"
    end

    if Log then Log.Info("Executing: CITY_PRODUCE city=%d item=%s type=%s", cityId, tostring(itemId), tostring(itemType)) end

    local city, errCode, errMsg = safeGetCity(playerId, cityId)
    if not city then
        return false, errCode, errMsg
    end

    -- Primary method: CityManager.RequestOperation (correct Civ6 API)
    if CityManager and CityManager.RequestOperation and CityOperationTypes then
        local paramKey = PRODUCTION_PARAM_KEYS[itemType] or PRODUCTION_PARAM_KEYS["unit"]
        local params = {}
        params[CityOperationTypes[paramKey]] = itemId
        CityManager.RequestOperation(city, CityOperationTypes.BUILD, params)
        return true, nil, string.format("City %d producing %s (%s)", cityId, tostring(itemId), tostring(itemType))
    end

    -- Fallback: buildQueue methods (for different Civ6 mod contexts)
    local buildQueue = city:GetBuildQueue()
    if buildQueue then
        local methodName = PRODUCTION_QUEUE_METHODS[itemType] or PRODUCTION_QUEUE_METHODS["unit"]
        local method = buildQueue[methodName]
        if method then
            local ok, result = pcall(function() return method(buildQueue, itemId) end)
            if ok and result then
                return true, nil, string.format("City %d producing %s (fallback)", cityId, tostring(itemId))
            end
        end
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

    if not Players then
        return false, ERROR_CODES.EXECUTION_FAILED, "Players table not available"
    end

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

    if not Players then
        return false, ERROR_CODES.EXECUTION_FAILED, "Players table not available"
    end

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

-- Stub handler for unimplemented actions
local function executeNotImplemented(playerId, command)
    local actionType = command and command.action_type or "unknown"
    local msg = string.format("Action type '%s' is not yet implemented", tostring(actionType))
    if Log then Log.Warn(msg) end
    return false, ERROR_CODES.NOT_IMPLEMENTED, msg
end

-- Action dispatch table
local ACTION_HANDLERS = {
    [ACTION_TYPES.END_TURN] = executeEndTurn,
    [ACTION_TYPES.UNIT_MOVE] = executeUnitMove,
    [ACTION_TYPES.UNIT_ATTACK] = executeUnitAttack,
    [ACTION_TYPES.UNIT_FORTIFY] = executeUnitFortify,
    [ACTION_TYPES.UNIT_SKIP] = executeUnitSkip,
    [ACTION_TYPES.UNIT_SLEEP] = executeUnitSleep,
    [ACTION_TYPES.UNIT_FOUND_CITY] = executeFoundCity,
    [ACTION_TYPES.UNIT_BUILD] = executeUnitBuild,
    [ACTION_TYPES.CITY_PRODUCE] = executeCityProduce,
    [ACTION_TYPES.CITY_BUY] = executeNotImplemented,
    [ACTION_TYPES.CITY_CITIZEN] = executeNotImplemented,
    [ACTION_TYPES.RESEARCH_TECH] = executeResearchTech,
    [ACTION_TYPES.RESEARCH_CIVIC] = executeResearchCivic,
    [ACTION_TYPES.UNIT_PROMOTE] = executeNotImplemented,
    [ACTION_TYPES.UNIT_DELETE] = executeNotImplemented,
    [ACTION_TYPES.DIPLOMACY] = executeNotImplemented,
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
