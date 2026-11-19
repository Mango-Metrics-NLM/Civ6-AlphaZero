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

-- Shared attack body. Issues the named Civ6 UnitOperation against the
-- target tile. `opName` must be a key on UnitOperationTypes (e.g.
-- "MOVE_TO" for melee engagement, "RANGE_ATTACK" for ranged units).
local function _executeAttackOperation(playerId, command, opName, logLabel)
    local unitId = command.unit_id
    local target = command.target

    if not unitId or not target then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id or target"
    end

    if Log then
        Log.Info("Executing: %s unit=%d at (%d,%d)", logLabel, unitId, target.x, target.y)
    end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    -- Check attacks remaining (applies to both melee and ranged).
    if (unit:GetAttacksRemaining() or 0) <= 0 then
        return false, ERROR_CODES.NOT_ENOUGH_MOVEMENT, "Unit has no attacks remaining"
    end

    if UnitManager and UnitManager.RequestOperation and UnitOperationTypes then
        local op = UnitOperationTypes[opName]
        if op == nil then
            return false, ERROR_CODES.EXECUTION_FAILED,
                string.format("UnitOperationTypes.%s not available", opName)
        end
        local params = {}
        params[UnitOperationTypes.PARAM_X] = target.x
        params[UnitOperationTypes.PARAM_Y] = target.y
        UnitManager.RequestOperation(unit, op, params)
        return true, nil, string.format("%s executed at (%d,%d)", logLabel, target.x, target.y)
    end

    return false, ERROR_CODES.EXECUTION_FAILED, "Attack execution failed"
end

-- Melee attack: move onto the target tile to engage. Civ6's MOVE_TO
-- operation against an occupied enemy tile resolves to a melee attack.
-- (Hardening fix for gap-analysis C1 on commit 3572832: previously this
-- routed through RANGE_ATTACK, which silent-no-ops for melee units.)
local function executeUnitAttack(playerId, command)
    return _executeAttackOperation(playerId, command, "MOVE_TO", "UNIT_ATTACK")
end

-- Ranged attack: explicit RANGE_ATTACK operation for archers / catapults /
-- bombards. Distinct from melee per Python ActionType.UNIT_RANGED_ATTACK.
local function executeUnitRangedAttack(playerId, command)
    return _executeAttackOperation(playerId, command, "RANGE_ATTACK", "UNIT_RANGED_ATTACK")
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

-- Delete (disband) a unit.
local function executeUnitDelete(playerId, command)
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

    -- Primary path: GetDiplomacy():DoAction(...). Civ6 expects a string or
    -- int action constant from DiplomaticActions table; pass civ6ActionName.
    if diplomacy.DoAction then
        local ok = pcall(function() diplomacy:DoAction(civ6ActionName, targetPlayerId) end)
        if ok then
            return true, nil, string.format("%s against player %d", subActionName, targetPlayerId)
        end
    end

    -- Fallback: UI.RequestDiplomacyMove for older mod contexts.
    if UI and UI.RequestDiplomacyMove then
        local ok = pcall(function() UI.RequestDiplomacyMove(civ6ActionName, targetPlayerId) end)
        if ok then
            return true, nil, string.format("%s requested via UI (player %d)",
                subActionName, targetPlayerId)
        end
    end

    return false, ERROR_CODES.EXECUTION_FAILED,
        string.format("No diplomacy API available for %s", subActionName)
end

local function diplomacyDeclareWar(playerId, targetPlayerId)
    return _diplomacyRequest(playerId, targetPlayerId, "DIPLOACTION_DECLARE_WAR", "DECLARE_WAR")
end

local function diplomacyMakePeace(playerId, targetPlayerId)
    return _diplomacyRequest(playerId, targetPlayerId, "DIPLOACTION_MAKE_PEACE", "MAKE_PEACE")
end

local function diplomacyDenounce(playerId, targetPlayerId)
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
