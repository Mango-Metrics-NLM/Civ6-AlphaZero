-- ============================================================================
-- civ6_stub.lua - a hand-written stand-in for the Civilization VI Lua globals
-- ============================================================================
-- Test fixture only. It lets the AlphaZeroAI mod's Lua run outside the game, so
-- tests can execute it instead of reading it as text.
--
-- Everything here is INVENTED: a 4x3 map, two players, three units, one city and
-- a handful of type rows, costs and constants. No game data. Names follow the game's naming style
-- (UNIT_WARRIOR, TECH_POTTERY) only so the mod's string checks behave as they do
-- in the game.
--
-- It models only what the mod calls today. A method the mod calls and this file
-- lacks does NOT raise in the exporter: every export component is wrapped in
-- pcall and falls back to an empty value. Tests must therefore assert that the
-- exported components are non-empty, or a hole here reads as a pass.
--
-- Civ6Stub.calls records every call that would change the game (UI requests,
-- setters, notification changes), so a test can assert what the mod asked the
-- game to do.
--
-- The invented game's changeable state lives in tables on Civ6Stub and is read
-- when a method is CALLED, so a test can change it after the mod has loaded:
--   Civ6Stub.players[id].tech / .civic   research state per player
--   Civ6Stub.cities[cityId]              buildings and districts of a city
--   Civ6Stub.notifications[id]           notifications; none by default
--   Civ6Stub.canEndTurn                  nil: derived from the notifications;
--                                        true / false: forced answer
-- UI.RequestPlayerOperation records and queues: like the game's, it changes
-- nothing by the time it returns. Civ6Stub.applyRequests() plays the "next
-- frame": a queued research request takes effect only if its type parameter
-- is the row's Hash (what the game's own choosers send) and the player can
-- research it now; anything else is dropped without an error, as the game
-- drops it. ASSUMPTION, not checked against a game: when research is set this
-- way the stub also dismisses the "choose research" notification, as the
-- game's own notification system is taken to do (its chooser does not dismiss
-- anything itself). The GameCore setters (SetResearchingTech,
-- SetProgressingCivic) record and change the state at once, and touch no
-- notification.
-- ============================================================================

Civ6Stub = { calls = {} }

local function record(name, ...)
    table.insert(Civ6Stub.calls, { name = name, args = { ... } })
end

local function shallowCopy(source)
    local copy = {}
    for key, value in pairs(source or {}) do copy[key] = value end
    return copy
end

-- GameInfo tables are called and iterated: `for row in GameInfo.Units() do`.
local function rows(list)
    return function()
        local i = 0
        return function()
            i = i + 1
            return list[i]
        end
    end
end

GameInfo = {
    Units = rows({
        { Index = 0, Hash = 6100, UnitType = "UNIT_SETTLER", Combat = 0, FormationClass = "FORMATION_CLASS_CIVILIAN",
          Cost = 80, BaseMoves = 2 },
        { Index = 1, Hash = 6111, UnitType = "UNIT_WARRIOR", Combat = 20, FormationClass = "FORMATION_CLASS_LAND_COMBAT",
          Cost = 40, BaseMoves = 2 },
        { Index = 2, Hash = 6122, UnitType = "UNIT_ARCHER", Combat = 15, FormationClass = "FORMATION_CLASS_LAND_COMBAT",
          Cost = 60, BaseMoves = 2, PrereqTech = "TECH_BRONZE_WORKING" },
    }),
    Buildings = rows({
        { Index = 0, Hash = 6200, BuildingType = "BUILDING_MONUMENT", Cost = 60 },
        { Index = 1, Hash = 6211, BuildingType = "BUILDING_GRANARY", Cost = 65, PrereqTech = "TECH_POTTERY" },
    }),
    Districts = rows({
        { Index = 0, Hash = 6300, DistrictType = "DISTRICT_CITY_CENTER" },
        { Index = 1, Hash = 6311, DistrictType = "DISTRICT_CAMPUS" },
    }),
    Terrains = rows({
        { Index = 0, TerrainType = "TERRAIN_GRASS" },
        { Index = 1, TerrainType = "TERRAIN_PLAINS" },
        { Index = 2, TerrainType = "TERRAIN_GRASS_MOUNTAIN" },
    }),
    Features = rows({
        { Index = 0, FeatureType = "FEATURE_FOREST" },
    }),
    Resources = rows({
        { Index = 0, ResourceType = "RESOURCE_WHEAT" },
    }),
    Improvements = rows({
        { Index = 0, ImprovementType = "IMPROVEMENT_FARM" },
    }),
    -- Hash values are invented and deliberately far from the indexes, so a
    -- request that sends an index where a hash is due matches nothing.
    Civics = rows({
        { Index = 0, Hash = 8100, CivicType = "CIVIC_CODE_OF_LAWS", Cost = 20, EraType = "ERA_ANCIENT" },
        { Index = 1, Hash = 8111, CivicType = "CIVIC_CRAFTSMANSHIP", Cost = 40, EraType = "ERA_ANCIENT" },
        { Index = 2, Hash = 8122, CivicType = "CIVIC_FOREIGN_TRADE", Cost = 40, EraType = "ERA_ANCIENT" },
        { Index = 3, Hash = 8133, CivicType = "CIVIC_EARLY_EMPIRE", Cost = 70, EraType = "ERA_ANCIENT" },
    }),
    -- Craftsmanship and Foreign Trade need Code of Laws; Early Empire needs
    -- Foreign Trade.
    CivicPrereqs = rows({
        { Civic = "CIVIC_CRAFTSMANSHIP", PrereqCivic = "CIVIC_CODE_OF_LAWS" },
        { Civic = "CIVIC_FOREIGN_TRADE", PrereqCivic = "CIVIC_CODE_OF_LAWS" },
        { Civic = "CIVIC_EARLY_EMPIRE", PrereqCivic = "CIVIC_FOREIGN_TRADE" },
    }),
    Technologies = rows({
        { Index = 0, Hash = 7100, TechnologyType = "TECH_POTTERY", Cost = 25, EraType = "ERA_ANCIENT" },
        { Index = 1, Hash = 7111, TechnologyType = "TECH_MINING", Cost = 50, EraType = "ERA_ANCIENT" },
        { Index = 2, Hash = 7122, TechnologyType = "TECH_WRITING", Cost = 60, EraType = "ERA_ANCIENT" },
        { Index = 3, Hash = 7133, TechnologyType = "TECH_BRONZE_WORKING", Cost = 120, EraType = "ERA_CLASSICAL" },
    }),
    -- Bronze Working needs Mining.
    TechnologyPrereqs = rows({
        { Technology = "TECH_BRONZE_WORKING", PrereqTech = "TECH_MINING" },
    }),
}

YieldTypes = {
    YIELD_FOOD = 0, YIELD_PRODUCTION = 1, YIELD_GOLD = 2,
    YIELD_SCIENCE = 3, YIELD_CULTURE = 4, YIELD_FAITH = 5,
}

ActionTypes = { ACTION_ENDTURN = 7 }

-- Invented numbers; only their identity matters.
PlayerOperations = {
    RESEARCH = 11, PROGRESS_CIVIC = 12,
    PARAM_TECH_TYPE = 501, PARAM_CIVIC_TYPE = 502, PARAM_INSERT_MODE = 503,
    VALUE_EXCLUSIVE = 601, VALUE_APPEND = 602,
}

EndTurnBlockingTypes = {
    NO_ENDTURN_BLOCKING = 0,
    ENDTURN_BLOCKING_UNITS = 1,
    ENDTURN_BLOCKING_PRODUCTION = 2,
    ENDTURN_BLOCKING_RESEARCH = 3,
    ENDTURN_BLOCKING_CIVIC = 4,
}

-- ----------------------------------------------------------------------------
-- Map: 4 wide, 3 high. (3, 2) is a mountain; (1, 1) is a forested hill owned by
-- player 0 with wheat and a farm.
-- ----------------------------------------------------------------------------
local MAP_W, MAP_H = 4, 3

local function makePlot(x, y)
    local mountain = (x == 3 and y == 2)
    local special = (x == 1 and y == 1)
    local plot = {}
    function plot:GetX() return x end
    function plot:GetY() return y end
    function plot:GetTerrainType() return mountain and 2 or ((x + y) % 2) end
    function plot:GetFeatureType() return special and 0 or -1 end
    function plot:GetResourceType() return special and 0 or -1 end
    function plot:GetImprovementType() return special and 0 or -1 end
    function plot:GetOwner() return special and 0 or -1 end
    function plot:IsHills() return special end
    function plot:IsMountain() return mountain end
    function plot:IsImpassable() return mountain end
    function plot:GetYield(yieldType) return (yieldType == 0) and 2 or ((yieldType == 1) and 1 or 0) end
    return plot
end

local plots = {}
for x = 0, MAP_W - 1 do
    plots[x] = {}
    for y = 0, MAP_H - 1 do
        plots[x][y] = makePlot(x, y)
    end
end

Map = {}
function Map.GetGridSize() return MAP_W, MAP_H end
function Map.GetPlot(x, y)
    return plots[x] and plots[x][y] or nil
end

-- ----------------------------------------------------------------------------
-- Units and cities
-- ----------------------------------------------------------------------------
-- One object per record, kept: a test that replaces a method on a unit or a
-- city it got from the stand-in must find the same object again.
local objects = setmetatable({}, { __mode = "k" })

local function makeUnit(spec)
    if objects[spec] then return objects[spec] end
    local unit = { _spec = spec }
    objects[spec] = unit
    function unit:GetComponentID() return { player = spec.owner, id = spec.id, type = 1 } end
    function unit:GetMovesRemaining() return spec.moves end
    function unit:GetID() return spec.id end
    function unit:GetOwner() return spec.owner end
    function unit:GetX() return spec.x end
    function unit:GetY() return spec.y end
    function unit:GetUnitType() return spec.unitType end
    function unit:GetDamage() return spec.damage or 0 end
    function unit:GetMovementMovesRemaining() return spec.moves end
    function unit:GetMaxMoves() return 2 end
    function unit:GetCombat() return spec.combat or 0 end
    function unit:GetRangedCombat() return spec.ranged or 0 end
    function unit:GetRange() return spec.range or 0 end
    function unit:GetAttacksRemaining() return spec.attacks or 0 end
    function unit:IsCivilian() return spec.unitType == 0 end
    function unit:GetExperience()
        local exp = {}
        function exp:GetPromotions() return {} end
        return exp
    end
    return unit
end

-- Collections are iterated as `for i, member in collection:Members() do`.
local function collection(list)
    local c = {}
    function c:Members()
        local i = 0
        return function()
            i = i + 1
            if list[i] then return i, list[i] end
        end
    end
    function c:GetCount() return #list end
    -- The game's collections find a member by id (units, cities).
    function c:FindID(id)
        for _, member in ipairs(list) do
            if member.GetID and member:GetID() == id then return member end
        end
        return nil
    end
    return c
end

-- One changeable record per unit. Unit objects read their record when a method
-- is called, so a test (or an applied request) that changes a record changes
-- what the mod sees. Extra fields used by the command stand-in
-- (civ6_stub_commands.lua): activity ("fortified" / "sleeping"), skipped.
Civ6Stub.units = {
    { id = 101, owner = 0, x = 0, y = 0, unitType = 0, moves = 2 },                           -- settler
    { id = 102, owner = 0, x = 1, y = 0, unitType = 1, moves = 2, combat = 20, attacks = 1 }, -- warrior
    { id = 201, owner = 1, x = 2, y = 0, unitType = 1, moves = 2, combat = 20, attacks = 1 }, -- enemy warrior
}
local function allUnits()
    local list = {}
    for _, spec in ipairs(Civ6Stub.units) do table.insert(list, makeUnit(spec)) end
    return list
end

local function unitsOf(owner)
    local list = {}
    for _, unit in ipairs(allUnits()) do
        if unit:GetOwner() == owner then table.insert(list, unit) end
    end
    return collection(list)
end

Units = {}
function Units.GetUnitsInPlot(plot)
    local list = {}
    for _, unit in ipairs(allUnits()) do
        if unit:GetX() == plot:GetX() and unit:GetY() == plot:GetY() then
            table.insert(list, unit)
        end
    end
    return collection(list)
end

-- Stubville has a monument (building 0) and two districts: the city centre
-- (district 0) and a campus (district 1). A district entry is its type, or
-- { type = n, complete = false } for one still under construction.
Civ6Stub.cities = {
    [301] = { buildings = { [0] = true }, districts = { 0, 1 } },
}

local function makeDistrict(entry, x, y)
    local districtType, complete = entry, true
    if type(entry) == "table" then
        districtType, complete = entry.type, entry.complete ~= false
    end
    local district = {}
    function district:GetType() return districtType end
    function district:GetX() return x end
    function district:GetY() return y end
    function district:IsComplete() return complete end
    function district:IsPillaged() return false end
    return district
end

local function makeCity(spec)
    if objects[spec] then return objects[spec] end
    local city = { _spec = spec }
    objects[spec] = city
    local queue = {}
    local function state() return Civ6Stub.cities[spec.id] or {} end
    function city:GetBuildings()
        local buildings = {}
        function buildings:HasBuilding(index) return (state().buildings or {})[index] == true end
        function buildings:IsPillaged(_) return false end
        return buildings
    end
    function city:GetDistricts()
        local list = {}
        for _, districtType in ipairs(state().districts or {}) do
            table.insert(list, makeDistrict(districtType, spec.x, spec.y))
        end
        return collection(list)
    end
    function queue:CurrentlyBuilding() return spec.building end
    function queue:GetCurrentProductionTypeHash()
        if not spec.building then return 0 end
        for _, kind in ipairs({
            { rows = GameInfo.Units, col = "UnitType" },
            { rows = GameInfo.Buildings, col = "BuildingType" },
            { rows = GameInfo.Districts, col = "DistrictType" },
        }) do
            if kind.rows then
                for row in kind.rows() do
                    if row[kind.col] == spec.building and row.Hash ~= nil then
                        return row.Hash
                    end
                end
            end
        end
        return 0
    end
    function queue:GetTurnsLeft() return 3 end
    -- Asked by row Hash (what the game's own production panel sends):
    -- `CanProduce(hash, true)` is the unlock check, `CanProduce(hash, false, true)`
    -- the strict exclusion check ("can start now"). `"later"` passes only `true`;
    -- `"locked"` mimics live Civ6 unresearched items where `(hash, true)` is false
    -- even though `(hash, false, true)` returns true.
    -- Anything that is not the Hash of a producible row, a row Index included,
    -- answers false: the game knows nothing by that number.
    function queue:CanProduce(ref, loose, strictFlag)
        local name = Civ6Stub.producibleName and Civ6Stub.producibleName(ref)
        if not name then return false end
        local table_ = state().canProduce or Civ6Stub.defaultCanProduce
        local entry = table_[name]
        if entry == nil then return false end
        if loose == false and strictFlag == true then
            return entry == "now" or entry == "locked"
        end
        return entry ~= "locked"
    end
    function city:GetID() return spec.id end
    function city:GetName() return spec.name end
    function city:GetX() return spec.x end
    function city:GetY() return spec.y end
    function city:GetPopulation() return spec.population end
    function city:IsCapital() return true end
    function city:GetStrengthValue() return 10 end
    function city:GetBuildQueue() return queue end
    return city
end

-- ----------------------------------------------------------------------------
-- Players. Player 0 has researched TECH_POTTERY (index 0) and CIVIC_CODE_OF_LAWS
-- (index 0), is researching TECH_MINING (index 1; 20 of 50 done, 8 turns left,
-- boost triggered), is working on CIVIC_CRAFTSMANSHIP (index 1; 10 of 40 done,
-- 15 turns left, no boost) and has one city. It can research TECH_MINING and
-- TECH_WRITING now; TECH_BRONZE_WORKING is out of reach. It can progress
-- CIVIC_CRAFTSMANSHIP and CIVIC_FOREIGN_TRADE; CIVIC_EARLY_EMPIRE is out of
-- reach (`can` is what the game answers; the prerequisite table agrees).
-- ----------------------------------------------------------------------------
local function researchState()
    return {
        tech = {
            current = 1,
            has = { [0] = true },
            can = { [1] = true, [2] = true },
            cost = { [0] = 25, [1] = 50, [2] = 60, [3] = 120 },
            progress = { [0] = 25, [1] = 20 },
            turns = { [1] = 8, [2] = 15, [3] = 30 },
            boosted = { [1] = true },
        },
        civic = {
            current = 1,
            has = { [0] = true },
            can = { [1] = true, [2] = true },
            cost = { [0] = 20, [1] = 40, [2] = 40, [3] = 70 },
            progress = { [0] = 20, [1] = 10 },
            turns = { [1] = 15, [2] = 20, [3] = 35 },
            turnsLeft = 15,
            boosted = {},
        },
    }
end

Civ6Stub.players = { [0] = researchState(), [1] = researchState() }
-- One changeable record per city, by owner; city objects read it when called.
-- `building` is what the city is producing (a type name), nil for nothing.
Civ6Stub.cityRecords = {
    [0] = { { id = 301, name = "Stubville", x = 1, y = 1, population = 3 } },
    [1] = {},
}
-- What a city can produce unless Civ6Stub.cities[id].canProduce says otherwise:
-- "now" passes the strict check, "later" only the loose one.
Civ6Stub.defaultCanProduce = {
    UNIT_WARRIOR = "now", BUILDING_GRANARY = "now", DISTRICT_CAMPUS = "now", UNIT_SETTLER = "later",
}
local function makePlayer(id)
    local player = {}
    function player:GetID() return id end
    function player:IsAlive() return true end
    function player:IsHuman() return id == 0 end
    function player:GetUnits() return unitsOf(id) end
    function player:GetCities()
        local list = {}
        for _, spec in ipairs(Civ6Stub.cityRecords[id] or {}) do table.insert(list, makeCity(spec)) end
        return collection(list)
    end
    function player:GetScore() return 10 + id end
    function player:GetTreasury()
        local t = {}
        function t:GetGoldBalance() return 50 end
        function t:GetGoldYield() return 3 end
        return t
    end
    function player:GetCulture()
        local c = {}
        local function state() return Civ6Stub.players[id].civic end
        function c:GetCultureYield() return 2 end
        function c:GetProgressingCivic() return state().current end
        function c:GetResearchedCivics()
            local list = {}
            for index = 0, 3 do
                if state().has[index] then table.insert(list, index) end
            end
            return list
        end
        function c:HasCivic(index) return state().has[index] == true end
        function c:GetCultureCost(index) return state().cost[index] or 0 end
        function c:GetCulturalProgress(index) return state().progress[index] or 0 end
        function c:GetTurnsLeftOnCurrentCivic() return state().turnsLeft end
        function c:GetTurnsToProgressCivic(index) return state().turns[index] or -1 end
        function c:CanProgress(index) return state().can[index] == true end
        function c:HasBoostBeenTriggered(index) return state().boosted[index] == true end
        function c:SetProgressingCivic(index)
            record("SetProgressingCivic", id, index)
            state().current = index
        end
        return c
    end
    function player:GetTechs()
        local t = {}
        local function state() return Civ6Stub.players[id].tech end
        function t:GetScienceYield() return 4 end
        function t:GetResearchingTech() return state().current end
        function t:HasTech(index) return state().has[index] == true end
        function t:CanResearch(index) return state().can[index] == true end
        function t:GetResearchCost(index) return state().cost[index] or 0 end
        function t:GetResearchProgress(index) return state().progress[index] or 0 end
        function t:GetTurnsToResearch(index) return state().turns[index] or -1 end
        function t:HasBoostBeenTriggered(index) return state().boosted[index] == true end
        function t:SetResearchingTech(index)
            record("SetResearchingTech", id, index)
            state().current = index
        end
        return t
    end
    function player:GetReligion()
        local r = {}
        function r:GetFaithBalance() return 0 end
        return r
    end
    return player
end

Players = {
    [0] = makePlayer(0),
    [1] = makePlayer(1),
}

PlayerConfigurations = {}
for id, name in pairs({ [0] = "CIVILIZATION_STUB_A", [1] = "CIVILIZATION_STUB_B" }) do
    local config = {}
    function config:GetCivilizationTypeName() return name end
    PlayerConfigurations[id] = config
end

PlayerManager = {}
function PlayerManager.GetWasEverAliveCount() return 2 end

Game = {}
function Game.GetCurrentGameTurn() return 12 end
function Game.GetLocalPlayer() return 0 end
function Game.GetEras()
    local eras = {}
    function eras:GetCurrentEra() return 1 end
    return eras
end

-- ----------------------------------------------------------------------------
-- UI and events: record, do nothing.
-- ----------------------------------------------------------------------------
-- Notifications. Each is { id, blocking (an EndTurnBlockingTypes value),
-- message, typeName, dismissed }. None by default, so the turn can end.
Civ6Stub.notifications = { [0] = {}, [1] = {} }
Civ6Stub.canEndTurn = nil

local function findNotification(player, notificationId)
    for _, notification in ipairs(Civ6Stub.notifications[player] or {}) do
        if notification.id == notificationId then return notification end
    end
    return nil
end

NotificationManager = {}
function NotificationManager.GetList(player)
    local ids = {}
    for _, notification in ipairs(Civ6Stub.notifications[player] or {}) do
        table.insert(ids, notification.id)
    end
    return ids
end
function NotificationManager.Find(player, notificationId)
    local notification = findNotification(player, notificationId)
    if not notification then return nil end
    local entry = {}
    function entry:IsDismissed() return notification.dismissed == true end
    function entry:GetEndTurnBlocking() return notification.blocking or 0 end
    function entry:GetMessage() return notification.message end
    function entry:GetTypeName() return notification.typeName end
    return entry
end
-- The game's own queries (its action panel uses these two).
function NotificationManager.GetFirstEndTurnBlocking(player)
    for _, notification in ipairs(Civ6Stub.notifications[player] or {}) do
        if not notification.dismissed and (notification.blocking or 0) ~= 0 then
            return notification.blocking
        end
    end
    return EndTurnBlockingTypes.NO_ENDTURN_BLOCKING
end
function NotificationManager.FindEndTurnBlocking(blocking, player)
    for _, notification in ipairs(Civ6Stub.notifications[player] or {}) do
        if not notification.dismissed and notification.blocking == blocking then
            return NotificationManager.Find(player, notification.id)
        end
    end
    return nil
end
function NotificationManager.SendActivated(player, notificationId)
    record("NotificationManager.SendActivated", player, notificationId)
end
function NotificationManager.Dismiss(player, notificationId)
    record("NotificationManager.Dismiss", player, notificationId)
    local notification = findNotification(player, notificationId)
    if notification then notification.dismissed = true end
end

UI = {}
function UI.RequestAction(action) record("UI.RequestAction", action) end
Civ6Stub.pendingRequests = {}
function UI.RequestPlayerOperation(player, operation, params)
    local copy = shallowCopy(params)
    record("UI.RequestPlayerOperation", player, operation, copy)
    table.insert(Civ6Stub.pendingRequests, { player = player, operation = operation, params = copy })
end

-- The "next frame": apply the queued requests as the game is taken to.
-- Returns how many took effect.
local RESEARCH_REQUESTS = {
    [PlayerOperations.RESEARCH] = {
        rows = GameInfo.Technologies, param = PlayerOperations.PARAM_TECH_TYPE,
        state = "tech", blocker = EndTurnBlockingTypes.ENDTURN_BLOCKING_RESEARCH,
    },
    [PlayerOperations.PROGRESS_CIVIC] = {
        rows = GameInfo.Civics, param = PlayerOperations.PARAM_CIVIC_TYPE,
        state = "civic", blocker = EndTurnBlockingTypes.ENDTURN_BLOCKING_CIVIC,
    },
}
function Civ6Stub.applyRequests()
    local applied = 0
    for _, request in ipairs(Civ6Stub.pendingRequests) do
        local kind = RESEARCH_REQUESTS[request.operation]
        local state = kind and Civ6Stub.players[request.player] and Civ6Stub.players[request.player][kind.state]
        if state then
            for row in kind.rows() do
                if row.Hash == request.params[kind.param] and state.can[row.Index] and not state.has[row.Index] then
                    state.current = row.Index
                    applied = applied + 1
                    for _, notification in ipairs(Civ6Stub.notifications[request.player] or {}) do
                        if notification.blocking == kind.blocker then notification.dismissed = true end
                    end
                end
            end
        end
    end
    Civ6Stub.pendingRequests = {}
    return applied
end
function UI.CanEndTurn()
    if Civ6Stub.canEndTurn ~= nil then return Civ6Stub.canEndTurn end
    for _, notification in ipairs(Civ6Stub.notifications[Game.GetLocalPlayer()] or {}) do
        if not notification.dismissed and (notification.blocking or 0) ~= 0 then return false end
    end
    return true
end

local function event()
    local handlers = {}
    local e = {}
    function e.Add(handler) table.insert(handlers, handler) end
    function e.Remove(handler)
        for i, h in ipairs(handlers) do
            if h == handler then table.remove(handlers, i) return end
        end
    end
    function e.Count() return #handlers end
    -- Not the game's: lets a test play "the game published a frame".
    function e.Fire(...)
        local snapshot = {}
        for i, h in ipairs(handlers) do snapshot[i] = h end
        for _, h in ipairs(snapshot) do h(...) end
    end
    return e
end

Events = {
    GameCoreEventPublishComplete = event(),
    PlayerTurnActivated = event(),
    PlayerTurnDeactivated = event(),
    LoadGameViewStateDone = event(),
    LocalPlayerTurnBegin = event(),
    LocalPlayerTurnEnd = event(),
}

ExposedMembers = {}
