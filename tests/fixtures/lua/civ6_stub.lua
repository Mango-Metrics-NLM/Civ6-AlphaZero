-- ============================================================================
-- civ6_stub.lua - a hand-written stand-in for the Civilization VI Lua globals
-- ============================================================================
-- Test fixture only. It lets the AlphaZeroAI mod's Lua run outside the game, so
-- tests can execute it instead of reading it as text.
--
-- Everything here is INVENTED: a 4x3 map, two players, three units, one city and
-- a handful of type rows. No game data. Names follow the game's naming style
-- (UNIT_WARRIOR, TECH_POTTERY) only so the mod's string checks behave as they do
-- in the game.
--
-- It models only what the mod calls today. A method the mod calls and this file
-- lacks does NOT raise in the exporter: every export component is wrapped in
-- pcall and falls back to an empty value. Tests must therefore assert that the
-- exported components are non-empty, or a hole here reads as a pass.
--
-- Civ6Stub.calls records every call that would change the game (UI requests),
-- so a test can assert what the mod asked the game to do.
-- ============================================================================

Civ6Stub = { calls = {} }

local function record(name, ...)
    table.insert(Civ6Stub.calls, { name = name, args = { ... } })
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
        { Index = 0, UnitType = "UNIT_SETTLER", Combat = 0, FormationClass = "FORMATION_CLASS_CIVILIAN" },
        { Index = 1, UnitType = "UNIT_WARRIOR", Combat = 20, FormationClass = "FORMATION_CLASS_LAND_COMBAT" },
        { Index = 2, UnitType = "UNIT_ARCHER", Combat = 15, FormationClass = "FORMATION_CLASS_LAND_COMBAT" },
    }),
    Buildings = rows({
        { Index = 0, BuildingType = "BUILDING_MONUMENT" },
        { Index = 1, BuildingType = "BUILDING_GRANARY" },
    }),
    Districts = rows({
        { Index = 0, DistrictType = "DISTRICT_CITY_CENTER" },
        { Index = 1, DistrictType = "DISTRICT_CAMPUS" },
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
    Civics = rows({
        { Index = 0, CivicType = "CIVIC_CODE_OF_LAWS" },
        { Index = 1, CivicType = "CIVIC_CRAFTSMANSHIP" },
    }),
    Technologies = rows({
        { Index = 0, TechnologyType = "TECH_POTTERY" },
        { Index = 1, TechnologyType = "TECH_MINING" },
        { Index = 2, TechnologyType = "TECH_WRITING" },
    }),
}

YieldTypes = {
    YIELD_FOOD = 0, YIELD_PRODUCTION = 1, YIELD_GOLD = 2,
    YIELD_SCIENCE = 3, YIELD_CULTURE = 4, YIELD_FAITH = 5,
}

ActionTypes = { ACTION_ENDTURN = 7 }

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
local function makeUnit(spec)
    local unit = {}
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
    return c
end

local allUnits = {
    makeUnit({ id = 101, owner = 0, x = 0, y = 0, unitType = 0, moves = 2 }),                           -- settler
    makeUnit({ id = 102, owner = 0, x = 1, y = 0, unitType = 1, moves = 2, combat = 20, attacks = 1 }), -- warrior
    makeUnit({ id = 201, owner = 1, x = 2, y = 0, unitType = 1, moves = 2, combat = 20, attacks = 1 }), -- enemy warrior
}

local function unitsOf(owner)
    local list = {}
    for _, unit in ipairs(allUnits) do
        if unit:GetOwner() == owner then table.insert(list, unit) end
    end
    return collection(list)
end

Units = {}
function Units.GetUnitsInPlot(plot)
    local list = {}
    for _, unit in ipairs(allUnits) do
        if unit:GetX() == plot:GetX() and unit:GetY() == plot:GetY() then
            table.insert(list, unit)
        end
    end
    return collection(list)
end

local function makeCity(spec)
    local city = {}
    local queue = {}
    function queue:CurrentlyBuilding() return spec.building end
    function queue:GetTurnsLeft() return 3 end
    -- Only units can be produced in this invented city.
    function queue:CanProduce(index, _) return index == 1 end
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
-- (index 0), is researching TECH_MINING (index 1) and has one city.
-- ----------------------------------------------------------------------------
local function makePlayer(id, cities)
    local player = {}
    function player:GetID() return id end
    function player:IsAlive() return true end
    function player:IsHuman() return id == 0 end
    function player:GetUnits() return unitsOf(id) end
    function player:GetCities() return collection(cities) end
    function player:GetScore() return 10 + id end
    function player:GetTreasury()
        local t = {}
        function t:GetGoldBalance() return 50 end
        function t:GetGoldYield() return 3 end
        return t
    end
    function player:GetCulture()
        local c = {}
        function c:GetCultureYield() return 2 end
        function c:GetProgressingCivic() return 1 end
        function c:GetResearchedCivics() return { 0 } end
        function c:HasCivic(index) return index == 0 end
        return c
    end
    function player:GetTechs()
        local t = {}
        function t:GetScienceYield() return 4 end
        function t:GetResearchingTech() return 1 end
        function t:HasTech(index) return index == 0 end
        function t:CanResearch(index) return index == 1 or index == 2 end
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
    [0] = makePlayer(0, { makeCity({ id = 301, name = "Stubville", x = 1, y = 1, population = 3 }) }),
    [1] = makePlayer(1, {}),
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
UI = {}
function UI.RequestAction(action) record("UI.RequestAction", action) end
function UI.RequestPlayerOperation(player, operation, params) record("UI.RequestPlayerOperation", player, operation) end

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
