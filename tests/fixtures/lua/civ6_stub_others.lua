-- ============================================================================
-- civ6_stub_others.lua - the other players, what the local player sees, and
-- diplomacy, for the stand-in game (GM-B8a)
-- ============================================================================
-- Test fixture only, loaded after civ6_stub.lua and civ6_stub_commands.lua.
-- Models the calls the game's own UI scripts make (CQUI mirror), and nothing the
-- UI code does not call: there is NO PlayerManager.GetAliveIDs, and
-- PlayerManager.GetWasEverAliveCount stays at 2 (what it always answered), so a
-- mod that still enumerates players by it finds neither the city-state nor the
-- barbarians.
--
--   PlayersVisibility[p]:IsVisible(x, y) / :IsRevealed(x, y) / :IsUnitVisible(unit)
--   Game.GetPlayers{ Alive = true }          list of player objects
--   GameDefines.MAX_PLAYERS                  64
--   Players[i]:IsAlive() / IsMajor() / IsBarbarian() ; IsFreeCities only on request
--   Players[me]:GetDiplomacy():IsAtWarWith(id) / :HasMet(id)
--   Cities.GetCityInPlot(x, y)
--
-- DEFAULTS keep every earlier test meaning what it meant: every tile is visible
-- and revealed to every player, every unit is visible, and nobody has any
-- extra unit or city. Players: 0 (local), 1 (a second major, as before), 10 (a
-- city-state: IsMajor false) and 63 (barbarians: IsBarbarian true).
--
-- Diplomacy: war and peace are the combat stand-in's one record
-- (Civ6Stub.setPeace): a pair is at war unless set at peace, so IsAtWarWith and
-- CombatManager.IsAttackChangeWarState never disagree. HasMet answers false
-- for a pair with no record.
--
-- Test hooks (all on Civ6Stub):
--   hideTile(x, y [, p])           not visible, not revealed
--   fogTile(x, y [, p])            revealed, not visible now
--   showTile(x, y [, p])           the default again
--   hideUnit(unitId [, p])         a stealth unit: IsUnitVisible false
--   removeVisibility()             PlayersVisibility = nil
--   removeVisibilityEntry([p])     PlayersVisibility[p] = nil
--   removeVisibilityMethod(name [, p])   e.g. "IsUnitVisible"
--   addUnit{ id, owner, x, y, ... }      a unit record with the stand-in's defaults
--   addCity{ id, owner, x, y, ... }      a city record (name, population)
--   removeGetPlayers()             Game.GetPlayers = nil
--   breakGetPlayers()              Game.GetPlayers raises
--   removeMaxPlayers()             GameDefines = nil
--   kill(playerId)                 IsAlive false, not listed by GetPlayers
--   addFreeCities(playerId, answer)     the player gains IsFreeCities() -> answer
--   setWar(a, b, bool) / setMet(a, b, bool)
--   removeDiplomacy([p])           Players[p].GetDiplomacy = nil (the game cannot be asked)
-- ============================================================================

local LOCAL = Game.GetLocalPlayer()
local IDS = { 0, 1, 10, 63 }

-- ----------------------------------------------------------------------------
-- Players
-- ----------------------------------------------------------------------------
local function deepCopy(value)
    if type(value) ~= "table" then return value end
    local copy = {}
    for k, v in pairs(value) do copy[k] = deepCopy(v) end
    return copy
end

Civ6Stub.dead = {}
Civ6Stub.kinds = {
    [0] = { major = true }, [1] = { major = true }, [10] = { major = false },
    [63] = { barbarian = true },
}
Civ6Stub.cityRecords[10] = Civ6Stub.cityRecords[10] or {}
Civ6Stub.cityRecords[63] = Civ6Stub.cityRecords[63] or {}

for _, id in ipairs(IDS) do
    if not Players[id] then
        Players[id] = Civ6Stub.makePlayer(id)
        Civ6Stub.players[id] = deepCopy(Civ6Stub.players[1])
        local config = {}
        function config:GetCivilizationTypeName() return "CIVILIZATION_STUB_" .. id end
        PlayerConfigurations[id] = config
    end
    local player = Players[id]
    function player:IsAlive() return not Civ6Stub.dead[id] end
    function player:IsMajor() return Civ6Stub.kinds[id].major == true end
    function player:IsBarbarian() return Civ6Stub.kinds[id].barbarian == true end
end

GameDefines = { MAX_PLAYERS = 64 }

-- The game's list: player objects, in id order.
function Game.GetPlayers(args)
    local list = {}
    for id = 0, GameDefines.MAX_PLAYERS - 1 do
        local player = Players[id]
        if player and (not (args and args.Alive) or player:IsAlive()) then
            table.insert(list, player)
        end
    end
    return list
end

function Civ6Stub.removeGetPlayers() Game.GetPlayers = nil end
function Civ6Stub.breakGetPlayers()
    Game.GetPlayers = function() error("Game.GetPlayers is not available in this context") end
end
function Civ6Stub.removeMaxPlayers() GameDefines = nil end
function Civ6Stub.kill(playerId) Civ6Stub.dead[playerId] = true end
function Civ6Stub.addFreeCities(playerId, answer)
    Players[playerId].IsFreeCities = function() return answer end
end

-- ----------------------------------------------------------------------------
-- Units and cities of the other players
-- ----------------------------------------------------------------------------
function Civ6Stub.addUnit(spec)
    local record = { moves = 2, unitType = 1, combat = 20, attacks = 1 }
    for key, value in pairs(spec) do record[key] = value end
    table.insert(Civ6Stub.units, record)
    return record
end

function Civ6Stub.addCity(spec)
    local record = { name = "City " .. spec.id, population = 2 }
    for key, value in pairs(spec) do record[key] = value end
    local owner = record.owner
    record.owner = nil
    Civ6Stub.cityRecords[owner] = Civ6Stub.cityRecords[owner] or {}
    table.insert(Civ6Stub.cityRecords[owner], record)
    return record
end

-- The game's call: the city on a plot, with its owner.
Cities = {}
function Cities.GetCityInPlot(x, y)
    local spec, owner = nil, nil
    for who, list in pairs(Civ6Stub.cityRecords) do
        for _, record in ipairs(list) do
            if record.x == x and record.y == y then spec, owner = record, who end
        end
    end
    if not spec then return nil end
    local city = Civ6Stub.makeCity(spec)
    city.GetOwner = function() return owner end
    return city
end

-- ----------------------------------------------------------------------------
-- What the local player sees
-- ----------------------------------------------------------------------------
-- Per player: tiles by "x,y" -> "hidden" | "fogged"; unit ids that are stealth.
Civ6Stub.vis = {}
local function state(playerId)
    Civ6Stub.vis[playerId] = Civ6Stub.vis[playerId] or { tiles = {}, hiddenUnits = {} }
    return Civ6Stub.vis[playerId]
end

local function makeVisibility(playerId)
    local vis = {}
    local s = state(playerId)
    function vis:IsVisible(x, y) return s.tiles[x .. "," .. y] == nil end
    function vis:IsRevealed(x, y) return s.tiles[x .. "," .. y] ~= "hidden" end
    function vis:IsUnitVisible(unit)
        local id = unit and unit.GetID and unit:GetID()
        return not s.hiddenUnits[id]
    end
    return vis
end

PlayersVisibility = {}
for _, id in ipairs(IDS) do PlayersVisibility[id] = makeVisibility(id) end

function Civ6Stub.hideTile(x, y, p) state(p or LOCAL).tiles[x .. "," .. y] = "hidden" end
function Civ6Stub.fogTile(x, y, p) state(p or LOCAL).tiles[x .. "," .. y] = "fogged" end
function Civ6Stub.showTile(x, y, p) state(p or LOCAL).tiles[x .. "," .. y] = nil end
function Civ6Stub.hideUnit(unitId, p) state(p or LOCAL).hiddenUnits[unitId] = true end
function Civ6Stub.removeVisibility() PlayersVisibility = nil end
function Civ6Stub.removeVisibilityEntry(p) PlayersVisibility[p or LOCAL] = nil end
function Civ6Stub.removeVisibilityMethod(name, p) PlayersVisibility[p or LOCAL][name] = nil end

-- ----------------------------------------------------------------------------
-- Diplomacy of the local player
-- ----------------------------------------------------------------------------
Civ6Stub.war = {}
Civ6Stub.met = {}
local function key(a, b)
    if a > b then a, b = b, a end
    return a .. ":" .. b
end

-- ONE record for war and peace, the combat stand-in's (Civ6Stub.atPeace): a
-- pair is at war unless it was set at peace, for IsAtWarWith exactly as for
-- CombatManager.IsAttackChangeWarState. Two models would let the mod be
-- offered an attack on a unit of a player it is told it is at peace with.
function Civ6Stub.setWar(a, b, atWar) Civ6Stub.setPeace(a, b, not atWar) end
function Civ6Stub.setMet(a, b, met) Civ6Stub.met[key(a, b)] = met and true or nil end

for _, id in ipairs(IDS) do
    Players[id].GetDiplomacy = function(self)
        local d = {}
        function d:IsAtWarWith(other)
            return other ~= id and Civ6Stub.atPeace[key(id, other)] ~= true
        end
        function d:HasMet(other)
            return other ~= id and Civ6Stub.met[key(id, other)] == true
        end
        return d
    end
end
function Civ6Stub.removeDiplomacy(p) Players[p or LOCAL].GetDiplomacy = nil end
