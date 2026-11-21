-- ============================================================================
-- civ6_stub_commands.lua - unit, city and combat requests for the stand-in game
-- ============================================================================
-- Test fixture only, loaded after civ6_stub.lua. Everything here is INVENTED
-- except the SHAPE of the calls, which follows the game's own UI scripts (read
-- in the public CQUI mod, commit 1e6c208; nothing is copied):
--
--   UnitManager.CanStartOperation(unit, hash, nil, true)          loose: could this
--                                                                  unit ever do it
--   UnitManager.CanStartOperation(unit, hash, nil, false, hint)   strict: now
--   UnitManager.CanStartOperation(unit, hash, nil, params)        strict, with a target
--   UnitManager.RequestOperation(unit, hash, params)
--   UnitManager.CanStartCommand / RequestCommand                  same, for commands
--   CityManager.RequestOperation(city, CityOperationTypes.BUILD, params)
--   CombatManager.IsAttackChangeWarState(componentId, x, y)       {} or { defender }
--
-- What the stand-in does NOT prove: that the game accepts these calls. It
-- proves that the mod makes them the way the game's UI does.
--
-- Rules that make a test mean something:
--   * UnitOperationTypes and UnitCommandTypes hold exactly the hashes of the
--     GameInfo.UnitOperations / GameInfo.UnitCommands rows. There is no MOVETO
--     and no WAIT, FORTIFY or SLEEP command.
--   * A request changes nothing when it returns. Civ6Stub.applyRequests() (or
--     Civ6Stub.tick(), which also publishes a frame) plays the game's next
--     frame. A request the game would not carry out is dropped without an
--     error and listed in Civ6Stub.dropped with the reason.
--   * The loose and the strict check differ: a unit with no moves left passes
--     only the loose one.
--   * A build is carried out only with the row's Hash under the parameter of
--     its own kind and the exclusive insert mode.
--   * An attack on a player the attacker is at peace with (Civ6Stub.atPeace)
--     is carried out and starts a war, listed in Civ6Stub.wars: a test asserts
--     that the list stays empty.
--   * With Civ6Stub.unitsNeedOrders = true, an own unit that has moves and no
--     orders blocks the end of turn (ENDTURN_BLOCKING_UNITS).
--   * Map.GetAdjacentPlot(x, y, DirectionTypes.*) is the only source of
--     neighbours; a city answers CanProduce only for a row Hash (GM-B7b).
-- ============================================================================

local function record(name, ...)
    table.insert(Civ6Stub.calls, { name = name, args = { ... } })
end

local function shallowCopy(source)
    local copy = {}
    for key, value in pairs(source or {}) do copy[key] = value end
    return copy
end

local function rows(list)
    return function()
        local i = 0
        return function()
            i = i + 1
            return list[i]
        end
    end
end

-- Invented hashes, far from every index and from each other.
local OP = {
    MOVE_TO = 91001, RANGE_ATTACK = 91002, FOUND_CITY = 91003,
    SKIP_TURN = 91004, FORTIFY = 91005, SLEEP = 91006,
}
local CMD = { DELETE = 92001, WAKE = 92002 }

Civ6Stub.unitOperationRows = {
    { Index = 0, Hash = OP.MOVE_TO, OperationType = "UNITOPERATION_MOVE_TO" },
    { Index = 1, Hash = OP.RANGE_ATTACK, OperationType = "UNITOPERATION_RANGE_ATTACK" },
    { Index = 2, Hash = OP.FOUND_CITY, OperationType = "UNITOPERATION_FOUND_CITY" },
    { Index = 3, Hash = OP.SKIP_TURN, OperationType = "UNITOPERATION_SKIP_TURN" },
    { Index = 4, Hash = OP.FORTIFY, OperationType = "UNITOPERATION_FORTIFY" },
    { Index = 5, Hash = OP.SLEEP, OperationType = "UNITOPERATION_SLEEP" },
}
Civ6Stub.unitCommandRows = {
    { Index = 0, Hash = CMD.DELETE, CommandType = "UNITCOMMAND_DELETE" },
    { Index = 1, Hash = CMD.WAKE, CommandType = "UNITCOMMAND_WAKE" },
}
GameInfo.UnitOperations = rows(Civ6Stub.unitOperationRows)
GameInfo.UnitCommands = rows(Civ6Stub.unitCommandRows)

UnitOperationTypes = {
    TYPE = 9001,
    PARAM_X = 9101, PARAM_Y = 9102, PARAM_MODIFIERS = 9103,
    MOVE_TO = OP.MOVE_TO, RANGE_ATTACK = OP.RANGE_ATTACK, FOUND_CITY = OP.FOUND_CITY,
    SKIP_TURN = OP.SKIP_TURN, FORTIFY = OP.FORTIFY, SLEEP = OP.SLEEP,
}
UnitCommandTypes = { TYPE = 9002, DELETE = CMD.DELETE, WAKE = CMD.WAKE }
UnitOperationMoveModifiers = { NONE = 0, ATTACK = 1, MOVE_IGNORE_UNEXPLORED_DESTINATION = 2 }
UnitOperationResults = { FAILURE_REASONS = 9201 }
UnitCommandResults = { FAILURE_REASONS = 9202 }
OperationResultsTypes = { NO_TARGETS = 9301 }

CityOperationTypes = {
    BUILD = 9401, PURCHASE = 9402,
    PARAM_UNIT_TYPE = 9411, PARAM_BUILDING_TYPE = 9412, PARAM_DISTRICT_TYPE = 9413,
    PARAM_INSERT_MODE = 9414, PARAM_X = 9415, PARAM_Y = 9416,
    VALUE_EXCLUSIVE = 9421, VALUE_APPEND = 9422,
}

-- ----------------------------------------------------------------------------
-- State
-- ----------------------------------------------------------------------------
Civ6Stub.unitRequests = {}   -- queued, not yet applied
Civ6Stub.dropped = {}        -- { what, reason } for every request not carried out
Civ6Stub.wars = {}           -- { attacker, defender } for every war an attack started
Civ6Stub.atPeace = {}        -- ["0:1"] = true: players 0 and 1 are at peace
Civ6Stub.unitsNeedOrders = false
Civ6Stub.canFoundCity = true
Civ6Stub.nextCityId = 302

local function peaceKey(a, b)
    if a > b then a, b = b, a end
    return a .. ":" .. b
end
function Civ6Stub.setPeace(a, b, atPeace)
    Civ6Stub.atPeace[peaceKey(a, b)] = atPeace and true or nil
end
local function atPeace(a, b)
    return a ~= b and Civ6Stub.atPeace[peaceKey(a, b)] == true
end

local function unitRecord(unitId)
    for index, spec in ipairs(Civ6Stub.units) do
        if spec.id == unitId then return spec, index end
    end
    return nil
end
function Civ6Stub.unit(unitId) return unitRecord(unitId) end

local function unitAt(x, y, exceptId)
    for _, spec in ipairs(Civ6Stub.units) do
        if spec.x == x and spec.y == y and spec.id ~= exceptId then return spec end
    end
    return nil
end

local function cityAt(x, y)
    for owner, list in pairs(Civ6Stub.cityRecords) do
        for _, spec in ipairs(list) do
            if spec.x == x and spec.y == y then return spec, owner end
        end
    end
    return nil
end
function Civ6Stub.city(cityId)
    for owner, list in pairs(Civ6Stub.cityRecords) do
        for _, spec in ipairs(list) do
            if spec.id == cityId then return spec, owner end
        end
    end
    return nil
end

-- Steps between two tiles on a grid whose odd rows sit half a tile to the
-- right. Invented geometry: it only has to be the same everywhere in the
-- stand-in.
local function cube(x, y)
    local q = x - (y - (y % 2)) / 2
    return q, y
end
function Map.GetPlotDistance(x1, y1, x2, y2)
    local q1, r1 = cube(x1, y1)
    local q2, r2 = cube(x2, y2)
    local dq, dr = q1 - q2, r1 - r2
    return math.max(math.abs(dq), math.abs(dr), math.abs(dq + dr))
end

-- The game's own adjacency: the plot one step from (x, y) in a direction, or
-- nil off the map. Same geometry as GetPlotDistance above (odd rows half a
-- tile to the right). The invented map does not wrap.
DirectionTypes = {
    DIRECTION_NORTHEAST = 0, DIRECTION_EAST = 1, DIRECTION_SOUTHEAST = 2,
    DIRECTION_SOUTHWEST = 3, DIRECTION_WEST = 4, DIRECTION_NORTHWEST = 5,
    NUM_DIRECTION_TYPES = 6,
}
local STEPS = {
    -- { dx on an even row, dx on an odd row, dy }
    [DirectionTypes.DIRECTION_NORTHEAST] = { 0, 1, 1 },
    [DirectionTypes.DIRECTION_EAST] = { 1, 1, 0 },
    [DirectionTypes.DIRECTION_SOUTHEAST] = { 0, 1, -1 },
    [DirectionTypes.DIRECTION_SOUTHWEST] = { -1, 0, -1 },
    [DirectionTypes.DIRECTION_WEST] = { -1, -1, 0 },
    [DirectionTypes.DIRECTION_NORTHWEST] = { -1, 0, 1 },
}
function Map.GetAdjacentPlot(x, y, direction)
    local step = STEPS[direction]
    if not step then return nil end
    local dx = (y % 2 == 0) and step[1] or step[2]
    return Map.GetPlot(x + dx, y + step[3])
end

local function hasBit(value, bit)
    return math.floor((value or 0) / bit) % 2 == 1
end

-- ----------------------------------------------------------------------------
-- What a unit can do
-- ----------------------------------------------------------------------------
local function canEver(spec, op)
    if op == OP.MOVE_TO or op == OP.SKIP_TURN or op == OP.SLEEP then return true end
    if op == OP.FORTIFY then return (spec.combat or 0) > 0 end
    if op == OP.RANGE_ATTACK then return (spec.ranged or 0) > 0 end
    if op == OP.FOUND_CITY then return spec.unitType == 0 end
    return false
end

-- true, or false and a reason.
local function canNow(spec, op, params)
    if not canEver(spec, op) then return false, "this unit can never do that" end
    if (spec.moves or 0) <= 0 then return false, "the unit has no moves left" end
    if op == OP.MOVE_TO or op == OP.RANGE_ATTACK then
        local x = params and params[UnitOperationTypes.PARAM_X]
        local y = params and params[UnitOperationTypes.PARAM_Y]
        if type(x) ~= "number" or type(y) ~= "number" then return false, "no target" end
        local plot = Map.GetPlot(x, y)
        if not plot then return false, "the target is off the map" end
        local occupant = unitAt(x, y, spec.id)
        if op == OP.RANGE_ATTACK then
            if not occupant or occupant.owner == spec.owner then return false, "nothing to attack there" end
            if (spec.attacks or 0) <= 0 then return false, "no attacks left" end
            if Map.GetPlotDistance(spec.x, spec.y, x, y) > (spec.range or 0) then
                return false, "the target is out of range"
            end
            return true
        end
        if x == spec.x and y == spec.y then return false, "the unit is already there" end
        if plot:IsImpassable() then return false, "the target is impassable" end
        if occupant then
            if occupant.owner == spec.owner then return false, "the tile is occupied" end
            local attacking = hasBit(params[UnitOperationTypes.PARAM_MODIFIERS], UnitOperationMoveModifiers.ATTACK)
            if not attacking then return false, "the tile is occupied" end
            if (spec.combat or 0) <= 0 then return false, "this unit cannot attack" end
            if (spec.attacks or 0) <= 0 then return false, "no attacks left" end
        end
        return true
    end
    if op == OP.FOUND_CITY then
        if Civ6Stub.canFoundCity == false then return false, "a city cannot be founded here" end
        if cityAt(spec.x, spec.y) then return false, "there is a city here already" end
        return true
    end
    if op == OP.FORTIFY and spec.activity == "fortified" then return false, "already fortified" end
    if op == OP.SLEEP and spec.activity == "sleeping" then return false, "already sleeping" end
    return true
end
Civ6Stub.canNow = canNow

local function answer(ok, reason, resultKey)
    if ok then return true, {} end
    return false, { [resultKey] = { reason } }
end

UnitManager = {}
function UnitManager.CanStartOperation(unit, op, _plot, fourth, _fifth)
    local spec = unit and unit._spec
    if not spec or not unitRecord(spec.id) then return false, {} end
    if fourth == true then
        return canEver(spec, op), {}
    end
    local params = type(fourth) == "table" and fourth or nil
    local ok, reason = canNow(spec, op, params)
    return answer(ok, reason, UnitOperationResults.FAILURE_REASONS)
end
function UnitManager.RequestOperation(unit, op, params)
    local spec = unit and unit._spec
    local copy = shallowCopy(params)
    record("UnitManager.RequestOperation", spec and spec.id, op, copy)
    table.insert(Civ6Stub.unitRequests, { kind = "operation", unitId = spec and spec.id, op = op, params = copy })
end

local function commandNow(spec, command)
    if command == CMD.DELETE then return true end
    if command == CMD.WAKE then
        if spec.activity == nil then return false, "the unit is not asleep or fortified" end
        return true
    end
    return false, "this unit can never do that"
end
function UnitManager.CanStartCommand(unit, command, loose, _wantResults)
    local spec = unit and unit._spec
    if not spec or not unitRecord(spec.id) then return false, {} end
    if loose == true then return command == CMD.DELETE or command == CMD.WAKE, {} end
    local ok, reason = commandNow(spec, command)
    return answer(ok, reason, UnitCommandResults.FAILURE_REASONS)
end
function UnitManager.RequestCommand(unit, command, params)
    local spec = unit and unit._spec
    record("UnitManager.RequestCommand", spec and spec.id, command, shallowCopy(params))
    table.insert(Civ6Stub.unitRequests, { kind = "command", unitId = spec and spec.id, op = command })
end

CombatManager = {}
-- The players a move or attack onto (x, y) would put the attacker at war with.
function CombatManager.IsAttackChangeWarState(componentId, x, y)
    local attacker = componentId and componentId.player
    local found = {}
    local occupant = unitAt(x, y, componentId and componentId.id)
    if occupant and atPeace(attacker, occupant.owner) then table.insert(found, occupant.owner) end
    local _, cityOwner = cityAt(x, y)
    if cityOwner and atPeace(attacker, cityOwner) and found[1] ~= cityOwner then
        table.insert(found, cityOwner)
    end
    return found
end

-- ----------------------------------------------------------------------------
-- Production
-- ----------------------------------------------------------------------------
local PRODUCIBLE = {
    { rows = GameInfo.Units, column = "UnitType", param = CityOperationTypes.PARAM_UNIT_TYPE },
    { rows = GameInfo.Buildings, column = "BuildingType", param = CityOperationTypes.PARAM_BUILDING_TYPE },
    { rows = GameInfo.Districts, column = "DistrictType", param = CityOperationTypes.PARAM_DISTRICT_TYPE,
      placed = true },
}
local function producible(hash)
    for _, kind in ipairs(PRODUCIBLE) do
        for row in kind.rows() do
            if row.Hash == hash then return row[kind.column], kind end
        end
    end
    return nil
end
-- civ6_stub.lua's CanProduce asks this to tell a row Hash from a small number.
function Civ6Stub.producibleName(ref)
    local name = producible(ref)
    return name
end

CityManager = {}
function CityManager.RequestOperation(city, operation, params)
    local spec = city and city._spec
    local copy = shallowCopy(params)
    record("CityManager.RequestOperation", spec and spec.id, operation, copy)
    table.insert(Civ6Stub.unitRequests, { kind = "city", cityId = spec and spec.id, op = operation, params = copy })
end

local function applyBuild(request)
    local spec, owner = Civ6Stub.city(request.cityId)
    if not spec then return false, "no such city" end
    if request.op ~= CityOperationTypes.BUILD then return false, "not a build operation" end
    local params = request.params
    if params[CityOperationTypes.PARAM_INSERT_MODE] ~= CityOperationTypes.VALUE_EXCLUSIVE then
        return false, "no exclusive insert mode"
    end
    local name, kind
    for _, candidate in ipairs(PRODUCIBLE) do
        local value = params[candidate.param]
        if value ~= nil then
            local found, foundKind = producible(value)
            if not found or foundKind ~= candidate then
                return false, "the type parameter is not the Hash of a row of its kind"
            end
            name, kind = found, candidate
        end
    end
    if not name then return false, "no type parameter" end
    if kind.placed and (params[CityOperationTypes.PARAM_X] == nil or params[CityOperationTypes.PARAM_Y] == nil) then
        return false, "a placed item needs a plot"
    end
    local allowed = (Civ6Stub.cities[spec.id] or {}).canProduce or Civ6Stub.defaultCanProduce
    if allowed[name] ~= "now" then return false, "the city cannot start that now" end
    spec.building = name
    for _, notification in ipairs(Civ6Stub.notifications[owner] or {}) do
        if notification.blocking == EndTurnBlockingTypes.ENDTURN_BLOCKING_PRODUCTION then
            notification.dismissed = true
        end
    end
    return true
end

-- ----------------------------------------------------------------------------
-- The next frame
-- ----------------------------------------------------------------------------
local function removeUnit(unitId)
    local _, index = unitRecord(unitId)
    if index then table.remove(Civ6Stub.units, index) end
end

local function applyOperation(request)
    local spec = unitRecord(request.unitId)
    if not spec then return false, "no such unit" end
    local op, params = request.op, request.params
    local ok, reason = canNow(spec, op, params)
    if not ok then return false, reason end
    if op == OP.MOVE_TO then
        local x, y = params[UnitOperationTypes.PARAM_X], params[UnitOperationTypes.PARAM_Y]
        local occupant = unitAt(x, y, spec.id)
        if occupant then
            if atPeace(spec.owner, occupant.owner) then
                table.insert(Civ6Stub.wars, { attacker = spec.owner, defender = occupant.owner })
                Civ6Stub.setPeace(spec.owner, occupant.owner, false)
            end
            occupant.damage = (occupant.damage or 0) + 30
            spec.attacks = (spec.attacks or 0) - 1
            spec.moves = 0
            if occupant.damage >= 100 then
                removeUnit(occupant.id)
                spec.x, spec.y = x, y
            end
        else
            spec.x, spec.y = x, y
            spec.moves = spec.moves - 1
        end
        spec.activity = nil
    elseif op == OP.RANGE_ATTACK then
        local x, y = params[UnitOperationTypes.PARAM_X], params[UnitOperationTypes.PARAM_Y]
        local occupant = unitAt(x, y, spec.id)
        if atPeace(spec.owner, occupant.owner) then
            table.insert(Civ6Stub.wars, { attacker = spec.owner, defender = occupant.owner })
            Civ6Stub.setPeace(spec.owner, occupant.owner, false)
        end
        occupant.damage = (occupant.damage or 0) + 25
        spec.attacks = spec.attacks - 1
        if occupant.damage >= 100 then removeUnit(occupant.id) end
    elseif op == OP.FOUND_CITY then
        local id = Civ6Stub.nextCityId
        Civ6Stub.nextCityId = id + 1
        Civ6Stub.cityRecords[spec.owner] = Civ6Stub.cityRecords[spec.owner] or {}
        table.insert(Civ6Stub.cityRecords[spec.owner],
            { id = id, name = "City " .. id, x = spec.x, y = spec.y, population = 1 })
        Civ6Stub.cities[id] = { buildings = {}, districts = { 0 } }
        removeUnit(spec.id)
    elseif op == OP.SKIP_TURN then
        spec.skipped = true
    elseif op == OP.FORTIFY then
        spec.activity = "fortified"
    elseif op == OP.SLEEP then
        spec.activity = "sleeping"
    end
    return true
end

local function applyCommand(request)
    local spec = unitRecord(request.unitId)
    if not spec then return false, "no such unit" end
    local ok, reason = commandNow(spec, request.op)
    if not ok then return false, reason end
    if request.op == CMD.DELETE then removeUnit(spec.id) else spec.activity = nil end
    return true
end

local applyPlayerRequests = Civ6Stub.applyRequests
function Civ6Stub.applyRequests()
    local applied = applyPlayerRequests()
    local queued = Civ6Stub.unitRequests
    Civ6Stub.unitRequests = {}
    for _, request in ipairs(queued) do
        local ok, reason
        if request.kind == "operation" then
            ok, reason = applyOperation(request)
        elseif request.kind == "command" then
            ok, reason = applyCommand(request)
        else
            ok, reason = applyBuild(request)
        end
        if ok then
            applied = applied + 1
        else
            table.insert(Civ6Stub.dropped, { what = request.kind, op = request.op, reason = reason })
        end
    end
    return applied
end

-- The game's next frame: carry out what was requested, then publish.
function Civ6Stub.tick()
    local applied = Civ6Stub.applyRequests()
    Events.GameCoreEventPublishComplete.Fire()
    return applied
end

-- ----------------------------------------------------------------------------
-- "A unit needs orders" blocks the end of turn
-- ----------------------------------------------------------------------------
local function unitNeedingOrders(player)
    if not Civ6Stub.unitsNeedOrders then return nil end
    for _, spec in ipairs(Civ6Stub.units) do
        if spec.owner == player and (spec.moves or 0) > 0 and not spec.skipped and spec.activity == nil then
            return spec
        end
    end
    return nil
end
Civ6Stub.unitNeedingOrders = unitNeedingOrders

local firstBlocking = NotificationManager.GetFirstEndTurnBlocking
function NotificationManager.GetFirstEndTurnBlocking(player)
    local blocking = firstBlocking(player)
    if blocking ~= EndTurnBlockingTypes.NO_ENDTURN_BLOCKING then return blocking end
    if unitNeedingOrders(player) then return EndTurnBlockingTypes.ENDTURN_BLOCKING_UNITS end
    return blocking
end
local findBlocking = NotificationManager.FindEndTurnBlocking
function NotificationManager.FindEndTurnBlocking(blocking, player)
    local entry = findBlocking(blocking, player)
    if entry then return entry end
    if blocking == EndTurnBlockingTypes.ENDTURN_BLOCKING_UNITS and unitNeedingOrders(player) then
        local needs = {}
        function needs:IsDismissed() return false end
        function needs:GetEndTurnBlocking() return blocking end
        function needs:GetMessage() return "A unit needs orders" end
        function needs:GetTypeName() return "NOTIFICATION_COMMAND_UNITS" end
        return needs
    end
    return nil
end
local canEndTurn = UI.CanEndTurn
function UI.CanEndTurn()
    if Civ6Stub.canEndTurn ~= nil then return Civ6Stub.canEndTurn end
    if unitNeedingOrders(Game.GetLocalPlayer()) then return false end
    return canEndTurn()
end

-- ----------------------------------------------------------------------------
-- Frames without game events (an idle game)
-- ----------------------------------------------------------------------------
-- The mod's context: ContextPtr:SetUpdate(f) asks for f on every frame, as the
-- game's own panels do. Civ6Stub.tick() plays a frame in which the game also
-- publishes a batch of events and does NOT call the frame update, so a test
-- says which of the two it means. Civ6Stub.idleFrames(n) plays n frames with
-- no event at all: requests are carried out, then only the update runs.
ContextPtr = ContextPtr or {}
function ContextPtr:SetUpdate(callback) Civ6Stub.frameUpdate = callback end
function ContextPtr:ClearUpdate() Civ6Stub.frameUpdate = nil end

function Civ6Stub.idleFrames(count)
    local applied = 0
    for _ = 1, count or 1 do
        applied = applied + Civ6Stub.applyRequests()
        if Civ6Stub.frameUpdate then Civ6Stub.frameUpdate(0.016) end
    end
    return applied
end
