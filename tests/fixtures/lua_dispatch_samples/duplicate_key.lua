-- Synthetic Lua sample with a duplicate key in ACTION_HANDLERS.
-- The parser MUST raise LuaParseError; silently picking one would mask the bug.

local ACTION_TYPES = {
    END_TURN = "end_turn",
    UNIT_MOVE = "unit_move",
}

local function executeEndTurn(p, c) end
local function executeUnitMove(p, c) end

local ACTION_HANDLERS = {
    [ACTION_TYPES.END_TURN] = executeEndTurn,
    [ACTION_TYPES.UNIT_MOVE] = executeUnitMove,
    [ACTION_TYPES.UNIT_MOVE] = executeEndTurn,  -- duplicate, ambiguous dispatch
}
