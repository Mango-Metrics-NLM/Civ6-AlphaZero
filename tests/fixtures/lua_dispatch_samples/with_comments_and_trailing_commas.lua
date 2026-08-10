-- Synthetic Lua sample exercising line comments + inline comments + trailing
-- commas in the ACTION_TYPES and ACTION_HANDLERS blocks.

local ACTION_TYPES = {
    END_TURN = "end_turn",     -- inline comment
    -- line comment between entries
    UNIT_MOVE = "unit_move",
    UNIT_ATTACK = "unit_attack",   -- trailing comma OK
}

local function executeEndTurn(p, c) end
local function executeUnitMove(p, c) end
local function executeNotImplemented(p, c) end

-- Dispatch table with one real and one stub handler.
local ACTION_HANDLERS = {
    [ACTION_TYPES.END_TURN] = executeEndTurn,
    [ACTION_TYPES.UNIT_MOVE] = executeUnitMove,  -- real
    [ACTION_TYPES.UNIT_ATTACK] = executeNotImplemented,  -- stub
}
