-- Synthetic Lua sample with ACTION_TYPES but no ACTION_HANDLERS block.
-- The parser MUST raise LuaParseError rather than silently returning empty.

local ACTION_TYPES = {
    END_TURN = "end_turn",
    UNIT_MOVE = "unit_move",
}

-- No ACTION_HANDLERS = { ... } block on purpose.
