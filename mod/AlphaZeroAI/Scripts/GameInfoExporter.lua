-- ============================================================================
-- GameInfoExporter.lua - a snapshot of the game's rule tables (GM-B4)
-- ============================================================================
-- Copies a few of the game's own GameInfo tables (technologies, civics,
-- buildings, units and their prerequisites) so the Python side can turn them
-- into a rule set: the real technology tree instead of a hand-written one.
-- The snapshot holds identifiers and whole numbers only, no text, and its
-- shape is exactly what `python -m alpha_zero.rules import-gameinfo` reads.
-- It is written once per game load (ENABLE_GAMEINFO_EXPORT), through FileIO.
-- ============================================================================

AlphaZeroGameInfo = AlphaZeroGameInfo or {}

local Log = nil  -- Will be set after Logger loads

-- The tables and columns of a snapshot. Adding a table or a column here means
-- teaching the Python importer about it first: it refuses a table it does not
-- know.
local SNAPSHOT_TABLES = {
    { name = "Technologies", columns = { "TechnologyType", "Cost", "EraType" } },
    { name = "TechnologyPrereqs", columns = { "Technology", "PrereqTech" } },
    { name = "Civics", columns = { "CivicType", "Cost", "EraType" } },
    { name = "CivicPrereqs", columns = { "Civic", "PrereqCivic" } },
    { name = "Buildings", columns = { "BuildingType", "Cost", "PrereqTech" } },
    { name = "Units", columns = { "UnitType", "Cost", "Combat", "BaseMoves", "PrereqTech" } },
}

-- Whole numbers a JSON reader gets back exactly.
local MAX_WHOLE_NUMBER = 2 ^ 53

-- A value the importer accepts (an identifier or a whole number), or nil.
local function sanitise(value)
    if type(value) == "number" then
        if value == math.floor(value) and value > -MAX_WHOLE_NUMBER and value < MAX_WHOLE_NUMBER then
            return value
        end
        return nil
    end
    if type(value) == "string" and string.match(value, "^[A-Za-z][A-Za-z0-9_]*$") then
        return value
    end
    return nil
end

-- The rows of one game table, or nil when the game has no such table or it
-- cannot be read whole. A column the game leaves empty (nil) is left out of
-- its row. A column that is present but is not an identifier or a whole
-- number, or a row with none of the columns, fails the whole table: dropping
-- the value or the row would hand the importer a prerequisite that is not
-- there or a move count of zero, and it could not tell.
local function exportTable(spec)
    local gameTable = GameInfo and GameInfo[spec.name]
    if not gameTable then
        return nil
    end

    local rows = {}
    local ok = pcall(function()
        for row in gameTable() do
            local copy, kept = {}, 0
            for _, column in ipairs(spec.columns) do
                local raw = row[column]
                if raw ~= nil then
                    local value = sanitise(raw)
                    if value == nil then
                        error(string.format("%s.%s holds a value that is not exported",
                            spec.name, column))
                    end
                    copy[column] = value
                    kept = kept + 1
                end
            end
            if kept == 0 then
                error(string.format("%s has a row with none of its columns", spec.name))
            end
            table.insert(rows, copy)
        end
    end)
    if not ok then
        return nil
    end
    return rows
end

-- ----------------------------------------------------------------------------
-- Public API
-- ----------------------------------------------------------------------------

function AlphaZeroGameInfo.Init()
    if AlphaZeroLogger then
        Log = AlphaZeroLogger.CreateLogger("GameInfoExporter")
    end
end

-- The snapshot document (table name -> rows), and the names of the tables
-- that could not be read.
function AlphaZeroGameInfo.Export()
    local document, missing = {}, {}
    for _, spec in ipairs(SNAPSHOT_TABLES) do
        local rows = exportTable(spec)
        if rows then
            document[spec.name] = rows
        else
            table.insert(missing, spec.name)
        end
    end
    return document, missing
end

-- Export the snapshot and hand it to FileIO. Returns true, or false and why.
function AlphaZeroGameInfo.ExportAndWrite()
    -- false: the previous load's snapshot could be neither marked nor removed.
    local begun = true
    if AlphaZeroFileIO and AlphaZeroFileIO.BeginGameInfo then
        begun = AlphaZeroFileIO.BeginGameInfo()
    end
    local document, missing = AlphaZeroGameInfo.Export()
    if #missing > 0 and Log then
        Log.Warn("GameInfo tables not exported: %s", table.concat(missing, ", "))
    end
    if next(document) == nil then
        if begun == false then
            return false, "no GameInfo table could be read, and the previous snapshot "
                .. "could not be removed: the file on disk is from an earlier game load"
        end
        return false, "no GameInfo table could be read"
    end
    if not AlphaZeroFileIO or not AlphaZeroFileIO.WriteGameInfo then
        return false, "FileIO not available"
    end

    local ok, err = AlphaZeroFileIO.WriteGameInfo(document)
    if ok and Log then
        Log.Info("GameInfo snapshot exported (%d of %d tables)",
            #SNAPSHOT_TABLES - #missing, #SNAPSHOT_TABLES)
    end
    return ok, err
end

-- Initialize on load
AlphaZeroGameInfo.Init()

print("[AlphaZero] GameInfoExporter module loaded")
