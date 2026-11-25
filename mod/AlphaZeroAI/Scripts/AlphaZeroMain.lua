-- ============================================================================
-- AlphaZeroMain.lua - Bundled entry point for AlphaZero AI Bridge
-- ============================================================================
-- All modules concatenated into a single file for Civ6 AddUserInterfaces.
-- ============================================================================


-- ========== BEGIN Config.lua ==========
-- ============================================================================
-- Config.lua - Configuration management for AlphaZero AI Bridge
-- ============================================================================
-- Provides centralized configuration with sensible defaults and dynamic
-- overrides via JSON file-based configuration (safe, no code execution).
-- Fixes: cross-platform paths, JSON config loading instead of load().
-- ============================================================================

AlphaZeroConfig = AlphaZeroConfig or {}

-- Default path separator: backslash (Windows/Steam, the common case). This
-- is only a fallback -- AlphaZeroConfig.Init() calls detectPathSep() on the
-- resolved base path so the mod also works when Civ6 runs natively on
-- macOS/Linux (UI.GetAppSavegameDirectory returns the OS-correct separator).
-- The detected value is stored in currentConfig.PATH_SEP so other modules
-- (e.g. FileIO) reuse it via AlphaZeroConfig.Get("PATH_SEP").
local DEFAULT_PATH_SEP = "\\"

-- ----------------------------------------------------------------------------
-- Default Configuration Values
-- ----------------------------------------------------------------------------
-- These can be overridden by placing an alphazero_config.json file in the
-- communication directory.

local DEFAULT_CONFIG = {
    -- Communication settings
    COMM_DIR_NAME = "AlphaZeroAI",
    STATE_FILE = "state.json",
    COMMANDS_FILE = "commands.json",
    ACK_FILE = "ack.json",
    HEARTBEAT_FILE = "heartbeat.txt",
    ERROR_FILE = "errors.json",
    GAMEINFO_FILE = "gameinfo.json",

    -- Polling intervals (in seconds)
    COMMAND_POLL_INTERVAL = 0.01,  -- 10ms
    STATE_WRITE_DELAY = 0.001,     -- 1ms delay after write
    HEARTBEAT_INTERVAL = 1.0,      -- Write heartbeat every second

    -- Timeouts
    COMMAND_TIMEOUT = 30.0,        -- Wait 30s for command
    ACK_TIMEOUT = 5.0,             -- Wait 5s for ack read

    -- Feature flags
    ENABLE_LOGGING = true,
    LOG_LEVEL = "INFO",            -- DEBUG, INFO, WARN, ERROR
    ENABLE_STATE_COMPRESSION = false,
    ENABLE_DETAILED_UNITS = true,
    ENABLE_DETAILED_CITIES = true,
    ENABLE_LEGAL_ACTIONS = true,
    ENABLE_GAMEINFO_EXPORT = true,  -- Write the rule-table snapshot once per game load

    -- Player configuration
    AI_PLAYER_ID = nil,            -- Auto-detect first human player
    OBSERVE_ALL_PLAYERS = false,   -- Only observe AI player's visible tiles
    ALLOW_DECLARE_WAR = false,     -- The declare-war command is refused unless this is true
    -- Making peace is refused unless this is true; with it on, only while the
    -- game reports the pair at war.
    ALLOW_MAKE_PEACE = false,
    -- Denouncing is refused unless this is true; with it on, only a player the
    -- game reports as met and not at war.
    ALLOW_DENOUNCE = false,

    -- Map settings
    MAX_TILES_PER_STATE = 10000,   -- Limit tiles in state dump
    INCLUDE_FOG_OF_WAR = true,

    -- Safety settings
    MAX_ACTIONS_PER_TURN = 1000,   -- Prevent infinite action loops (per message)
    -- Commands handled in one game turn before only END_TURN is accepted. A
    -- loop that keeps sending a command the game ignores stops here.
    MAX_COMMANDS_PER_TURN = 200,
    -- Batches of game events between handling a command message and exporting
    -- the next state. A request is carried out by the game after the script
    -- returns; a state exported at once would still show the old game.
    STATE_EXPORT_DELAY_TICKS = 2,
    -- The same wait in frames, for a game that raises no event batch (it has
    -- gone idle): the state is exported after this many frames at the latest.
    STATE_EXPORT_MAX_FRAMES = 30,
    VALIDATE_COMMANDS = true,
    MAX_FILE_READ_BYTES = 10 * 1024 * 1024,  -- 10MB max file read

    -- Version
    VERSION = "1.1.0",
    PROTOCOL_VERSION = 2,
}

-- Current configuration (merged defaults + overrides)
local currentConfig = nil

-- ----------------------------------------------------------------------------
-- Helper Functions
-- ----------------------------------------------------------------------------

-- Deep copy a table
local function deepCopy(orig)
    local copy
    if type(orig) == 'table' then
        copy = {}
        for k, v in pairs(orig) do
            copy[k] = deepCopy(v)
        end
    else
        copy = orig
    end
    return copy
end

-- Merge two tables (b overrides a)
local function mergeTables(a, b)
    local result = deepCopy(a)
    for k, v in pairs(b) do
        if type(v) == 'table' and type(result[k]) == 'table' then
            result[k] = mergeTables(result[k], v)
        else
            result[k] = v
        end
    end
    return result
end

-- Get the base path for Civ6 documents.
-- Uses UI.GetAppSavegameDirectory() which is available in the
-- AddUserInterfaces context (not in AddGameplayScripts).
local function getDocumentsPath()
    -- Attempt 1: Civ6 save-game directory via UI API (available in-game).
    local ok, saveDir = pcall(function()
        return UI.GetAppSavegameDirectory()
    end)
    if ok and saveDir and saveDir ~= "" then
        print("[AlphaZero] Config: resolved save dir via UI API: " .. saveDir)
        return saveDir
    end

    -- Attempt 2: Environment variable (set before launching Civ6).
    if os.getenv then
        local envDir = os.getenv("ALPHAZERO_CIV6_IPC_DIR")
        if envDir and envDir ~= "" then
            print("[AlphaZero] Config: resolved via env var: " .. envDir)
            return envDir
        end
    end

    -- Fallback: relative path (for testing)
    print("[AlphaZero] Config: WARNING - using relative path fallback")
    return "."
end

-- Detect the path separator implied by a resolved base path. Returns "/"
-- when basePath looks like a Unix-style path (contains "/" but no "\\"),
-- otherwise falls back to DEFAULT_PATH_SEP (backslash).
local function detectPathSep(basePath)
    if type(basePath) == "string" and basePath:find("/", 1, true) and not basePath:find("\\", 1, true) then
        return "/"
    end
    return DEFAULT_PATH_SEP
end

-- ----------------------------------------------------------------------------
-- Public API
-- ----------------------------------------------------------------------------

-- Initialize configuration
function AlphaZeroConfig.Init()
    -- Start with defaults
    currentConfig = deepCopy(DEFAULT_CONFIG)

    -- Set derived paths using detected path separator
    local basePath = getDocumentsPath()
    local PATH_SEP = detectPathSep(basePath)
    currentConfig.PATH_SEP = PATH_SEP
    print("[AlphaZero] Config: base path = " .. tostring(basePath))
    currentConfig.COMM_DIR = basePath .. PATH_SEP .. currentConfig.COMM_DIR_NAME
    print("[AlphaZero] Config: COMM_DIR = " .. tostring(currentConfig.COMM_DIR))
    currentConfig.STATE_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.STATE_FILE
    currentConfig.COMMANDS_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.COMMANDS_FILE
    currentConfig.ACK_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.ACK_FILE
    currentConfig.HEARTBEAT_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.HEARTBEAT_FILE
    currentConfig.ERROR_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.ERROR_FILE
    currentConfig.GAMEINFO_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.GAMEINFO_FILE

    -- Try to load override configuration
    AlphaZeroConfig.LoadOverrides()

    return currentConfig
end

-- The files the bridge reads and writes: each path is its directory and name.
local DERIVED_PATHS = {
    { path = "STATE_PATH", file = "STATE_FILE" },
    { path = "COMMANDS_PATH", file = "COMMANDS_FILE" },
    { path = "ACK_PATH", file = "ACK_FILE" },
    { path = "HEARTBEAT_PATH", file = "HEARTBEAT_FILE" },
    { path = "ERROR_PATH", file = "ERROR_FILE" },
    { path = "GAMEINFO_PATH", file = "GAMEINFO_FILE" },
}

-- Merge overrides into the configuration. A path follows its directory and
-- its file name: overriding either moves the path, unless the path itself is
-- overridden, which wins.
function AlphaZeroConfig.ApplyOverrides(overrides)
    if not currentConfig then
        AlphaZeroConfig.Init()
    end
    if type(overrides) ~= "table" then
        return currentConfig
    end
    currentConfig = mergeTables(currentConfig, overrides)
    local PATH_SEP = currentConfig.PATH_SEP or DEFAULT_PATH_SEP
    for _, entry in ipairs(DERIVED_PATHS) do
        if overrides[entry.path] == nil
            and (overrides.COMM_DIR ~= nil or overrides[entry.file] ~= nil) then
            currentConfig[entry.path] = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig[entry.file]
        end
    end
    return currentConfig
end

-- Load configuration overrides from JSON file (safe, no code execution)
function AlphaZeroConfig.LoadOverrides()
    if not currentConfig then
        AlphaZeroConfig.Init()
        return
    end

    -- io may not be available at all script loading contexts in Civ6
    if not io or not io.open then
        print("[AlphaZero] Config: io not available, skipping override file load")
        return
    end

    local PATH_SEP = currentConfig.PATH_SEP or DEFAULT_PATH_SEP
    local configPath = currentConfig.COMM_DIR .. PATH_SEP .. "alphazero_config.json"
    local ok, file = pcall(io.open, configPath, "r")
    if not ok or not file then return end

    local content = file:read("*all")
    file:close()

    -- Use JSON decoder (safe, no code execution)
    if AlphaZeroJson and content and #content > 0 then
        local overrides, err = AlphaZeroJson.Decode(content)
        if overrides and type(overrides) == 'table' then
            AlphaZeroConfig.ApplyOverrides(overrides)
        elseif err then
            print("[AlphaZero] Config override parse error: " .. tostring(err))
        end
    end
end

-- Get a configuration value
function AlphaZeroConfig.Get(key)
    if not currentConfig then
        AlphaZeroConfig.Init()
    end
    return currentConfig[key]
end

-- Set a configuration value at runtime
function AlphaZeroConfig.Set(key, value)
    if not currentConfig then
        AlphaZeroConfig.Init()
    end
    currentConfig[key] = value
end

-- Get all configuration as table
function AlphaZeroConfig.GetAll()
    if not currentConfig then
        AlphaZeroConfig.Init()
    end
    return deepCopy(currentConfig)
end

-- Check if feature is enabled
function AlphaZeroConfig.IsEnabled(feature)
    local key = "ENABLE_" .. string.upper(feature)
    return AlphaZeroConfig.Get(key) == true
end

-- Get log level as number for comparison
function AlphaZeroConfig.GetLogLevelNum()
    local levels = { DEBUG = 1, INFO = 2, WARN = 3, ERROR = 4 }
    local level = AlphaZeroConfig.Get("LOG_LEVEL") or "INFO"
    return levels[level] or 2
end

-- Validate configuration
function AlphaZeroConfig.Validate()
    local errors = {}

    if not currentConfig then
        AlphaZeroConfig.Init()
    end

    if not currentConfig.COMM_DIR then
        table.insert(errors, "COMM_DIR not set")
    end

    if currentConfig.COMMAND_POLL_INTERVAL <= 0 then
        table.insert(errors, "COMMAND_POLL_INTERVAL must be positive")
    end

    if currentConfig.MAX_ACTIONS_PER_TURN <= 0 then
        table.insert(errors, "MAX_ACTIONS_PER_TURN must be positive")
    end

    return #errors == 0, errors
end

-- Get path separator (for reuse by other modules)
function AlphaZeroConfig.GetPathSep()
    if currentConfig and currentConfig.PATH_SEP then
        return currentConfig.PATH_SEP
    end
    return DEFAULT_PATH_SEP
end

-- Initialize on load
AlphaZeroConfig.Init()

print("[AlphaZero] Config module loaded - Version " .. AlphaZeroConfig.Get("VERSION"))

-- ========== END Config.lua ==========

-- ========== BEGIN Logger.lua ==========
-- ============================================================================
-- Logger.lua - Logging system for AlphaZero AI Bridge
-- ============================================================================
-- Provides consistent logging with levels, timestamps, and optional file output.
-- Fixes: cross-platform path separators, simple log rotation.
-- ============================================================================

AlphaZeroLogger = AlphaZeroLogger or {}

-- Log levels
local LOG_LEVELS = {
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
}

-- Current minimum log level
local minLogLevel = LOG_LEVELS.INFO

-- Log buffer for deferred writes
local logBuffer = {}
local MAX_BUFFER_SIZE = 100

-- Path separator: hardcoded for Civ6 Windows/Steam (os.getenv stripped from sandbox)
local PATH_SEP = "\\"

-- Log rotation settings
local MAX_LOG_SIZE = 5 * 1024 * 1024  -- 5MB
local rotationChecked = false

-- ----------------------------------------------------------------------------
-- Helper Functions
-- ----------------------------------------------------------------------------

-- Get current timestamp
local function getTimestamp()
    -- Civ6 doesn't have os.date, so we use game turn as pseudo-timestamp
    local turn = 0
    if Game then
        turn = Game.GetCurrentGameTurn() or 0
    end
    return string.format("T%04d", turn)
end

-- Format log message
local function formatMessage(level, module, message)
    local timestamp = getTimestamp()
    return string.format("[%s][%s][%s] %s", timestamp, level, module, message)
end

-- Write to Civ6 console
local function writeToConsole(formatted)
    print(formatted)
end

-- Check and perform log rotation if needed
local function checkLogRotation(logPath)
    if rotationChecked then return end
    rotationChecked = true  -- Only check once per session

    -- io/os may be stripped in Civ6 sandbox
    if not io or not io.open then return end

    local f = io.open(logPath, "rb")
    if f then
        local size = f:seek("end")
        f:close()
        if size and size > MAX_LOG_SIZE and os and os.remove then
            -- Rotate: remove old backup, rename current to .old
            pcall(os.remove, logPath .. ".old")
            pcall(os.rename, logPath, logPath .. ".old")
        end
    end
end

-- Write to log file (if enabled)
local function writeToFile(formatted)
    if not AlphaZeroConfig or not AlphaZeroConfig.IsEnabled("LOGGING") then
        return
    end

    -- io may be stripped in Civ6 sandbox; skip file logging
    if not io or not io.open then return end

    local logPath = AlphaZeroConfig.Get("COMM_DIR")
    if not logPath then return end

    logPath = logPath .. PATH_SEP .. "alphazero.log"

    -- Check rotation on first write
    checkLogRotation(logPath)

    local file = io.open(logPath, "a")
    if file then
        file:write(formatted .. "\n")
        file:close()
    end
end

-- ----------------------------------------------------------------------------
-- Public API
-- ----------------------------------------------------------------------------

-- Initialize logger
function AlphaZeroLogger.Init()
    if AlphaZeroConfig then
        local levelName = AlphaZeroConfig.Get("LOG_LEVEL") or "INFO"
        minLogLevel = LOG_LEVELS[levelName] or LOG_LEVELS.INFO
    end
    -- Reset rotation check for new session
    rotationChecked = false
end

-- Set minimum log level
function AlphaZeroLogger.SetLevel(level)
    if LOG_LEVELS[level] then
        minLogLevel = LOG_LEVELS[level]
    end
end

-- Generic log function
function AlphaZeroLogger.Log(level, module, message, ...)
    local levelNum = LOG_LEVELS[level] or LOG_LEVELS.INFO

    if levelNum < minLogLevel then
        return
    end

    -- Format message with varargs
    local formattedMsg = message
    if select("#", ...) > 0 then
        formattedMsg = string.format(message, ...)
    end

    local formatted = formatMessage(level, module, formattedMsg)

    -- Output
    writeToConsole(formatted)
    writeToFile(formatted)

    -- Buffer for batch operations
    table.insert(logBuffer, {
        level = level,
        module = module,
        message = formattedMsg,
        formatted = formatted,
    })

    -- Trim buffer
    while #logBuffer > MAX_BUFFER_SIZE do
        table.remove(logBuffer, 1)
    end
end

-- Convenience functions for each level
function AlphaZeroLogger.Debug(module, message, ...)
    AlphaZeroLogger.Log("DEBUG", module, message, ...)
end

function AlphaZeroLogger.Info(module, message, ...)
    AlphaZeroLogger.Log("INFO", module, message, ...)
end

function AlphaZeroLogger.Warn(module, message, ...)
    AlphaZeroLogger.Log("WARN", module, message, ...)
end

function AlphaZeroLogger.Error(module, message, ...)
    AlphaZeroLogger.Log("ERROR", module, message, ...)
end

-- Get recent log entries
function AlphaZeroLogger.GetRecentLogs(count)
    count = count or 10
    local result = {}
    local start = math.max(1, #logBuffer - count + 1)
    for i = start, #logBuffer do
        table.insert(result, logBuffer[i])
    end
    return result
end

-- Clear log buffer
function AlphaZeroLogger.ClearBuffer()
    logBuffer = {}
end

-- Flush logs to file
function AlphaZeroLogger.Flush()
    if not AlphaZeroConfig then return end

    local logPath = AlphaZeroConfig.Get("COMM_DIR")
    if not logPath then return end

    logPath = logPath .. PATH_SEP .. "alphazero.log"

    local file = io.open(logPath, "a")
    if file then
        for _, entry in ipairs(logBuffer) do
            file:write(entry.formatted .. "\n")
        end
        file:close()
    end
end

-- Create a scoped logger for a module
function AlphaZeroLogger.CreateLogger(moduleName)
    return {
        Debug = function(msg, ...) AlphaZeroLogger.Debug(moduleName, msg, ...) end,
        Info = function(msg, ...) AlphaZeroLogger.Info(moduleName, msg, ...) end,
        Warn = function(msg, ...) AlphaZeroLogger.Warn(moduleName, msg, ...) end,
        Error = function(msg, ...) AlphaZeroLogger.Error(moduleName, msg, ...) end,
    }
end

-- Initialize on load
AlphaZeroLogger.Init()

print("[AlphaZero] Logger module loaded")

-- ========== END Logger.lua ==========

-- ========== BEGIN JsonEncoder.lua ==========
-- ============================================================================
-- JsonEncoder.lua - Pure Lua JSON encoder/decoder for AlphaZero AI Bridge
-- ============================================================================
-- Provides JSON serialization without external dependencies since Civ6's
-- Lua runtime is sandboxed.
-- ============================================================================

AlphaZeroJson = AlphaZeroJson or {}

local Log = nil  -- Will be set after Logger loads

-- ----------------------------------------------------------------------------
-- Encoder Implementation
-- ----------------------------------------------------------------------------

local encodeValue  -- Forward declaration

-- Escape special characters in strings
local escapeChars = {
    ["\\"] = "\\\\",
    ["\""] = "\\\"",
    ["\n"] = "\\n",
    ["\r"] = "\\r",
    ["\t"] = "\\t",
    ["\b"] = "\\b",
    ["\f"] = "\\f",
}

-- One pass over the characters JSON does not allow raw in a string: the quote,
-- the backslash and the control characters. The short forms come from
-- escapeChars; any other control character is written as \u00XX.
--
-- The control characters are written out as bytes 0 to 31 and 127 (%z is the
-- zero byte, which a Lua 5.1 pattern cannot hold literally). The class %c is
-- not used: it asks the C library, and under the Windows C runtime that
-- answers "control" for bytes at and above 0x80, which broke every non-ASCII
-- character into \u00XX pieces.
--
-- This used to be two passes, the second with the pattern '[\x00-\x1f]'. Lua
-- 5.1 has no \x escape, so that was the class "x, 0 to x, 1, f": it rewrote
-- most letters, every digit and the underscore as \u00XX (six times the
-- length), and re-escaped the backslashes the first pass had just written, so
-- a newline came back as the two characters "\n" and a quote ended the string
-- early.
local function escapeString(str)
    str = string.gsub(str, '[%z\1-\31\127"\\]', function(c)
        return escapeChars[c] or string.format("\\u%04x", string.byte(c))
    end)
    return str
end

-- Encode a string value
local function encodeString(str)
    return '"' .. escapeString(str) .. '"'
end

-- Encode a number value
local function encodeNumber(num)
    -- Handle special cases
    if num ~= num then  -- NaN
        return "null"
    elseif num == math.huge then
        return "1e308"
    elseif num == -math.huge then
        return "-1e308"
    end

    -- Use integer format if whole number (%.0f avoids 32-bit C long truncation
    -- of %d on Windows LLP64 for safe IEEE-754 integers >= 2^31).
    if math.floor(num) == num and math.abs(num) < 2^53 then
        if num == 0 then
            return "0"
        end
        return string.format("%.0f", num)
    end

    return string.format("%.14g", num)
end

-- Check if table is an array
local function isArray(t)
    if type(t) ~= "table" then return false end

    local maxIndex = 0
    local count = 0

    for k, _ in pairs(t) do
        if type(k) ~= "number" then
            return false
        end
        if k <= 0 or math.floor(k) ~= k then
            return false
        end
        maxIndex = math.max(maxIndex, k)
        count = count + 1
    end

    return count == maxIndex
end

-- Encode an array
local function encodeArray(arr, indent, currentIndent)
    if #arr == 0 then
        return "[]"
    end

    local parts = {}
    local newIndent = currentIndent

    if indent then
        newIndent = currentIndent .. indent
    end

    for i, v in ipairs(arr) do
        local encoded = encodeValue(v, indent, newIndent)
        if indent then
            table.insert(parts, newIndent .. encoded)
        else
            table.insert(parts, encoded)
        end
    end

    if indent then
        return "[\n" .. table.concat(parts, ",\n") .. "\n" .. currentIndent .. "]"
    else
        return "[" .. table.concat(parts, ",") .. "]"
    end
end

-- Encode an object
local function encodeObject(obj, indent, currentIndent)
    local parts = {}
    local newIndent = currentIndent

    if indent then
        newIndent = currentIndent .. indent
    end

    -- Sort keys for consistent output
    local keys = {}
    for k in pairs(obj) do
        if type(k) == "string" or type(k) == "number" then
            table.insert(keys, k)
        end
    end
    table.sort(keys, function(a, b)
        return tostring(a) < tostring(b)
    end)

    for _, k in ipairs(keys) do
        local v = obj[k]
        local keyStr = encodeString(tostring(k))
        local valStr = encodeValue(v, indent, newIndent)

        if indent then
            table.insert(parts, newIndent .. keyStr .. ": " .. valStr)
        else
            table.insert(parts, keyStr .. ":" .. valStr)
        end
    end

    if #parts == 0 then
        return "{}"
    end

    if indent then
        return "{\n" .. table.concat(parts, ",\n") .. "\n" .. currentIndent .. "}"
    else
        return "{" .. table.concat(parts, ",") .. "}"
    end
end

-- Main encode function
encodeValue = function(value, indent, currentIndent)
    indent = indent or nil
    currentIndent = currentIndent or ""

    local valueType = type(value)

    if value == nil then
        return "null"
    elseif valueType == "boolean" then
        return value and "true" or "false"
    elseif valueType == "number" then
        return encodeNumber(value)
    elseif valueType == "string" then
        return encodeString(value)
    elseif valueType == "table" then
        if isArray(value) then
            return encodeArray(value, indent, currentIndent)
        else
            return encodeObject(value, indent, currentIndent)
        end
    elseif valueType == "function" or valueType == "userdata" or valueType == "thread" then
        return "null"  -- Can't serialize these
    else
        return "null"
    end
end

-- ----------------------------------------------------------------------------
-- Decoder Implementation
-- ----------------------------------------------------------------------------

local decodeValue  -- Forward declaration

-- Skip whitespace
local function skipWhitespace(str, pos)
    local pattern = "^[ \t\n\r]*"
    local _, endPos = string.find(str, pattern, pos)
    return endPos + 1
end

-- Decode a string
local function decodeString(str, pos)
    if string.sub(str, pos, pos) ~= '"' then
        return nil, pos, "Expected string"
    end

    pos = pos + 1
    local result = {}
    local startPos = pos

    while pos <= #str do
        local char = string.sub(str, pos, pos)

        if char == '"' then
            return table.concat(result), pos + 1, nil
        elseif char == '\\' then
            pos = pos + 1
            local escape = string.sub(str, pos, pos)

            if escape == '"' or escape == '\\' or escape == '/' then
                table.insert(result, escape)
            elseif escape == 'n' then
                table.insert(result, '\n')
            elseif escape == 'r' then
                table.insert(result, '\r')
            elseif escape == 't' then
                table.insert(result, '\t')
            elseif escape == 'b' then
                table.insert(result, '\b')
            elseif escape == 'f' then
                table.insert(result, '\f')
            elseif escape == 'u' then
                local hex = string.sub(str, pos + 1, pos + 4)
                local codepoint = tonumber(hex, 16)
                if codepoint then
                    -- Simple UTF-8 encoding for BMP characters
                    if codepoint < 0x80 then
                        table.insert(result, string.char(codepoint))
                    elseif codepoint < 0x800 then
                        table.insert(result, string.char(
                            0xC0 + math.floor(codepoint / 0x40),
                            0x80 + (codepoint % 0x40)
                        ))
                    else
                        table.insert(result, string.char(
                            0xE0 + math.floor(codepoint / 0x1000),
                            0x80 + math.floor((codepoint % 0x1000) / 0x40),
                            0x80 + (codepoint % 0x40)
                        ))
                    end
                    pos = pos + 4
                else
                    return nil, pos, "Invalid unicode escape"
                end
            else
                return nil, pos, "Invalid escape sequence"
            end
            pos = pos + 1
        else
            table.insert(result, char)
            pos = pos + 1
        end
    end

    return nil, pos, "Unterminated string"
end

-- Decode a number
local function decodeNumber(str, pos)
    local pattern = "^-?%d+%.?%d*[eE]?[+-]?%d*"
    local numStr = string.match(str, pattern, pos)

    if not numStr or #numStr == 0 then
        return nil, pos, "Expected number"
    end

    local num = tonumber(numStr)
    if not num then
        return nil, pos, "Invalid number"
    end

    return num, pos + #numStr, nil
end

-- Decode an array
local function decodeArray(str, pos)
    if string.sub(str, pos, pos) ~= '[' then
        return nil, pos, "Expected array"
    end

    pos = skipWhitespace(str, pos + 1)
    local result = {}

    if string.sub(str, pos, pos) == ']' then
        return result, pos + 1, nil
    end

    while true do
        local value, newPos, err = decodeValue(str, pos)
        if err then
            return nil, newPos, err
        end

        table.insert(result, value)
        pos = skipWhitespace(str, newPos)

        local char = string.sub(str, pos, pos)
        if char == ']' then
            return result, pos + 1, nil
        elseif char == ',' then
            pos = skipWhitespace(str, pos + 1)
        else
            return nil, pos, "Expected ',' or ']'"
        end
    end
end

-- Decode an object
local function decodeObject(str, pos)
    if string.sub(str, pos, pos) ~= '{' then
        return nil, pos, "Expected object"
    end

    pos = skipWhitespace(str, pos + 1)
    local result = {}

    if string.sub(str, pos, pos) == '}' then
        return result, pos + 1, nil
    end

    while true do
        -- Key
        local key, newPos, err = decodeString(str, pos)
        if err then
            return nil, newPos, err
        end

        pos = skipWhitespace(str, newPos)

        if string.sub(str, pos, pos) ~= ':' then
            return nil, pos, "Expected ':'"
        end

        pos = skipWhitespace(str, pos + 1)

        -- Value
        local value
        value, newPos, err = decodeValue(str, pos)
        if err then
            return nil, newPos, err
        end

        result[key] = value
        pos = skipWhitespace(str, newPos)

        local char = string.sub(str, pos, pos)
        if char == '}' then
            return result, pos + 1, nil
        elseif char == ',' then
            pos = skipWhitespace(str, pos + 1)
        else
            return nil, pos, "Expected ',' or '}'"
        end
    end
end

-- Main decode function
decodeValue = function(str, pos)
    pos = skipWhitespace(str, pos)

    local char = string.sub(str, pos, pos)

    if char == '"' then
        return decodeString(str, pos)
    elseif char == '[' then
        return decodeArray(str, pos)
    elseif char == '{' then
        return decodeObject(str, pos)
    elseif char == 't' then
        if string.sub(str, pos, pos + 3) == "true" then
            return true, pos + 4, nil
        end
        return nil, pos, "Invalid literal"
    elseif char == 'f' then
        if string.sub(str, pos, pos + 4) == "false" then
            return false, pos + 5, nil
        end
        return nil, pos, "Invalid literal"
    elseif char == 'n' then
        if string.sub(str, pos, pos + 3) == "null" then
            return nil, pos + 4, nil
        end
        return nil, pos, "Invalid literal"
    elseif char == '-' or (char >= '0' and char <= '9') then
        return decodeNumber(str, pos)
    else
        return nil, pos, "Unexpected character: " .. char
    end
end

-- ----------------------------------------------------------------------------
-- Public API
-- ----------------------------------------------------------------------------

-- Encode Lua value to JSON string
function AlphaZeroJson.Encode(value, pretty)
    local indent = pretty and "  " or nil
    local ok, result = pcall(encodeValue, value, indent, "")
    if not ok then
        if Log then Log.Error("JSON encode failed: " .. tostring(result)) end
        return nil, result
    end
    return result, nil
end

-- Decode JSON string to Lua value
function AlphaZeroJson.Decode(str)
    if type(str) ~= "string" then
        return nil, "Input must be a string"
    end

    if #str == 0 then
        return nil, "Empty string"
    end

    local ok, value, pos, err = pcall(decodeValue, str, 1)
    if not ok then
        if Log then Log.Error("JSON decode failed: " .. tostring(value)) end
        return nil, value
    end

    if err then
        return nil, string.format("JSON parse error at position %d: %s", pos, err)
    end

    return value, nil
end

-- Validate JSON string
function AlphaZeroJson.Validate(str)
    local _, err = AlphaZeroJson.Decode(str)
    return err == nil, err
end

-- Initialize logger reference
function AlphaZeroJson.Init()
    if AlphaZeroLogger then
        Log = AlphaZeroLogger.CreateLogger("JSON")
    end
end

-- Try to initialize
if AlphaZeroLogger then
    AlphaZeroJson.Init()
end

print("[AlphaZero] JsonEncoder module loaded")

-- ========== END JsonEncoder.lua ==========

-- ========== BEGIN FileIO.lua ==========
-- ============================================================================
-- FileIO.lua - File I/O operations for AlphaZero AI Bridge
-- ============================================================================
-- Handles reading and writing JSON files for communication with Python.
-- Fixes: cross-platform mkdir, path traversal prevention, file size limits,
--        content-hash-based change detection.
-- ============================================================================

AlphaZeroFileIO = AlphaZeroFileIO or {}

local Log = nil  -- Will be set after Logger loads

-- Default path separator (fallback only). AlphaZeroConfig detects the real
-- separator from the resolved base path in Init() and stores it as
-- currentConfig.PATH_SEP; getPathSep() below reuses that value so FileIO
-- stays consistent with Config across platforms.
local DEFAULT_PATH_SEP = "\\"

-- Get the path separator to use, preferring the one detected by Config.
local function getPathSep()
    return (AlphaZeroConfig and AlphaZeroConfig.Get("PATH_SEP")) or DEFAULT_PATH_SEP
end

-- Track file content hashes for change detection
local fileContentHashes = {}

-- Configurable max read size (default 10MB)
local MAX_READ_BYTES = 10 * 1024 * 1024

-- ----------------------------------------------------------------------------
-- Helper Functions
-- ----------------------------------------------------------------------------

-- Simple content hash for change detection
local function simpleHash(content)
    local hash = 0
    for i = 1, #content do
        hash = (hash * 31 + string.byte(content, i)) % 2147483647
    end
    return hash
end

-- Validate path for security (prevent traversal and injection)
local function isPathSafe(path)
    if not path or type(path) ~= "string" then
        return false, "Invalid path"
    end
    -- Block path traversal
    if path:match('%.%.') then
        return false, "Path traversal blocked"
    end
    -- Allow alphanumeric, underscore, hyphen, dot, spaces, colons (drive letters),
    -- and path separators
    if path:match('[^%w%._/\\%- :]') then
        return false, "Invalid characters in path"
    end
    return true, nil
end

-- Escape a path string for safe use in shell commands
local function shellEscape(s)
    -- Replace double quotes with escaped double quotes
    return s:gsub('"', '\\"')
end

-- Ensure directory exists (cross-platform)
local function ensureDirectory(path)
    -- os.execute may be stripped in Civ6 sandbox; skip if unavailable
    if not os or not os.execute then
        return
    end

    local safe, err = isPathSafe(path)
    if not safe then
        if Log then Log.Error("Security warning: %s: %s", err, path) end
        return
    end

    local safePath = shellEscape(path)
    local ok = pcall(function()
        if getPathSep() == "/" then
            os.execute('mkdir -p "' .. safePath .. '" 2>/dev/null')
        else
            os.execute('mkdir "' .. safePath .. '" 2>nul')
        end
    end)
    if not ok then
        print("[AlphaZero] FileIO: could not create directory (os.execute sandboxed)")
    end
end

-- Check if file exists
local function fileExists(path)
    if not io or not io.open then return false end
    local file = io.open(path, "r")
    if file then
        file:close()
        return true
    end
    return false
end

-- ----------------------------------------------------------------------------
-- Public API
-- ----------------------------------------------------------------------------

-- Initialize FileIO module
function AlphaZeroFileIO.Init()
    if AlphaZeroLogger then
        Log = AlphaZeroLogger.CreateLogger("FileIO")
    end

    -- Diagnostic: check io availability
    if io and io.open then
        print("[AlphaZero] FileIO: io.open is AVAILABLE")
    else
        print("[AlphaZero] FileIO: WARNING - io.open is NOT available, file I/O will fail")
    end

    -- Load configurable max read size
    if AlphaZeroConfig then
        MAX_READ_BYTES = AlphaZeroConfig.Get("MAX_FILE_READ_BYTES") or MAX_READ_BYTES
    end

    -- Ensure communication directory exists
    local commDir = AlphaZeroConfig and AlphaZeroConfig.Get("COMM_DIR")
    if commDir then
        ensureDirectory(commDir)
        print("[AlphaZero] FileIO: Comm directory: " .. commDir)
    end
end

-- Read a file and return contents (with size limit)
function AlphaZeroFileIO.Read(path)
    if not io or not io.open then
        return nil, "io not available in this Lua context"
    end
    local file, err = io.open(path, "r")
    if not file then
        if Log then Log.Debug("Could not open file for reading: %s", path) end
        return nil, "Could not open file: " .. tostring(err)
    end

    local content = file:read(MAX_READ_BYTES)
    local remaining = file:read(1)
    file:close()

    if remaining then
        if Log then Log.Warn("File %s exceeds max read size (%d bytes), truncated", path, MAX_READ_BYTES) end
    end

    if content and Log then
        Log.Debug("Read %d bytes from %s", #content, path)
    end

    return content, nil
end

-- Write content to a file
function AlphaZeroFileIO.Write(path, content)
    if not io or not io.open then
        return false, "io not available in this Lua context"
    end
    local file, err = io.open(path, "w")
    if not file then
        if Log then Log.Error("Could not open file for writing: %s - %s", path, tostring(err)) end
        return false, "Could not open file: " .. tostring(err)
    end

    local ok, writeErr = file:write(content)
    file:close()

    if not ok then
        if Log then Log.Error("Write failed: %s", tostring(writeErr)) end
        return false, "Write failed: " .. tostring(writeErr)
    end

    if Log then Log.Debug("Wrote %d bytes to %s", #content, path) end
    return true, nil
end

-- Append content to a file
function AlphaZeroFileIO.Append(path, content)
    if not io or not io.open then
        return false, "io not available in this Lua context"
    end
    local file, err = io.open(path, "a")
    if not file then
        if Log then Log.Error("Could not open file for appending: %s", path) end
        return false, "Could not open file: " .. tostring(err)
    end

    local ok, writeErr = file:write(content)
    file:close()

    if not ok then
        return false, "Write failed: " .. tostring(writeErr)
    end

    return true, nil
end

-- Delete a file
-- True when the file is gone afterwards (or was never there); false and why
-- when it is still there. The removal is attempted first and the file looked
-- for afterwards: os.remove reports a failure by returning nil, which is not
-- an error pcall sees, and a file that cannot be read can often still be
-- removed. Limit: a file that can be neither read nor removed looks absent.
function AlphaZeroFileIO.Delete(path)
    local reason = "os.remove not available"
    if os and os.remove then
        local ok, removed, err = pcall(os.remove, path)
        reason = (not ok and removed) or err or "os.remove failed"
    end
    if fileExists(path) then
        if Log then Log.Warn("Could not delete file: %s", path) end
        return false, tostring(reason)
    end
    return true, nil
end

-- Check if file exists
function AlphaZeroFileIO.Exists(path)
    return fileExists(path)
end

-- Check if file has changed since last read (content-hash based)
function AlphaZeroFileIO.HasChanged(path)
    local content, err = AlphaZeroFileIO.Read(path)
    if not content then
        return false
    end

    local hash = simpleHash(content)
    local lastHash = fileContentHashes[path]

    if hash ~= lastHash then
        fileContentHashes[path] = hash
        return true
    end
    return false
end

-- Reset change tracking for a file
function AlphaZeroFileIO.ResetChangeTracking(path)
    fileContentHashes[path] = nil
end

-- Read JSON file and parse
function AlphaZeroFileIO.ReadJson(path)
    local content, readErr = AlphaZeroFileIO.Read(path)
    if not content then
        return nil, readErr
    end

    if not AlphaZeroJson then
        return nil, "JSON encoder not available"
    end

    local data, parseErr = AlphaZeroJson.Decode(content)
    if parseErr then
        if Log then Log.Error("JSON parse error in %s: %s", path, parseErr) end
        return nil, parseErr
    end

    return data, nil
end

-- Write data as JSON to file
function AlphaZeroFileIO.WriteJson(path, data, pretty)
    if not AlphaZeroJson then
        return false, "JSON encoder not available"
    end

    local json, encodeErr = AlphaZeroJson.Encode(data, pretty)
    if not json then
        if Log then Log.Error("JSON encode error: %s", encodeErr) end
        return false, encodeErr
    end

    return AlphaZeroFileIO.Write(path, json)
end

-- Write state to state file
-- ============================================================================
-- Print-based IPC fallback (used when io.open is nil in Civ6 sandbox)
-- Protocol: Lua prints "AZ_STATE:{json}" etc. to Lua.log
--           Python tails Lua.log for these prefixed lines
--           Python sets ExposedMembers.AlphaZeroCommand for commands
-- ============================================================================

local HAS_IO = (io ~= nil and io.open ~= nil)

-- Safe timestamp helper (os.time may be stripped)
local function safeTimestamp()
    return (os and os.time) and os.time() or 0
end

-- Get current game turn
local function currentTurn()
    return (Game and Game.GetCurrentGameTurn) and Game.GetCurrentGameTurn() or 0
end

function AlphaZeroFileIO.WriteState(state)
    if not AlphaZeroJson then
        return false, "JSON encoder not available"
    end

    local json, err = AlphaZeroJson.Encode(state, false)
    if not json then
        return false, err
    end

    if HAS_IO then
        -- File-based IPC
        local path = AlphaZeroConfig and AlphaZeroConfig.Get("STATE_PATH")
        if not path then return false, "STATE_PATH not configured" end
        return AlphaZeroFileIO.Write(path, json)
    else
        -- Print-based IPC fallback with chunking for large states
        -- Lua.log truncates lines at ~4KB; split into 3500-char chunks
        local MAX_CHUNK = 1800
        local jsonLen = #json
        if jsonLen <= MAX_CHUNK then
            -- Small enough for a single line
            print("AZ_STATE:" .. json)
        else
            -- Chunked output: AZ_STATE_CHUNK:seq/total:data
            local totalChunks = math.ceil(jsonLen / MAX_CHUNK)
            for i = 1, totalChunks do
                local startPos = (i - 1) * MAX_CHUNK + 1
                local endPos = math.min(i * MAX_CHUNK, jsonLen)
                local chunk = json:sub(startPos, endPos)
                print("AZ_STATE_CHUNK:" .. i .. "/" .. totalChunks .. ":" .. chunk)
            end
            print("AZ_STATE_END:" .. totalChunks)
        end
        return true, nil
    end
end

-- Adler-32 of a string, as zlib computes it. The snapshot's END line carries
-- it, so the reader can tell a whole snapshot from pieces of two.
local function adler32(text)
    local a, b = 1, 0
    for i = 1, #text do
        a = (a + string.byte(text, i)) % 65521
        b = (b + a) % 65521
    end
    return b * 65536 + a
end

-- Announce a snapshot attempt (GM-B4), before any table is read: an attempt
-- that then exports nothing must not leave the previous game load's snapshot
-- looking current. Removes the previous file, or prints a line the reader
-- takes as "a newer export started".
function AlphaZeroFileIO.BeginGameInfo()
    if ExposedMembers then
        ExposedMembers.AlphaZeroGameInfo = nil
    end
    if HAS_IO then
        -- Overwritten, not deleted: a delete can fail without a word, and the
        -- previous load's snapshot would then stand as this load's. The
        -- capture refuses a file that holds only this marker.
        local path = AlphaZeroConfig and AlphaZeroConfig.Get("GAMEINFO_PATH")
        if not path then return false end
        local ok, err = AlphaZeroFileIO.Write(path, "AZ_GAMEINFO_BEGIN")
        if ok then return true end
        -- A read-only file cannot take the marker but can often be removed.
        local deleted, deleteErr = AlphaZeroFileIO.Delete(path)
        if deleted then return true end
        if Log then
            Log.Error("The previous GameInfo snapshot could not be marked (%s) or removed (%s)",
                tostring(err), tostring(deleteErr))
        end
        return false
    end
    print("AZ_GAMEINFO_BEGIN")
    return true
end

-- Write the rule-table snapshot (GM-B4): a file when file access exists,
-- otherwise printed like a state, under its own prefixes so the state reader
-- does not take it for one.
function AlphaZeroFileIO.WriteGameInfo(document)
    if not AlphaZeroJson then
        return false, "JSON encoder not available"
    end

    local json, err = AlphaZeroJson.Encode(document, false)
    if not json then
        return false, err
    end

    if HAS_IO then
        local path = AlphaZeroConfig and AlphaZeroConfig.Get("GAMEINFO_PATH")
        if not path then return false, "GAMEINFO_PATH not configured" end
        return AlphaZeroFileIO.Write(path, json)
    end

    -- Lua.log truncates long lines, so a long snapshot is cut into pieces:
    -- AZ_GAMEINFO_CHUNK:seq/total:data ... AZ_GAMEINFO_END:total:checksum
    -- The checksum is printed with %.0f: it can exceed 2^31, which %d does
    -- not print on every build of Lua 5.1.
    local MAX_CHUNK = 1800
    local jsonLen = #json
    if jsonLen <= MAX_CHUNK then
        print("AZ_GAMEINFO:" .. json)
    else
        local totalChunks = math.ceil(jsonLen / MAX_CHUNK)
        for i = 1, totalChunks do
            local startPos = (i - 1) * MAX_CHUNK + 1
            local endPos = math.min(i * MAX_CHUNK, jsonLen)
            print("AZ_GAMEINFO_CHUNK:" .. i .. "/" .. totalChunks .. ":" .. json:sub(startPos, endPos))
        end
        print("AZ_GAMEINFO_END:" .. totalChunks .. ":" .. string.format("%.0f", adler32(json)))
    end
    if ExposedMembers then
        ExposedMembers.AlphaZeroGameInfo = json
    end
    return true, nil
end

-- Read commands from either file or ExposedMembers
function AlphaZeroFileIO.ReadCommands()
    if HAS_IO then
        -- File-based IPC
        local path = AlphaZeroConfig and AlphaZeroConfig.Get("COMMANDS_PATH")
        if not path then return nil, "COMMANDS_PATH not configured" end
        if not AlphaZeroFileIO.HasChanged(path) then return nil, nil end
        return AlphaZeroFileIO.ReadJson(path)
    else
        -- ExposedMembers-based IPC fallback
        if ExposedMembers and ExposedMembers.AlphaZeroCommand then
            local cmdJson = ExposedMembers.AlphaZeroCommand
            ExposedMembers.AlphaZeroCommand = nil  -- Consume command

            if AlphaZeroJson and AlphaZeroJson.Decode then
                local data, err = AlphaZeroJson.Decode(cmdJson)
                if data then
                    print("[AlphaZero] Command received via ExposedMembers")
                    return data, nil
                else
                    return nil, "JSON decode error: " .. tostring(err)
                end
            end
        end
        return nil, nil  -- No commands available
    end
end

-- Whether a command is waiting to be read, without reading it: the file
-- changed since it was last read, or ExposedMembers holds one.
function AlphaZeroFileIO.HasPendingCommand()
    if HAS_IO then
        local path = AlphaZeroConfig and AlphaZeroConfig.Get("COMMANDS_PATH")
        if not path then return false end
        local content = AlphaZeroFileIO.Read(path)
        return content ~= nil and simpleHash(content) ~= fileContentHashes[path]
    end
    return ExposedMembers ~= nil and ExposedMembers.AlphaZeroCommand ~= nil
end

-- Write acknowledgment
function AlphaZeroFileIO.WriteAck(msgId, success, errorCode, errorMessage, executed)
    local ack = {
        msg_id = msgId,
        timestamp = safeTimestamp(),
        success = success,
        error_code = errorCode,
        error_message = errorMessage,
        executed = executed or {},
    }

    if not AlphaZeroJson then return false, "JSON encoder not available" end
    local json, err = AlphaZeroJson.Encode(ack, false)
    if not json then return false, err end

    if HAS_IO then
        local path = AlphaZeroConfig and AlphaZeroConfig.Get("ACK_PATH")
        if not path then return false, "ACK_PATH not configured" end
        return AlphaZeroFileIO.Write(path, json)
    else
        -- Print-based fallback + ExposedMembers
        print("AZ_ACK:" .. json)
        if ExposedMembers then
            ExposedMembers.AlphaZeroAck = json
        end
        return true, nil
    end
end

-- Write heartbeat
function AlphaZeroFileIO.WriteHeartbeat()
    local heartbeat = {
        timestamp = safeTimestamp(),
        lua_alive = true,
        turn = currentTurn(),
    }

    if not AlphaZeroJson then return false, "JSON encoder not available" end
    local json, err = AlphaZeroJson.Encode(heartbeat, false)
    if not json then return false, err end

    if HAS_IO then
        local path = AlphaZeroConfig and AlphaZeroConfig.Get("HEARTBEAT_PATH")
        if not path then return false, "HEARTBEAT_PATH not configured" end
        return AlphaZeroFileIO.Write(path, json)
    else
        -- Print-based fallback + ExposedMembers
        print("AZ_HB:" .. json)
        if ExposedMembers then
            ExposedMembers.AlphaZeroHeartbeat = json
        end
        return true, nil
    end
end

-- Write error
function AlphaZeroFileIO.WriteError(code, message, context)
    local errorData = {
        timestamp = safeTimestamp(),
        code = code,
        message = message,
        context = context,
        turn = currentTurn(),
    }

    if not AlphaZeroJson then return false, "JSON encoder not available" end
    local json, err = AlphaZeroJson.Encode(errorData, false)
    if not json then return false, err end

    if HAS_IO then
        local path = AlphaZeroConfig and AlphaZeroConfig.Get("ERROR_PATH")
        if not path then return false, "ERROR_PATH not configured" end
        return AlphaZeroFileIO.Append(path, json .. "\n")
    else
        print("AZ_ERR:" .. json)
        return true, nil
    end
end

-- Clear all communication files / ExposedMembers
function AlphaZeroFileIO.ClearAll()
    if HAS_IO then
        local files = {
            AlphaZeroConfig and AlphaZeroConfig.Get("STATE_PATH"),
            AlphaZeroConfig and AlphaZeroConfig.Get("COMMANDS_PATH"),
            AlphaZeroConfig and AlphaZeroConfig.Get("ACK_PATH"),
            AlphaZeroConfig and AlphaZeroConfig.Get("ERROR_PATH"),
        }
        for _, path in ipairs(files) do
            if path then
                AlphaZeroFileIO.Delete(path)
                AlphaZeroFileIO.ResetChangeTracking(path)
            end
        end
    end

    -- Clear ExposedMembers regardless
    if ExposedMembers then
        ExposedMembers.AlphaZeroState = nil
        ExposedMembers.AlphaZeroCommand = nil
        ExposedMembers.AlphaZeroAck = nil
        ExposedMembers.AlphaZeroHeartbeat = nil
    end

    print("[AlphaZero] Cleared all communication channels")
end

-- Get path separator (for reuse by other modules)
function AlphaZeroFileIO.GetPathSep()
    return getPathSep()
end

-- Initialize on load
AlphaZeroFileIO.Init()

print("[AlphaZero] FileIO module loaded (io=" .. (HAS_IO and "file" or "print") .. " mode)")


-- ========== END FileIO.lua ==========

-- ========== BEGIN StateExporter.lua ==========
-- ============================================================================
-- StateExporter.lua - Game state serialization for AlphaZero AI Bridge
-- ============================================================================
-- Exports the current game state to JSON format for the Python agent.
-- Fixes: nil guards, hex-grid neighbors, production items, civics export.
-- ============================================================================

AlphaZeroStateExporter = AlphaZeroStateExporter or {}

local Log = nil  -- Will be set after Logger loads

-- Cache for expensive lookups
local unitTypeCache = {}
local buildingTypeCache = {}

-- Sentinel value for invalid Civ6 type IDs (features, resources, improvements)
local INVALID_TYPE_ID = -1

-- Version of the exported state's shape; a state without it is version 1.
-- 2: researched techs by name, current research and civic with cost, progress
--    and boost, city buildings and districts, research legal actions, and the
--    notification that blocks ending the turn.
local STATE_SCHEMA_VERSION = 2

-- GameInfo row type names by row Index, one table per GameInfo table name.
local typeNameCaches = {}

-- ----------------------------------------------------------------------------
-- Helper Functions
-- ----------------------------------------------------------------------------

-- Get unit type info
local function getUnitTypeInfo(unitTypeId)
    if unitTypeCache[unitTypeId] then
        return unitTypeCache[unitTypeId]
    end

    local info = {
        type_name = "UNIT_UNKNOWN",
        combat_class = "MELEE",
        is_civilian = false,
    }

    if GameInfo and GameInfo.Units then
        for row in GameInfo.Units() do
            if row.Index == unitTypeId then
                info.type_name = row.UnitType or "UNIT_UNKNOWN"
                info.combat_class = row.Combat or 0
                info.is_civilian = row.FormationClass == "FORMATION_CLASS_CIVILIAN"
                break
            end
        end
    end

    unitTypeCache[unitTypeId] = info
    return info
end

-- Get building type name
local function getBuildingTypeName(buildingTypeId)
    if buildingTypeCache[buildingTypeId] then
        return buildingTypeCache[buildingTypeId]
    end

    local name = "BUILDING_UNKNOWN"
    if GameInfo and GameInfo.Buildings then
        for row in GameInfo.Buildings() do
            if row.Index == buildingTypeId then
                name = row.BuildingType
                break
            end
        end
    end

    buildingTypeCache[buildingTypeId] = name
    return name
end

-- Get terrain type name
local function getTerrainTypeName(terrainTypeId)
    if GameInfo and GameInfo.Terrains then
        for row in GameInfo.Terrains() do
            if row.Index == terrainTypeId then
                return row.TerrainType
            end
        end
    end
    return "TERRAIN_UNKNOWN"
end

-- Get feature type name
local function getFeatureTypeName(featureTypeId)
    if featureTypeId == INVALID_TYPE_ID then return nil end

    if GameInfo and GameInfo.Features then
        for row in GameInfo.Features() do
            if row.Index == featureTypeId then
                return row.FeatureType
            end
        end
    end
    return nil
end

-- Get resource type name
local function getResourceTypeName(resourceTypeId)
    if resourceTypeId == INVALID_TYPE_ID then return nil end

    if GameInfo and GameInfo.Resources then
        for row in GameInfo.Resources() do
            if row.Index == resourceTypeId then
                return row.ResourceType
            end
        end
    end
    return nil
end

-- Get improvement type name
local function getImprovementTypeName(improvementTypeId)
    if improvementTypeId == INVALID_TYPE_ID then return nil end

    if GameInfo and GameInfo.Improvements then
        for row in GameInfo.Improvements() do
            if row.Index == improvementTypeId then
                return row.ImprovementType
            end
        end
    end
    return nil
end

-- Call a method on a game object. Returns nil when the object or the method is
-- missing, or when the call raises: which methods exist depends on the Lua
-- context the game runs this script in, and a missing one must cost one field,
-- not the whole export.
local function safeCall(object, methodName, ...)
    if object == nil then return nil end
    local okMethod, method = pcall(function() return object[methodName] end)
    if not okMethod or not method then return nil end
    local okCall, result = pcall(method, object, ...)
    if okCall then return result end
    return nil
end

-- Type name of a GameInfo row by its Index; nil when the table or row is missing.
local function typeNameByIndex(tableName, typeField, index)
    local cache = typeNameCaches[tableName]
    if not cache then
        cache = {}
        local gameTable = GameInfo and GameInfo[tableName]
        if gameTable then
            pcall(function()
                for row in gameTable() do
                    cache[row.Index] = row[typeField]
                end
            end)
        end
        -- Keep an empty result out of the cache: the table may not be ready yet.
        if next(cache) ~= nil then
            typeNameCaches[tableName] = cache
        end
    end
    return cache[index]
end

local productionHashCache = nil
local PRODUCTION_HASH_TABLES = {
    { "Units", "UnitType" },
    { "Buildings", "BuildingType" },
    { "Districts", "DistrictType" },
    { "Projects", "ProjectType" },
}

-- Lazy hash-to-type lookup for city build queues when CurrentlyBuilding() returns nil
-- (live InGame UI context exposes GetCurrentProductionTypeHash() instead).
local function typeNameByHash(hash)
    if type(hash) ~= "number" or hash == 0 then
        return nil
    end
    if not productionHashCache then
        local cache = {}
        if GameInfo then
            for _, entry in ipairs(PRODUCTION_HASH_TABLES) do
                local gameTable = GameInfo[entry[1]]
                if gameTable then
                    pcall(function()
                        for row in gameTable() do
                            if row.Hash ~= nil and row[entry[2]] then
                                cache[row.Hash] = row[entry[2]]
                            end
                        end
                    end)
                end
            end
        end
        if next(cache) ~= nil then
            productionHashCache = cache
        end
    end
    return productionHashCache and productionHashCache[hash] or nil
end

-- Resolve a city's active production item name across stub and live UI contexts.
local function getCurrentProductionName(buildQueue)
    local current = safeCall(buildQueue, "CurrentlyBuilding")
    if type(current) == "string" and current ~= "" then
        return current
    end
    local hash = safeCall(buildQueue, "GetCurrentProductionTypeHash")
    if type(hash) == "number" and hash ~= 0 then
        local resolved = typeNameByHash(hash)
        if resolved then
            if Log and Log.Debug then
                Log.Debug("Resolved production hash %.0f -> %s", hash, resolved)
            end
            return resolved
        end
        if Log and Log.Debug then
            Log.Debug("Unmapped production hash %.0f; falling back to PRODUCTION_HASH_", hash)
        end
        return "PRODUCTION_HASH_" .. string.format("%.0f", hash)
    end
    return nil
end

local gameInfoYieldCache = nil
local function yieldIndexFromGameInfo(prefixedKey)
    if not gameInfoYieldCache and GameInfo and GameInfo.Yields then
        local cache = {}
        pcall(function()
            for row in GameInfo.Yields() do
                if row.YieldType and row.Index ~= nil then
                    cache[row.YieldType] = row.Index
                end
            end
        end)
        if next(cache) ~= nil then
            gameInfoYieldCache = cache
        end
    end
    return gameInfoYieldCache and gameInfoYieldCache[prefixedKey] or nil
end

-- Live Civ6 YieldTypes uses short keys (FOOD=0..FAITH=5), whereas unit stubs
-- and XML types use YIELD_FOOD..YIELD_FAITH.
local function resolveYieldIndex(shortKey, prefixedKey)
    if YieldTypes then
        if YieldTypes[shortKey] ~= nil then return YieldTypes[shortKey] end
        if YieldTypes[prefixedKey] ~= nil then return YieldTypes[prefixedKey] end
    end
    return yieldIndexFromGameInfo(prefixedKey)
end

local function readPlotYield(plot, shortKey, prefixedKey)
    local idx = resolveYieldIndex(shortKey, prefixedKey)
    if idx == nil then return 0 end
    return plot:GetYield(idx) or 0
end

-- Type names of the GameInfo rows for which owner:<hasMethod>(row.Index) is
-- true, in GameInfo order (the techs a player has, the buildings in a city).
local function collectOwnedTypes(tableName, typeField, owner, hasMethod)
    local names = {}
    local gameTable = GameInfo and GameInfo[tableName]
    if not gameTable or owner == nil then
        return names
    end
    pcall(function()
        for row in gameTable() do
            if safeCall(owner, hasMethod, row.Index) then
                table.insert(names, row[typeField])
            end
        end
    end)
    return names
end

-- Name of an EndTurnBlockingTypes value.
local function endTurnBlockingName(blocking)
    if EndTurnBlockingTypes then
        for name, value in pairs(EndTurnBlockingTypes) do
            if value == blocking then
                return name
            end
        end
    end
    return "ENDTURN_BLOCKING_UNKNOWN_" .. tostring(blocking)
end

-- Get civic type name from index
local function getCivicTypeName(civicId)
    if GameInfo and GameInfo.Civics then
        for row in GameInfo.Civics() do
            if row.Index == civicId then
                return row.CivicType
            end
        end
    end
    return "CIVIC_UNKNOWN_" .. tostring(civicId)
end

-- The plots one step from (x, y), from the game's own adjacency (which also
-- knows where the map wraps). No table of offsets is kept here: the one this
-- file used to carry was wrong for both row parities. Where the game's call
-- is missing there are no neighbours, and so no moves are offered.
local DIRECTION_NAMES = {
    "DIRECTION_NORTHEAST", "DIRECTION_EAST", "DIRECTION_SOUTHEAST",
    "DIRECTION_SOUTHWEST", "DIRECTION_WEST", "DIRECTION_NORTHWEST",
}
local function getHexNeighbors(x, y)
    local neighbors = {}
    if not Map or not Map.GetAdjacentPlot or not DirectionTypes then
        return neighbors
    end
    for _, name in ipairs(DIRECTION_NAMES) do
        local direction = DirectionTypes[name]
        if direction ~= nil then
            local ok, plot = pcall(Map.GetAdjacentPlot, x, y, direction)
            if ok and plot then
                table.insert(neighbors, {plot:GetX(), plot:GetY()})
            end
        end
    end
    return neighbors
end

-- Get all hex tiles within a given range (for ranged attacks)
local function getHexTilesInRange(cx, cy, range)
    local tiles = {}
    local seen = {}
    -- BFS from center
    local queue = {{cx, cy, 0}}
    seen[cx .. "," .. cy] = true
    while #queue > 0 do
        local current = table.remove(queue, 1)
        local qx, qy, dist = current[1], current[2], current[3]
        if dist > 0 then
            table.insert(tiles, {qx, qy})
        end
        if dist < range then
            local neighbors = getHexNeighbors(qx, qy)
            for _, n in ipairs(neighbors) do
                local key = n[1] .. "," .. n[2]
                if not seen[key] then
                    seen[key] = true
                    table.insert(queue, {n[1], n[2], dist + 1})
                end
            end
        end
    end
    return tiles
end

-- Whether other players are exported at all (OBSERVE_ALL_PLAYERS).
local function observesOthers()
    return (AlphaZeroConfig and AlphaZeroConfig.Get("OBSERVE_ALL_PLAYERS")) == true
end

-- The ids of the other living players, as the game's own UI finds them:
-- Game.GetPlayers{Alive = true}, else every id below GameDefines.MAX_PLAYERS
-- that has a living player. Barbarians and city-states have high ids, so a
-- count is never used as an id. Nobody when neither can be asked.
local function otherPlayerIds(playerId)
    local ids, seen = {}, {}
    local function add(id)
        if type(id) == "number" and id ~= playerId and not seen[id] then
            seen[id] = true
            table.insert(ids, id)
        end
    end
    local listed = pcall(function()
        for _, player in ipairs(Game.GetPlayers({ Alive = true })) do
            add(player:GetID())
        end
    end)
    if not listed then
        if Log then Log.Info("Game.GetPlayers could not be asked; walking the player ids") end
        ids, seen = {}, {}
        pcall(function()
            for id = 0, GameDefines.MAX_PLAYERS - 1 do
                local player = Players[id]
                if player and player:IsAlive() then add(id) end
            end
        end)
    end
    table.sort(ids)
    return ids
end

-- Get observable player IDs (own player first, then the others when observed)
local function getObservablePlayerIds(playerId)
    local playerIds = {playerId}
    if observesOthers() then
        for _, id in ipairs(otherPlayerIds(playerId)) do
            table.insert(playerIds, id)
        end
    end
    return playerIds
end

-- The player's visibility table, or nil when the game cannot be asked.
local function visibilityOf(playerId)
    local ok, visibility = pcall(function()
        return PlayersVisibility and PlayersVisibility[playerId]
    end)
    return ok and visibility or nil
end

-- Whether the player sees the tile now. False when the game cannot say.
local function seesTile(visibility, x, y)
    if not visibility then return false end
    local ok, visible = pcall(function() return visibility:IsVisible(x, y) end)
    return ok and visible == true
end

-- Whether the player sees this unit: its tile and the unit itself, as the
-- game's UI asks (a stealth unit on a visible tile is not seen). False when
-- either cannot be asked.
local function seesUnit(visibility, unit, x, y)
    if not seesTile(visibility, x, y) then return false end
    local ok, visible = pcall(function() return visibility:IsUnitVisible(unit) end)
    return ok and visible == true
end

-- What another player is to this one: { id, kind, at_war, met }. at_war and
-- met are left out when the game cannot be asked (unknown, never "at peace").
local function describeOtherPlayer(playerId, otherId)
    local entry = { id = otherId, kind = "unknown" }
    local other = Players and Players[otherId]
    -- true / false as the game answers; nil when the method raises or gives
    -- no boolean. IsFreeCities does not exist without the expansions, which
    -- is an answer (no), not a failure.
    local function is(method, absentMeans)
        if other and other[method] == nil then return absentMeans end
        local ok, value = pcall(function() return other[method](other) end)
        if ok and type(value) == "boolean" then return value end
        return nil
    end
    local barbarian, free, major = is("IsBarbarian"), is("IsFreeCities", false), is("IsMajor")
    if barbarian == true then
        entry.kind = "barbarian"
    elseif free == true then
        entry.kind = "free"
    elseif major == true and barbarian == false then
        entry.kind = "major"
    elseif major == false and barbarian == false and free == false then
        entry.kind = "minor"
    end
    if entry.kind == "barbarian" then
        entry.at_war = true
    end
    pcall(function()
        local diplomacy = Players[playerId]:GetDiplomacy()
        local okWar, atWar = pcall(function() return diplomacy:IsAtWarWith(otherId) end)
        if okWar and type(atWar) == "boolean" and entry.at_war == nil then
            entry.at_war = atWar
        end
        local okMet, met = pcall(function() return diplomacy:HasMet(otherId) end)
        if okMet and type(met) == "boolean" then
            entry.met = met
        end
    end)
    return entry
end

-- Whether the executor would carry this command out now. The executor's
-- own check, so an offered action is one that will not be refused.
local function wouldBeAccepted(playerId, command)
    if not AlphaZeroActionExecutor or not AlphaZeroActionExecutor.Check then
        return false
    end
    local ok, accepted = pcall(AlphaZeroActionExecutor.Check, playerId, command)
    return ok and accepted == true
end

-- The items of one GameInfo table a city could start now, by type name.
-- Asked through the executor's check: by the row's Hash, the strict form,
-- and nothing that needs a plot.
local function collectProducibleItems(playerId, cityId, gameInfoTable, typeField, itemCategory, items, itemTypes)
    if not gameInfoTable then
        return
    end
    for row in gameInfoTable() do
        local name = row[typeField]
        if name and wouldBeAccepted(playerId, { action_type = "city_produce", city_id = cityId, item_id = name }) then
            table.insert(items, name)
            itemTypes[name] = itemCategory
        end
    end
end

-- ----------------------------------------------------------------------------
-- State Export Functions
-- ----------------------------------------------------------------------------

-- Export map data
local function exportMap(playerId)
    local mapData = {
        width = 0,
        height = 0,
        tiles = {},
    }

    if not Map then
        if Log then Log.Warn("Map object not available") end
        return mapData
    end

    local w, h = Map.GetGridSize()
    mapData.width = w
    mapData.height = h

    local maxTiles = (AlphaZeroConfig and AlphaZeroConfig.Get("MAX_TILES_PER_STATE")) or 10000
    local tileCount = 0
    local includeFog = AlphaZeroConfig and AlphaZeroConfig.Get("INCLUDE_FOG_OF_WAR")

    local playerVisibility = visibilityOf(playerId)

    for x = 0, mapData.width - 1 do
        for y = 0, mapData.height - 1 do
            if tileCount >= maxTiles then
                break
            end

            local plot = Map.GetPlot(x, y)
            if plot then
                -- Nothing is visible when the game cannot be asked.
                local isVisible = seesTile(playerVisibility, x, y)
                local isRevealed = false
                if playerVisibility then
                    local okRevealed, revealed = pcall(function()
                        return playerVisibility:IsRevealed(x, y)
                    end)
                    isRevealed = okRevealed and revealed == true
                end

                if isRevealed or not includeFog then
                    local tile = {
                        x = x,
                        y = y,
                        terrain = getTerrainTypeName(plot:GetTerrainType()),
                        features = {},
                        is_visible = isVisible,
                        is_revealed = isRevealed,
                        yields = {},
                    }
                    -- Who owns a tile, what is built on it and what it holds
                    -- are facts of now: only for a tile the player sees now.
                    if isVisible then
                        tile.resource = getResourceTypeName(plot:GetResourceType())
                        tile.improvement = getImprovementTypeName(plot:GetImprovementType())
                        tile.owner_id = plot:GetOwner()
                    end

                    -- Add features
                    local featureType = plot:GetFeatureType()
                    if featureType >= 0 then
                        local featureName = getFeatureTypeName(featureType)
                        if featureName then
                            table.insert(tile.features, featureName)
                        end
                    end

                    -- Check for hills
                    if plot:IsHills() then
                        table.insert(tile.features, "FEATURE_HILLS")
                    end

                    -- Check for mountains
                    if plot:IsMountain() then
                        table.insert(tile.features, "FEATURE_MOUNTAIN")
                    end

                    -- Add yields if visible (with nil guard on YieldTypes / GameInfo.Yields)
                    if isVisible and (YieldTypes or (GameInfo and GameInfo.Yields)) then
                        tile.yields = {
                            food = readPlotYield(plot, "FOOD", "YIELD_FOOD"),
                            production = readPlotYield(plot, "PRODUCTION", "YIELD_PRODUCTION"),
                            gold = readPlotYield(plot, "GOLD", "YIELD_GOLD"),
                            science = readPlotYield(plot, "SCIENCE", "YIELD_SCIENCE"),
                            culture = readPlotYield(plot, "CULTURE", "YIELD_CULTURE"),
                            faith = readPlotYield(plot, "FAITH", "YIELD_FAITH"),
                        }
                    end

                    table.insert(mapData.tiles, tile)
                    tileCount = tileCount + 1
                end
            end
        end
    end

    if Log then Log.Debug("Exported %d tiles", #mapData.tiles) end
    return mapData
end

-- Export unit data
local function exportUnits(playerId)
    local units = {}

    if not Players then
        return units
    end

    local playerVisibility = visibilityOf(playerId)
    local playerIds = getObservablePlayerIds(playerId)

    for _, pid in ipairs(playerIds) do
        local player = Players[pid]
        local okUnits, playerUnits = pcall(function() return player and player:GetUnits() end)
        if okUnits and playerUnits then
            local function walk()
                for i, unit in playerUnits:Members() do
                    local x, y = unit:GetX(), unit:GetY()
                    local own = (pid == playerId)

                    -- Another player's unit only when this player sees it.
                    local canSee = own
                    if not own then
                        canSee = x ~= -9999
                            and safeCall(unit, "IsDead") ~= true
                            and safeCall(unit, "IsDelayedDeath") ~= true
                            and seesUnit(playerVisibility, unit, x, y)
                    end

                    if canSee and not own then
                        -- Only what the game shows of somebody else's unit:
                        -- no moves left, attacks left or promotions.
                        local typeInfo = getUnitTypeInfo(unit:GetUnitType())
                        local dmg = safeCall(unit, "GetDamage") or 0
                        table.insert(units, {
                            id = unit:GetID(),
                            type_name = typeInfo.type_name,
                            owner = pid,
                            x = x,
                            y = y,
                            hp = 100 - dmg,
                            max_hp = 100,
                            damage = dmg,
                            combat_strength = safeCall(unit, "GetCombat") or 0,
                            ranged_strength = safeCall(unit, "GetRangedCombat") or 0,
                            range = safeCall(unit, "GetRange") or 0,
                            is_civilian = typeInfo.is_civilian,
                        })
                    elseif canSee then
                        local typeInfo = getUnitTypeInfo(unit:GetUnitType())

                        -- Wrap unit data construction in pcall — some methods
                        -- may not exist in AddUserInterfaces context
                        local ok_unit, unitData = pcall(function()
                            local dmg = unit:GetDamage() or 0
                            return {
                                id = unit:GetID(),
                                type_name = typeInfo.type_name,
                                owner = pid,
                                x = x,
                                y = y,
                                hp = 100 - dmg,
                                max_hp = 100,
                                damage = dmg,
                                movement = unit:GetMovementMovesRemaining() or 0,
                                max_movement = unit:GetMaxMoves() or 2,
                                combat_strength = unit:GetCombat() or 0,
                                ranged_strength = unit:GetRangedCombat() or 0,
                                range = unit:GetRange() or 0,
                                is_civilian = typeInfo.is_civilian,
                                can_attack = (unit:GetAttacksRemaining() or 0) > 0,
                                promotions = {},
                            }
                        end)

                        if not ok_unit then
                            -- Fallback: minimal unit data
                            unitData = {
                                id = unit:GetID(), type_name = typeInfo.type_name,
                                owner = pid, x = x, y = y,
                                hp = 100, max_hp = 100, damage = 0,
                                movement = 0, max_movement = 2,
                                combat_strength = 0, ranged_strength = 0, range = 0,
                                is_civilian = typeInfo.is_civilian, can_attack = false,
                                promotions = {},
                            }
                        end

                        -- Get promotions (best effort)
                        pcall(function()
                            local unitExp = unit:GetExperience()
                            if unitExp then
                                local promotions = unitExp:GetPromotions()
                                for _, promo in ipairs(promotions or {}) do
                                    table.insert(unitData.promotions, promo)
                                end
                            end
                        end)

                        table.insert(units, unitData)
                    end
                end
            end
            -- Somebody else's unit that cannot be read must not cost the
            -- player its own: only the own walk may fail the export.
            if pid == playerId then walk() else pcall(walk) end
        end
    end

    if Log then Log.Debug("Exported %d units", #units) end
    return units
end

-- Export city data
local function exportCities(playerId)
    local cities = {}

    if not Players then
        return cities
    end

    local playerVisibility = visibilityOf(playerId)
    local playerIds = getObservablePlayerIds(playerId)

    for _, pid in ipairs(playerIds) do
        local player = Players[pid]
        local okCities, playerCities = pcall(function() return player and player:GetCities() end)
        if okCities and playerCities then
            local function walk()
                for i, city in playerCities:Members() do
                    local x, y = city:GetX(), city:GetY()
                    local own = (pid == playerId)

                    -- Another player's city only while its tile is seen now:
                    -- a city seen once and now in fog would give away who
                    -- holds it today.
                    local canSee = own or seesTile(playerVisibility, x, y)

                    if canSee and not own then
                        local ok_cap, is_cap = pcall(function() return city:IsCapital() end)
                        table.insert(cities, {
                            id = city:GetID(),
                            name = safeCall(city, "GetName") or "Unknown",
                            owner = pid,
                            x = x,
                            y = y,
                            population = safeCall(city, "GetPopulation") or 1,
                            is_capital = (ok_cap and is_cap) or false,
                            buildings = {},
                            districts = {},
                            available_production = {},
                        })
                    elseif canSee then
                        -- pcall wrappers for city methods that may not exist in all contexts
                        local ok_cap, is_cap = pcall(function() return city:IsCapital() end)
                        local ok_str, str_val = pcall(function() return city:GetStrengthValue() end)

                        local cityData = {
                            id = city:GetID(),
                            name = city:GetName() or "Unknown",
                            owner = pid,
                            x = x,
                            y = y,
                            population = city:GetPopulation() or 1,
                            is_capital = (ok_cap and is_cap) or false,
                            defense_strength = (ok_str and str_val) or 0,
                            buildings = {},
                            districts = {},
                            current_production = nil,
                            production_progress = 0,
                            available_production = {},
                        }

                        -- Only include detailed info for own cities
                        if pid == playerId then
                            local cityBuildings = safeCall(city, "GetBuildings")
                            if cityBuildings then
                                cityData.buildings = collectOwnedTypes(
                                    "Buildings", "BuildingType", cityBuildings, "HasBuilding")
                            end

                            local cityDistricts = safeCall(city, "GetDistricts")
                            if cityDistricts then
                                pcall(function()
                                    for _, district in cityDistricts:Members() do
                                        local districtType = safeCall(district, "GetType")
                                        local districtName = districtType ~= nil
                                            and typeNameByIndex("Districts", "DistrictType", districtType)
                                        -- A district is listed once placed; only a finished
                                        -- one counts. Without IsComplete it is listed.
                                        if districtName and safeCall(district, "IsComplete") ~= false then
                                            table.insert(cityData.districts, districtName)
                                        end
                                    end
                                end)
                            end

                            local ok_bq, buildQueue = pcall(function() return city:GetBuildQueue() end)
                            if ok_bq and buildQueue then
                                local currentProduction = getCurrentProductionName(buildQueue)
                                if currentProduction then
                                    cityData.current_production = currentProduction
                                end
                                local ok_tl, turnsLeft = pcall(function() return buildQueue:GetTurnsLeft() end)
                                cityData.production_progress = (ok_tl and turnsLeft) or 0
                            end
                        end

                        table.insert(cities, cityData)
                    end
                end
            end
            if pid == playerId then walk() else pcall(walk) end
        end
    end

    if Log then Log.Debug("Exported %d cities", #cities) end
    return cities
end

-- Export player data (including civics)
local function exportPlayer(playerId)
    local playerData = {
        id = playerId,
        civilization = "UNKNOWN",
        gold = 0,
        science_per_turn = 0,
        culture_per_turn = 0,
        faith = 0,
        score = 0,
        is_alive = false,
        is_human = false,
        techs_researched = {},
        current_research = nil,
        civics_researched = {},
        current_civic = nil,
    }

    local player = Players and Players[playerId]
    if not player then
        return playerData
    end

    playerData.is_alive = player:IsAlive()
    playerData.is_human = player:IsHuman()

    -- Get civilization
    if PlayerConfigurations then
        local config = PlayerConfigurations[playerId]
        if config then
            playerData.civilization = config:GetCivilizationTypeName() or "UNKNOWN"
        end
    end

    -- Treasury
    local treasury = player:GetTreasury()
    if treasury then
        playerData.gold = treasury:GetGoldBalance() or 0
        playerData.gold_per_turn = treasury:GetGoldYield() or 0
    end

    -- Culture/Science
    local playerCulture = player:GetCulture()
    if playerCulture then
        playerData.culture_per_turn = playerCulture:GetCultureYield() or 0

        -- Export current civic
        if playerCulture.GetProgressingCivic then
            local currentCivic = playerCulture:GetProgressingCivic()
            if currentCivic and currentCivic >= 0 then
                playerData.current_civic = getCivicTypeName(currentCivic)
            end
        end

        -- Cost, progress and boost of the civic in progress
        local currentCivic = safeCall(playerCulture, "GetProgressingCivic")
        if type(currentCivic) == "number" and currentCivic >= 0 then
            playerData.civic_status = {
                type = getCivicTypeName(currentCivic),
                index = currentCivic,
                cost = safeCall(playerCulture, "GetCultureCost", currentCivic),
                progress = safeCall(playerCulture, "GetCulturalProgress", currentCivic),
                turns_left = safeCall(playerCulture, "GetTurnsToProgressCivic", currentCivic)
                    or safeCall(playerCulture, "GetTurnsLeftOnCurrentCivic"),
                boost_triggered = safeCall(playerCulture, "HasBoostBeenTriggered", currentCivic),
            }
        end

        -- Export researched civics: ask per civic; fall back to the list call
        -- where HasCivic does not exist.
        local okHasCivic, hasCivic = pcall(function() return playerCulture.HasCivic end)
        if okHasCivic and hasCivic then
            playerData.civics_researched = collectOwnedTypes(
                "Civics", "CivicType", playerCulture, "HasCivic")
        elseif playerCulture.GetResearchedCivics then
            local civics = playerCulture:GetResearchedCivics()
            if civics then
                for _, civicId in ipairs(civics) do
                    table.insert(playerData.civics_researched, getCivicTypeName(civicId))
                end
            end
        end
    end

    local playerTech = player:GetTechs()
    if playerTech then
        playerData.science_per_turn = playerTech:GetScienceYield() or 0
        playerData.techs_researched = collectOwnedTypes(
            "Technologies", "TechnologyType", playerTech, "HasTech")

        -- Current research by type name, with its cost, progress and boost
        local currentTech = safeCall(playerTech, "GetResearchingTech")
        if type(currentTech) == "number" and currentTech >= 0 then
            local techName = typeNameByIndex("Technologies", "TechnologyType", currentTech)
                or ("TECH_UNKNOWN_" .. tostring(currentTech))
            playerData.current_research = techName
            playerData.research_status = {
                type = techName,
                index = currentTech,
                cost = safeCall(playerTech, "GetResearchCost", currentTech),
                progress = safeCall(playerTech, "GetResearchProgress", currentTech),
                turns_left = safeCall(playerTech, "GetTurnsToResearch", currentTech),
                boost_triggered = safeCall(playerTech, "HasBoostBeenTriggered", currentTech),
            }
        end
    end

    -- Religion
    local playerReligion = player:GetReligion()
    if playerReligion then
        playerData.faith = playerReligion:GetFaithBalance() or 0
    end

    -- Score
    playerData.score = player:GetScore() or 0

    return playerData
end

-- Research choices open to the player now, as legal actions. `items` holds
-- GameInfo row indexes (what the research commands take) and `item_names` the
-- matching type names. The tech or civic already in progress is left out:
-- choosing it again changes nothing.
local function appendResearchActions(player, actions)
    -- Technologies: the game answers directly.
    local playerTech = safeCall(player, "GetTechs")
    local techTable = GameInfo and GameInfo.Technologies
    if playerTech and techTable then
        local currentTech = safeCall(playerTech, "GetResearchingTech")
        local items, names = {}, {}
        for row in techTable() do
            if row.Index ~= currentTech
               and safeCall(playerTech, "CanResearch", row.Index)
               and not safeCall(playerTech, "HasTech", row.Index) then
                table.insert(items, row.Index)
                table.insert(names, row.TechnologyType)
            end
        end
        if #items > 0 then
            table.insert(actions, { action = "research_tech", items = items, item_names = names })
        end
    end

    -- Civics: the game answers directly (CanProgress, as its own civics chooser
    -- asks). Where that method is missing, a civic is open when the player
    -- lacks it and has every prerequisite listed for it.
    local playerCulture = safeCall(player, "GetCulture")
    local civicTable = GameInfo and GameInfo.Civics
    local okHasCivic, hasCivicMethod = pcall(function() return playerCulture.HasCivic end)
    if playerCulture and civicTable and okHasCivic and hasCivicMethod then
        local okCanProgress, canProgressMethod = pcall(function() return playerCulture.CanProgress end)
        local askTheGame = okCanProgress and canProgressMethod ~= nil
        local prereqTable = GameInfo.CivicPrereqs

        local indexByType, prereqs = {}, {}
        if not askTheGame and prereqTable then
            for row in civicTable() do
                indexByType[row.CivicType] = row.Index
            end
            for row in prereqTable() do
                prereqs[row.Civic] = prereqs[row.Civic] or {}
                table.insert(prereqs[row.Civic], row.PrereqCivic)
            end
        end

        local currentCivic = safeCall(playerCulture, "GetProgressingCivic")
        local items, names = {}, {}
        if askTheGame or prereqTable then
            for row in civicTable() do
                if row.Index ~= currentCivic and not safeCall(playerCulture, "HasCivic", row.Index) then
                    local open
                    if askTheGame then
                        open = safeCall(playerCulture, "CanProgress", row.Index) and true or false
                    else
                        open = true
                        for _, prereqType in ipairs(prereqs[row.CivicType] or {}) do
                            local prereqIndex = indexByType[prereqType]
                            if prereqIndex ~= nil and not safeCall(playerCulture, "HasCivic", prereqIndex) then
                                open = false
                                break
                            end
                        end
                    end
                    if open then
                        table.insert(items, row.Index)
                        table.insert(names, row.CivicType)
                    end
                end
            end
        end
        if #items > 0 then
            table.insert(actions, { action = "research_civic", items = items, item_names = names })
        end
    end
end

-- Export legal actions (with hex-grid neighbors and actual production items)
local function exportLegalActions(playerId)
    local actions = {}

    -- End turn is always available
    table.insert(actions, { action = "end_turn" })

    local player = Players and Players[playerId]
    if not player then
        return actions
    end

    -- Unit actions
    local playerUnits = player:GetUnits()
    if playerUnits then
        for _, unit in playerUnits:Members() do
            local unitId = unit:GetID()
            local ux, uy = unit:GetX(), unit:GetY()
            local neighbors = getHexNeighbors(ux, uy)

            -- Every action below is offered exactly when the executor's own
            -- check accepts the command it stands for.
            local function accepted(actionType, target)
                local command = { action_type = actionType, unit_id = unitId }
                if target then command.target = { x = target[1], y = target[2] } end
                return wouldBeAccepted(playerId, command)
            end
            local function acceptedTargets(actionType, candidates)
                local targets = {}
                for _, candidate in ipairs(candidates) do
                    if accepted(actionType, candidate) then
                        table.insert(targets, {candidate[1], candidate[2]})
                    end
                end
                return targets
            end

            local moveTargets = acceptedTargets("unit_move", neighbors)
            if #moveTargets > 0 then
                table.insert(actions, { action = "move", unit_id = unitId, targets = moveTargets })
            end
            if accepted("unit_fortify") then
                table.insert(actions, { action = "fortify", unit_id = unitId })
            end
            if accepted("unit_skip") then
                table.insert(actions, { action = "skip", unit_id = unitId })
            end
            if accepted("unit_found_city") then
                table.insert(actions, { action = "found_city", unit_id = unitId })
            end

            -- An attack is offered only on what the same state shows. With
            -- OBSERVE_ALL_PLAYERS off the state holds no foreign unit or city,
            -- so no attack is offered; the executor still carries one out when
            -- it is sent by hand.
            local attackTargets = {}
            if observesOthers() then
                attackTargets = acceptedTargets("unit_attack", neighbors)
            end
            if #attackTargets > 0 then
                table.insert(actions, { action = "attack", unit_id = unitId, targets = attackTargets })
            end
            local range = unit:GetRange() or 0
            if range >= 1 and observesOthers() then
                local rangedTargets = acceptedTargets("unit_ranged_attack", getHexTilesInRange(ux, uy, range))
                if #rangedTargets > 0 then
                    table.insert(actions, { action = "ranged_attack", unit_id = unitId, targets = rangedTargets })
                end
            end
        end
    end

    -- City production actions (with actual available items)
    local ok_cities, playerCities = pcall(function() return player:GetCities() end)
    if ok_cities and playerCities then
        for _, city in playerCities:Members() do
            local ok_bq, buildQueue = pcall(function() return city:GetBuildQueue() end)
            if ok_bq and buildQueue then
                local currentlyBuilding = getCurrentProductionName(buildQueue)
                local needsProduction = not currentlyBuilding
                -- Always export available production for each city
                local items = {}
                local itemTypes = {}

                if GameInfo then
                    local cityId = city:GetID()
                    pcall(collectProducibleItems, playerId, cityId, GameInfo.Units, "UnitType", "unit", items, itemTypes)
                    pcall(collectProducibleItems, playerId, cityId, GameInfo.Buildings, "BuildingType", "building", items, itemTypes)
                end

                if #items > 0 or needsProduction then
                    table.insert(actions, {
                        action = "produce",
                        city_id = city:GetID(),
                        items = items,
                        item_types = itemTypes,
                    })
                end
            end
        end
    end

    -- Research choices. Guarded as a whole: a failure here must not cost the
    -- unit and city actions above.
    local okResearch, researchErr = pcall(appendResearchActions, player, actions)
    if not okResearch and Log then
        Log.Warn("Research legal actions failed: %s", tostring(researchErr))
    end

    if Log then Log.Debug("Exported %d legal actions", #actions) end
    return actions
end

-- ----------------------------------------------------------------------------
-- Public API
-- ----------------------------------------------------------------------------

-- Initialize module
function AlphaZeroStateExporter.Init()
    if AlphaZeroLogger then
        Log = AlphaZeroLogger.CreateLogger("StateExporter")
    end
end

-- A notification's message as text, or nil.
local function notificationMessage(entry)
    local text = safeCall(entry, "GetMessage")
    if type(text) == "string" and text ~= "" then
        return text
    end
    return nil
end

-- Whether an EndTurnBlockingTypes value means "something blocks".
local function isBlocking(blocking)
    if blocking == nil or blocking == 0 then
        return false
    end
    return not (EndTurnBlockingTypes and blocking == EndTurnBlockingTypes.NO_ENDTURN_BLOCKING)
end

-- The reason the player's turn cannot end, if any.
-- Returns the name of the EndTurnBlockingTypes value (for example
-- "ENDTURN_BLOCKING_RESEARCH") and the blocking notification's message, or nil
-- when nothing blocks. Asks the game the way its own action panel does
-- (GetFirstEndTurnBlocking); where that call is missing or fails, scans the
-- notification list instead.
-- The third value is true when the game could not be asked at all (no
-- NotificationManager, or neither way of asking worked). "Could not ask" is
-- not "nothing blocks": an end of turn sent while a blocker is active
-- deactivates the turn (LF-H1), so the caller has to refuse.
AlphaZeroStateExporter.UNKNOWN_BLOCKER = "ENDTURN_BLOCKING_NOT_ASKABLE"

function AlphaZeroStateExporter.GetEndTurnBlocker(playerId)
    if not NotificationManager then
        return nil, nil, true
    end

    if NotificationManager.GetFirstEndTurnBlocking then
        local okFirst, blocking = pcall(NotificationManager.GetFirstEndTurnBlocking, playerId)
        -- The game answers with a number (NO_ENDTURN_BLOCKING when nothing
        -- blocks). Anything else is not an answer: the list is scanned.
        if okFirst and type(blocking) == "number" then
            if not isBlocking(blocking) then
                return nil, nil, false
            end
            local message = nil
            if NotificationManager.FindEndTurnBlocking then
                local okFind, entry = pcall(NotificationManager.FindEndTurnBlocking, blocking, playerId)
                if okFind and entry then
                    message = notificationMessage(entry)
                end
            end
            return endTurnBlockingName(blocking), message, false
        end
    end

    -- Each entry is read under the same pcall: one that cannot be read makes
    -- the whole scan a failure, not "this one does not block".
    local ok, blockerName, message = pcall(function()
        local list = NotificationManager.GetList(playerId)
        if not list then
            return nil, nil
        end
        for _, notificationId in ipairs(list) do
            local entry = NotificationManager.Find(playerId, notificationId)
            if entry and not entry:IsDismissed() then
                local blocking = entry:GetEndTurnBlocking()
                if isBlocking(blocking) then
                    return endTurnBlockingName(blocking), notificationMessage(entry)
                end
            end
        end
        return nil, nil
    end)

    if ok then
        return blockerName, message, false
    end
    return nil, nil, true
end

-- Export full game state for a player
function AlphaZeroStateExporter.ExportState(playerId)
    local startTime = os.clock and os.clock() or 0

    local state = {
        schema_version = STATE_SCHEMA_VERSION,
        turn_id = 0,
        player_id = playerId,
        current_player = "",
        map = {},
        units = {},
        cities = {},
        player = {},
        players = {},
        legal_actions = {},
        era = 0,
        game_speed = "GAMESPEED_STANDARD",
        is_game_over = false,
        winner_id = -1,
    }

    -- Get turn info
    if Game then
        state.turn_id = Game.GetCurrentGameTurn() or 0
        local eras = Game.GetEras and Game.GetEras()
        if eras and eras.GetCurrentEra then
            state.era = eras:GetCurrentEra() or 0
        end
    end

    -- Get current player civilization name
    if PlayerConfigurations then
        local config = PlayerConfigurations[playerId]
        if config then
            state.current_player = config:GetCivilizationTypeName() or ""
        end
    end

    -- Export components (each wrapped in pcall for resilience)
    local ok, result

    ok, result = pcall(exportMap, playerId)
    state.map = ok and result or {width = 0, height = 0, tiles = {}}

    ok, result = pcall(exportUnits, playerId)
    state.units = ok and result or {}

    ok, result = pcall(exportCities, playerId)
    state.cities = ok and result or {}

    -- Who the other owners in this state are to the player (GM-B8a).
    state.observes_others = observesOthers()
    state.other_players = {}
    pcall(function()
        local owners, seen = {}, {}
        local function note(list)
            for _, item in ipairs(list) do
                local owner = item.owner
                if type(owner) == "number" and owner ~= playerId and not seen[owner] then
                    seen[owner] = true
                    table.insert(owners, owner)
                end
            end
        end
        note(state.units)
        note(state.cities)
        table.sort(owners)
        for _, owner in ipairs(owners) do
            table.insert(state.other_players, describeOtherPlayer(playerId, owner))
        end
    end)

    ok, result = pcall(exportPlayer, playerId)
    state.player = ok and result or {id = playerId}
    state.players = { state.player }

    if AlphaZeroConfig and AlphaZeroConfig.IsEnabled("LEGAL_ACTIONS") then
        ok, result = pcall(exportLegalActions, playerId)
        state.legal_actions = ok and result or {{action = "end_turn"}}
        if not ok then
            print("[AlphaZero] WARN: exportLegalActions failed: " .. tostring(result))
        end
    end

    -- What stops the turn from ending, so the agent can clear it before it
    -- asks to end the turn. Absent when nothing does.
    local blocking, blockingMessage, blockingUnknown = AlphaZeroStateExporter.GetEndTurnBlocker(playerId)
    if blockingUnknown then
        blocking = AlphaZeroStateExporter.UNKNOWN_BLOCKER
        blockingMessage = "The game could not be asked what blocks the end of turn"
    end
    state.end_turn_blocking, state.end_turn_blocking_message = blocking, blockingMessage

    -- Check game end
    if Game then
        -- Game end detection would go here
        -- state.is_game_over = Game.IsGameOver()
        -- state.winner_id = Game.GetWinner()
    end

    local elapsed = (os.clock and os.clock() or 0) - startTime
    if Log then
        Log.Info("State exported in %.3fs: turn=%d, tiles=%d, units=%d, cities=%d",
            elapsed, state.turn_id, #state.map.tiles, #state.units, #state.cities)
    end

    return state
end

-- Export and write state to file (with pcall safety net)
function AlphaZeroStateExporter.ExportAndWrite(playerId)
    local ok, state = pcall(AlphaZeroStateExporter.ExportState, playerId)
    if not ok then
        print("[AlphaZero] ERROR: ExportState crashed: " .. tostring(state))
        return false, tostring(state)
    end
    local writeOk, writeErr = AlphaZeroFileIO.WriteState(state)
    if not writeOk then
        print("[AlphaZero] ERROR: WriteState failed: " .. tostring(writeErr))
    end
    return writeOk, writeErr
end

-- Initialize on load
AlphaZeroStateExporter.Init()

print("[AlphaZero] StateExporter module loaded")

-- ========== END StateExporter.lua ==========

-- ========== BEGIN GameInfoExporter.lua ==========
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

-- ========== END GameInfoExporter.lua ==========

-- ========== BEGIN ActionExecutor.lua ==========
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
    -- The game will not end the turn until a pending choice is made. Not a
    -- malformed command: the agent clears the blocker and asks again.
    END_TURN_BLOCKED = "END_TURN_BLOCKED",
    -- The game's own check said the unit or city cannot start this now.
    OPERATION_REFUSED = "OPERATION_REFUSED",
    -- The move or attack would put the player at war. The mod never does that
    -- as a side effect; declaring war is the diplomacy command's business.
    WOULD_START_WAR = "WOULD_START_WAR",
    -- A diplomacy command whose setting (ALLOW_MAKE_PEACE, ALLOW_DENOUNCE) is
    -- not turned on. Nothing was sent to the game.
    DIPLOMACY_TURNED_OFF = "DIPLOMACY_TURNED_OFF",
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

-- End the current player's turn.
-- The request is refused, and nothing is sent to the game, while a
-- notification blocks ending the turn (an unanswered research, production, or
-- unit-orders choice): calling ACTION_ENDTURN while a blocker is active
-- deactivates the player's turn without advancing the turn counter (LF-H1).
-- UI.CanEndTurn only guards forced end-turn fallback and returns true even when
-- blockers exist in the live UI, so NotificationManager / GetEndTurnBlocker
-- is the sole end-turn blocking authority.
local function executeEndTurn(playerId, command)
    if Log then Log.Info("Executing: END_TURN") end

    -- When the game cannot be asked what blocks the turn, the turn is treated
    -- as blocked: nothing is sent.
    local blocker, blockerMessage, unknown = nil, nil, true
    if AlphaZeroStateExporter and AlphaZeroStateExporter.GetEndTurnBlocker then
        blocker, blockerMessage, unknown = AlphaZeroStateExporter.GetEndTurnBlocker(playerId)
    end
    if unknown and not blocker then
        blocker = "ENDTURN_BLOCKING_NOT_ASKABLE"
        blockerMessage = "the game could not be asked what blocks the end of turn"
    end

    -- ACTION_ENDTURN ends the LOCAL player's turn, whoever was asked about
    -- above. If the bridge's player is someone else, the blocker check was
    -- made for the wrong player: refuse.
    local okLocal, localId = pcall(function() return Game.GetLocalPlayer() end)
    if okLocal and type(localId) == "number" and localId >= 0 and localId ~= playerId then
        local message = string.format(
            "End turn refused: the bridge plays player %s and the local player is %s",
            tostring(playerId), tostring(localId))
        if Log then Log.Warn(message) end
        return false, ERROR_CODES.EXECUTION_FAILED, message
    end

    if blocker then
        local message = "End turn blocked: " .. tostring(blocker)
        if blockerMessage then
            message = message .. " (" .. tostring(blockerMessage) .. ")"
        end
        if Log then Log.Warn(message) end
        return false, ERROR_CODES.END_TURN_BLOCKED, message
    end

    -- In Civ6, ending turn is handled through the UI
    if UI and UI.RequestAction and ActionTypes then
        UI.RequestAction(ActionTypes.ACTION_ENDTURN)
        return true, nil, "End turn requested"
    end

    return false, ERROR_CODES.EXECUTION_FAILED,
        "End turn unavailable: no UI.RequestAction in this Lua context"
end

-- ----------------------------------------------------------------------------
-- Unit actions, requested the way the game's own UI requests them
-- ----------------------------------------------------------------------------
-- The game's unit panel does not hard-code operations. It walks
-- GameInfo.UnitOperations and GameInfo.UnitCommands, uses each row's Hash,
-- asks the game whether the unit can start it, and only then requests it.
-- Every unit action here goes the same way, by the row's type name, so a name
-- this version of the game does not have answers with a refusal that names it
-- instead of a request the game ignores. Not verified in a real game: the
-- names for skip and sleep are the least certain.

-- True while AlphaZeroActionExecutor.Check runs: every check is made and
-- nothing is requested, so "what the exporter offers" and "what the executor
-- carries out" are one piece of code and cannot drift apart.
local dryRun = false

-- The Hash and kind ("operation" / "command") of the row named `typeName`.
-- Found rows are remembered: the game's tables do not change while it runs,
-- and the exporter asks for the same few names for every unit and tile.
local actionRows = {}
local function findActionRow(typeName)
    local known = actionRows[typeName]
    if known then
        return known.hash, known.kind
    end
    local sources = {
        { table = "UnitOperations", column = "OperationType", kind = "operation" },
        { table = "UnitCommands", column = "CommandType", kind = "command" },
    }
    for _, source in ipairs(sources) do
        local gameTable = GameInfo and GameInfo[source.table]
        if gameTable then
            local found = nil
            pcall(function()
                for row in gameTable() do
                    if row[source.column] == typeName then
                        found = row
                        break
                    end
                end
            end)
            if found and found.Hash ~= nil then
                actionRows[typeName] = { hash = found.Hash, kind = source.kind }
                return found.Hash, source.kind
            end
        end
    end
    return nil, nil
end

-- The game's reasons for a refusal, as one line, or nil.
local function failureReasons(results, key)
    if type(results) ~= "table" or key == nil then return nil end
    local reasons = results[key]
    if type(reasons) ~= "table" or #reasons == 0 then return nil end
    local parts = {}
    for _, reason in ipairs(reasons) do
        -- The game gives text keys; show the text where it can be looked up.
        local text = tostring(reason)
        if Locale and Locale.Lookup then
            local ok, shown = pcall(Locale.Lookup, reason)
            if ok and type(shown) == "string" and shown ~= "" then text = shown end
        end
        table.insert(parts, text)
    end
    return table.concat(parts, "; ")
end

-- Ask the game's strict check and, if it agrees, request the action.
-- `params` is nil for an action without a target. Returns true, or false with
-- an error code and message. A request is fire-and-forget: true means the
-- game was asked, not that anything has happened yet.
local function requestUnitAction(unit, typeName, params, label)
    if not UnitManager then
        return false, ERROR_CODES.EXECUTION_FAILED,
            label .. " unavailable: no UnitManager in this Lua context"
    end
    local hash, kind = findActionRow(typeName)
    if hash == nil then
        return false, ERROR_CODES.EXECUTION_FAILED,
            string.format("%s: the game has no operation or command named %s", label, typeName)
    end

    local canStart, request, reasonsKey
    if kind == "operation" then
        canStart, request = UnitManager.CanStartOperation, UnitManager.RequestOperation
        reasonsKey = UnitOperationResults and UnitOperationResults.FAILURE_REASONS
    else
        canStart, request = UnitManager.CanStartCommand, UnitManager.RequestCommand
        -- The game's unit panel reads a command's reasons under the
        -- operation results key.
        reasonsKey = (UnitOperationResults and UnitOperationResults.FAILURE_REASONS)
            or (UnitCommandResults and UnitCommandResults.FAILURE_REASONS)
    end
    if not canStart or not request then
        return false, ERROR_CODES.EXECUTION_FAILED,
            string.format("%s unavailable: UnitManager cannot check or request a unit %s", label, kind)
    end

    -- The strict form: with a target, the parameters; without one, `false`
    -- ("for real", where `true` is the panel's loose "could it ever").
    local ok, allowed, results
    if kind == "command" then
        ok, allowed, results = pcall(canStart, unit, hash, false, true)
    elseif params then
        -- `true` after the parameters asks for results (the reasons).
        ok, allowed, results = pcall(canStart, unit, hash, nil, params, true)
    else
        ok, allowed, results = pcall(canStart, unit, hash, nil, false,
            OperationResultsTypes and OperationResultsTypes.NO_TARGETS)
    end
    if not ok then
        return false, ERROR_CODES.EXECUTION_FAILED,
            label .. ": the game's check raised: " .. tostring(allowed)
    end
    if allowed ~= true then
        local reason = failureReasons(results, reasonsKey)
        return false, ERROR_CODES.OPERATION_REFUSED,
            label .. " refused by the game" .. (reason and (": " .. reason) or "")
    end

    if dryRun then
        return true, nil, nil
    end
    local requested, err
    if params then
        requested, err = pcall(request, unit, hash, params)
    else
        requested, err = pcall(request, unit, hash)
    end
    if not requested then
        return false, ERROR_CODES.EXECUTION_FAILED, label .. ": the request raised: " .. tostring(err)
    end
    return true, nil, nil
end

-- Whether the player can see the plot. True when the game cannot say.
local function plotIsVisible(playerId, x, y)
    local visible = true
    pcall(function()
        local visibility = PlayersVisibility and PlayersVisibility[playerId]
        if visibility and visibility.IsVisible then
            visible = visibility:IsVisible(x, y) == true
        end
    end)
    return visible
end

-- Refuse anything that would put the player at war: nil when it would not,
-- else an error code and message. When the game cannot be asked, the answer
-- is a refusal too: the mod does not attack on a guess.
local warAnswerReported = false

local function warRefusal(unit, x, y)
    if not CombatManager or not CombatManager.IsAttackChangeWarState then
        return ERROR_CODES.EXECUTION_FAILED,
            "Cannot ask the game whether this would start a war (no CombatManager)"
    end
    local ok, results = pcall(function()
        return CombatManager.IsAttackChangeWarState(unit:GetComponentID(), x, y)
    end)
    if not ok then
        return ERROR_CODES.EXECUTION_FAILED,
            "Asking the game whether this would start a war raised: " .. tostring(results)
    end
    if results == nil then
        return nil, nil
    end
    if type(results) ~= "table" then
        -- The game's UI reads a list of players. Anything else is an answer
        -- this code does not understand, and "not understood" is not "no war".
        -- Said once per session with print, because the offer check runs with
        -- the log off: otherwise an agent that is offered no move says nothing.
        if not warAnswerReported then
            warAnswerReported = true
            print("[AlphaZero] WARN: IsAttackChangeWarState answered " .. type(results)
                .. " (" .. tostring(results) .. "); moves and attacks are refused until this is understood")
        end
        return ERROR_CODES.EXECUTION_FAILED,
            "The game's answer on whether this would start a war could not be read: "
            .. tostring(results)
    end
    local _, first = next(results)
    if first ~= nil then
        return ERROR_CODES.WOULD_START_WAR,
            string.format("Refused: (%d,%d) would start a war with player %s", x, y,
                tostring(results[1] or first))
    end
    return nil, nil
end

local function targetParams(target, modifierName)
    local params = {}
    params[UnitOperationTypes.PARAM_X] = target.x
    params[UnitOperationTypes.PARAM_Y] = target.y
    if modifierName and UnitOperationTypes.PARAM_MODIFIERS and UnitOperationMoveModifiers then
        params[UnitOperationTypes.PARAM_MODIFIERS] = UnitOperationMoveModifiers[modifierName]
    end
    return params
end

-- Whether the player sees something of another player's on the plot: a unit
-- the game says this player sees (its tile and the unit itself, so not a
-- stealth unit), or a city on a tile seen now. An occupant the player cannot
-- see is no target: offering an attack on it would give it away. When the
-- game cannot be asked about visibility, nothing is seen.
local function hasForeignOccupant(playerId, plot, x, y)
    local visibility = nil
    pcall(function() visibility = PlayersVisibility and PlayersVisibility[playerId] end)
    local plotSeen = false
    pcall(function() plotSeen = visibility:IsVisible(x, y) == true end)
    if not plotSeen then return false end

    local function seen(unit)
        local ok, visible = pcall(function() return visibility:IsUnitVisible(unit) end)
        return ok and visible == true
    end

    local found = false
    pcall(function()
        local units = Units and Units.GetUnitsInPlot and Units.GetUnitsInPlot(plot)
        if type(units) ~= "table" then return end
        if units.Members then
            for _, unit in units:Members() do
                if unit:GetOwner() ~= playerId and seen(unit) then found = true end
            end
        else
            for _, unit in ipairs(units) do
                if unit:GetOwner() ~= playerId and seen(unit) then found = true end
            end
        end
    end)
    if found then return true end
    pcall(function()
        local city = Cities and Cities.GetCityInPlot and Cities.GetCityInPlot(x, y)
        if city and city:GetOwner() ~= playerId then found = true end
    end)
    return found
end

-- Move a unit. The request carries no attack modifier, so a move onto another
-- player's unit is refused by the game's check and can never become an attack.
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

    local movesRemaining = unit:GetMovementMovesRemaining() or 0
    if movesRemaining <= 0 then
        return false, ERROR_CODES.NOT_ENOUGH_MOVEMENT, "Unit has no movement remaining"
    end

    local plot = Map and Map.GetPlot(target.x, target.y)
    if not plot then
        return false, ERROR_CODES.INVALID_TARGET, "Invalid target position"
    end
    if plot:IsImpassable() then
        return false, ERROR_CODES.INVALID_TARGET, "Target position is impassable"
    end
    if not UnitOperationTypes then
        return false, ERROR_CODES.EXECUTION_FAILED, "Move unavailable: no UnitOperationTypes in this Lua context"
    end

    -- As the game's own move routine does, ask about war only for a plot the
    -- player can see: a move into the fog is a move toward it, not onto it.
    -- A refusal must not describe something the player cannot see: when
    -- nothing of anybody else's is in sight on the plot, the answer says only
    -- that the move was refused, not that the tile is held or by whom.
    local unseen = not hasForeignOccupant(playerId, plot, target.x, target.y)
    local function refusedBlind()
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("Move to (%d,%d) refused", target.x, target.y)
    end
    if plotIsVisible(playerId, target.x, target.y) then
        local warCode, warMessage = warRefusal(unit, target.x, target.y)
        if warCode then
            if unseen and warCode == ERROR_CODES.WOULD_START_WAR then return refusedBlind() end
            return false, warCode, warMessage
        end
    end

    local ok, code, message = requestUnitAction(
        unit, "UNITOPERATION_MOVE_TO", targetParams(target, "NONE"), "Move")
    if not ok then
        if unseen and code == ERROR_CODES.OPERATION_REFUSED then return refusedBlind() end
        return false, code, message
    end
    return true, nil, string.format("Move of unit %d to (%d,%d) requested", unitId, target.x, target.y)
end

-- Shared by the melee and the ranged attack.
local function executeAttack(playerId, command, typeName, modifierName, label)
    local unitId = command.unit_id
    local target = command.target

    if not unitId or not target or not target.x or not target.y then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id or target"
    end

    if Log then
        Log.Info("Executing: %s unit=%d at (%d,%d)", label, unitId, target.x, target.y)
    end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    if (unit:GetAttacksRemaining() or 0) <= 0 then
        return false, ERROR_CODES.NOT_ENOUGH_MOVEMENT, "Unit has no attacks remaining"
    end

    local plot = Map and Map.GetPlot(target.x, target.y)
    if not plot then
        return false, ERROR_CODES.INVALID_TARGET, "Invalid target position"
    end
    if not hasForeignOccupant(playerId, plot, target.x, target.y) then
        return false, ERROR_CODES.INVALID_TARGET,
            string.format("Nothing to attack at (%d,%d)", target.x, target.y)
    end
    if not UnitOperationTypes then
        return false, ERROR_CODES.EXECUTION_FAILED,
            label .. " unavailable: no UnitOperationTypes in this Lua context"
    end

    local warCode, warMessage = warRefusal(unit, target.x, target.y)
    if warCode then
        return false, warCode, warMessage
    end

    local ok, code, message = requestUnitAction(unit, typeName, targetParams(target, modifierName), label)
    if not ok then
        return false, code, message
    end
    return true, nil, string.format("%s at (%d,%d) requested", label, target.x, target.y)
end

-- Melee attack: the move operation with the attack modifier, which is how the
-- game's own move routine engages a unit on the destination tile.
local function executeUnitAttack(playerId, command)
    return executeAttack(playerId, command, "UNITOPERATION_MOVE_TO", "ATTACK", "Attack")
end

-- Ranged attack: its own operation (archers, catapults, bombards).
local function executeUnitRangedAttack(playerId, command)
    return executeAttack(playerId, command, "UNITOPERATION_RANGE_ATTACK", nil, "Ranged attack")
end

-- A unit action without a target.
local function executeSimpleUnitAction(playerId, command, typeName, label)
    local unitId = command.unit_id
    if not unitId then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing unit_id"
    end

    if Log then Log.Info("Executing: %s unit=%d", label, unitId) end

    local unit, errCode, errMsg = safeGetUnit(playerId, unitId)
    if not unit then
        return false, errCode, errMsg
    end

    local ok, code, message = requestUnitAction(unit, typeName, nil, label)
    if not ok then
        return false, code, message
    end
    return true, nil, string.format("%s of unit %d requested", label, unitId)
end

local function executeUnitFortify(playerId, command)
    return executeSimpleUnitAction(playerId, command, "UNITOPERATION_FORTIFY", "Fortify")
end

local function executeUnitSkip(playerId, command)
    return executeSimpleUnitAction(playerId, command, "UNITOPERATION_SKIP_TURN", "Skip")
end

local function executeUnitSleep(playerId, command)
    return executeSimpleUnitAction(playerId, command, "UNITOPERATION_SLEEP", "Sleep")
end

local function executeFoundCity(playerId, command)
    return executeSimpleUnitAction(playerId, command, "UNITOPERATION_FOUND_CITY", "Found city")
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

-- Production. The game's own production panel sends the row's Hash under the
-- parameter of its kind, with the exclusive insert mode (replace, do not
-- queue). Districts and wonders also need a plot, which this command does not
-- carry, so they are refused.
local PRODUCTION_KINDS = {
    unit = { table = "Units", column = "UnitType", param = "PARAM_UNIT_TYPE" },
    building = { table = "Buildings", column = "BuildingType", param = "PARAM_BUILDING_TYPE" },
    district = { table = "Districts", column = "DistrictType", param = "PARAM_DISTRICT_TYPE", placed = true },
}
local PRODUCTION_ORDER = { "unit", "building", "district" }

local function findProductionRow(kind, matches)
    local gameTable = GameInfo and GameInfo[kind.table]
    if not gameTable then return nil end
    local found = nil
    pcall(function()
        for row in gameTable() do
            if matches(row) then
                found = row
                break
            end
        end
    end)
    return found
end

-- The row and kind for an item given by type name (looked up in every kind)
-- or by row index (in the kind `itemType` names; units when it names none).
-- Found rows are remembered (see findActionRow): without this, listing what
-- a city can build walked the unit table once per candidate item.
local productionRows = {}
local function resolveProductionItem(item, itemType)
    if type(item) == "string" then
        local known = productionRows[item]
        if known then return known.row, known.kind end
        for _, name in ipairs(PRODUCTION_ORDER) do
            local kind = PRODUCTION_KINDS[name]
            local row = findProductionRow(kind, function(r) return r[kind.column] == item end)
            if row then
                productionRows[item] = { row = row, kind = kind }
                return row, kind
            end
        end
        return nil, nil
    end
    local kind = PRODUCTION_KINDS[itemType or "unit"]
    if not kind then return nil, nil end
    local row = findProductionRow(kind, function(r) return r.Index == item end)
    return row, kind
end

local function executeCityProduce(playerId, command)
    local cityId = command.city_id
    local itemId = command.item_id
    local itemType = command.item_type

    if not cityId or itemId == nil then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing city_id or item_id"
    end

    if Log then
        Log.Info("Executing: CITY_PRODUCE city=%d item=%s type=%s", cityId, tostring(itemId), tostring(itemType))
    end

    local city, errCode, errMsg = safeGetCity(playerId, cityId)
    if not city then
        return false, errCode, errMsg
    end

    if itemType == "wonder" then
        return false, ERROR_CODES.NOT_IMPLEMENTED,
            "A wonder needs a plot; placement is not supported: " .. tostring(itemId)
    end

    local row, kind = resolveProductionItem(itemId, itemType)
    if not row then
        return false, ERROR_CODES.INVALID_COMMAND, "Unknown production item: " .. tostring(itemId)
    end
    local typeName = tostring(row[kind.column])
    -- A district always needs a plot; a building does when its row says so
    -- (wonders), as the game's production panel checks before it builds one.
    if kind.placed or row.RequiresPlacement == true then
        return false, ERROR_CODES.NOT_IMPLEMENTED,
            typeName .. " needs a plot; placement is not supported"
    end
    -- The game's production panel never offers to build these.
    if row.MustPurchase == true then
        return false, ERROR_CODES.OPERATION_REFUSED,
            typeName .. " can only be purchased, not built"
    end
    if row.Hash == nil then
        return false, ERROR_CODES.EXECUTION_FAILED, typeName .. " has no hash in this Lua context"
    end

    if not CityManager or not CityManager.RequestOperation or not CityOperationTypes then
        return false, ERROR_CODES.EXECUTION_FAILED,
            "Production unavailable: no CityManager in this Lua context"
    end

    -- Ask both CanProduce(hash, true) (unlock/prereq eligibility) and
    -- CanProduce(hash, false, true) (exclusion/start-now check), matching Civ6's
    -- ProductionPanel.lua. Checking only (hash, false, true) returns true for
    -- unresearched buildings (e.g. BUILDING_SEWER) and clears the city's queue.
    local eligible = nil
    local allowed = nil
    pcall(function()
        local buildQueue = city:GetBuildQueue()
        if buildQueue and buildQueue.CanProduce then
            eligible = buildQueue:CanProduce(row.Hash, true)
            if eligible == true then
                allowed = buildQueue:CanProduce(row.Hash, false, true)
            end
        end
    end)
    -- Anything but a clear yes from both checks is a refusal, a game that
    -- cannot be asked included: a request without it would be acknowledged as
    -- a success, and these checks also decide what the exporter offers.
    local couldNotAsk = (eligible == nil) or (eligible == true and allowed == nil)
    if eligible ~= true or allowed ~= true then
        if Log and Log.Debug then
            Log.Debug(string.format(
                "City %d CanProduce(%s): eligible=%s allowed=%s",
                cityId, tostring(typeName), tostring(eligible), tostring(allowed)
            ))
        end
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("City %d cannot start %s now%s", cityId, typeName,
                couldNotAsk and " (the game could not be asked)" or "")
    end

    local params = {}
    params[CityOperationTypes[kind.param]] = row.Hash
    params[CityOperationTypes.PARAM_INSERT_MODE] = CityOperationTypes.VALUE_EXCLUSIVE
    if not dryRun then
        local requested, err = pcall(CityManager.RequestOperation, city, CityOperationTypes.BUILD, params)
        if not requested then
            return false, ERROR_CODES.EXECUTION_FAILED, "Production request raised: " .. tostring(err)
        end
    end
    return true, nil, string.format("City %d: production of %s requested", cityId, typeName)
end

-- ----------------------------------------------------------------------------
-- Research (technologies and civics)
-- ----------------------------------------------------------------------------

-- Resolve a research item to its GameInfo row. The item is a row index (what
-- the Python action space sends) or a type name such as "TECH_MINING".
-- Returns index, type name and hash; all nil when the GameInfo table is there
-- and has no such row. Without the table an index is passed through unchecked.
local function resolveResearchItem(tableName, typeField, item)
    local gameTable = GameInfo and GameInfo[tableName]
    if not gameTable then
        if type(item) == "number" then
            return item, tostring(item), nil
        end
        return nil, nil, nil
    end

    local index, typeName, hash = nil, nil, nil
    pcall(function()
        for row in gameTable() do
            if row.Index == item or row[typeField] == item then
                index, typeName, hash = row.Index, row[typeField], row.Hash
                break
            end
        end
    end)
    return index, typeName, hash
end

-- Call a method on a game object; nil when it is missing or raises.
local function callIfPresent(object, methodName, ...)
    if object == nil then return nil end
    local okMethod, method = pcall(function() return object[methodName] end)
    if not okMethod or not method then return nil end
    local okCall, result = pcall(method, object, ...)
    if okCall then return result end
    return nil
end

-- Shared by technologies and civics. `kind` names the differences:
--   table/typeField   GameInfo table and its type-name column
--   getter            player method returning the progress object
--   has/can           its "already has" and "can choose now" methods (by index)
--   operation/param   PlayerOperations keys for the UI request
--   setter            GameCore method on the progress object (fallback)
local function executeResearch(playerId, command, kind)
    local item = command.item_id
    if item == nil then
        return false, ERROR_CODES.INVALID_COMMAND, "Missing item_id (" .. kind.label .. ")"
    end

    if Log then Log.Info("Executing: %s item=%s", kind.logName, tostring(item)) end

    if not Players then
        return false, ERROR_CODES.EXECUTION_FAILED, "Players table not available"
    end
    local player = Players[playerId]
    if not player then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player not found"
    end

    local index, typeName, hash = resolveResearchItem(kind.table, kind.typeField, item)
    if index == nil then
        return false, ERROR_CODES.INVALID_TARGET,
            "Unknown " .. kind.label .. ": " .. tostring(item)
    end

    -- Refuse what the game would drop without a word. Each check is skipped
    -- when its method is missing in this Lua context.
    local progressObject = callIfPresent(player, kind.getter)
    if callIfPresent(progressObject, kind.has, index) == true then
        return false, ERROR_CODES.INVALID_TARGET,
            tostring(typeName) .. " is already researched"
    end
    if callIfPresent(progressObject, kind.can, index) == false then
        return false, ERROR_CODES.INVALID_TARGET,
            tostring(typeName) .. " cannot be chosen now"
    end

    -- Primary path: ask the game as its own research and civics choosers do:
    -- the type parameter is the row's Hash (not its Index), with an exclusive
    -- insert mode so the choice replaces the queue. The request is
    -- fire-and-forget: the game applies it after this call returns and drops
    -- it without an error if it does not like it, so success here means
    -- "requested", not "done". The next exported state shows what the game
    -- did. No notification is touched: the game clears its own "choose
    -- research" blocker when the choice takes effect.
    local requested = false
    if hash ~= nil and UI and UI.RequestPlayerOperation and PlayerOperations
       and PlayerOperations[kind.operation] and PlayerOperations[kind.param] then
        local params = {}
        params[PlayerOperations[kind.param]] = hash
        if PlayerOperations.PARAM_INSERT_MODE and PlayerOperations.VALUE_EXCLUSIVE then
            params[PlayerOperations.PARAM_INSERT_MODE] = PlayerOperations.VALUE_EXCLUSIVE
        end
        requested = pcall(function()
            UI.RequestPlayerOperation(playerId, PlayerOperations[kind.operation], params)
        end)
    end
    if requested then
        return true, nil, kind.requestedMessage .. tostring(typeName)
    end

    -- Fallback: the GameCore setter, which takes the row index and exists
    -- only in gameplay contexts.
    if not progressObject then
        return false, ERROR_CODES.EXECUTION_FAILED, "Player " .. kind.label .. " data not available"
    end
    local okSetter, setter = pcall(function() return progressObject[kind.setter] end)
    if not okSetter or not setter then
        return false, ERROR_CODES.EXECUTION_FAILED,
            "No way to set the " .. kind.label .. " in this Lua context"
    end
    local okSet, setErr = pcall(setter, progressObject, index)
    if not okSet then
        return false, ERROR_CODES.EXECUTION_FAILED,
            kind.setter .. " failed: " .. tostring(setErr)
    end
    return true, nil, kind.setMessage .. tostring(typeName)
end

local RESEARCH_TECH_KIND = {
    label = "tech",
    logName = "RESEARCH_TECH",
    table = "Technologies",
    typeField = "TechnologyType",
    getter = "GetTechs",
    has = "HasTech",
    can = "CanResearch",
    operation = "RESEARCH",
    param = "PARAM_TECH_TYPE",
    setter = "SetResearchingTech",
    requestedMessage = "Research requested: ",
    setMessage = "Researching tech: ",
}

local RESEARCH_CIVIC_KIND = {
    label = "civic",
    logName = "RESEARCH_CIVIC",
    table = "Civics",
    typeField = "CivicType",
    getter = "GetCulture",
    has = "HasCivic",
    can = "CanProgress",
    operation = "PROGRESS_CIVIC",
    param = "PARAM_CIVIC_TYPE",
    setter = "SetProgressingCivic",
    requestedMessage = "Civic requested: ",
    setMessage = "Progressing civic: ",
}

-- Research a technology
local function executeResearchTech(playerId, command)
    return executeResearch(playerId, command, RESEARCH_TECH_KIND)
end

-- Research a civic
local function executeResearchCivic(playerId, command)
    return executeResearch(playerId, command, RESEARCH_CIVIC_KIND)
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

    -- The error of the last path that was tried and raised, so a failure says
    -- why instead of claiming there was nothing to call.
    local lastError = nil

    -- Primary path: GetDiplomacy():DoAction(...). Civ6 expects a string or
    -- int action constant from DiplomaticActions table; pass civ6ActionName.
    if diplomacy.DoAction then
        local ok, err = pcall(function() diplomacy:DoAction(civ6ActionName, targetPlayerId) end)
        if ok then
            return true, nil, string.format("%s against player %d", subActionName, targetPlayerId)
        end
        lastError = err
        if Log then Log.Warn("DoAction failed for %s: %s", subActionName, tostring(err)) end
    end

    -- Fallback: UI.RequestDiplomacyMove for older mod contexts.
    if UI and UI.RequestDiplomacyMove then
        local ok, err = pcall(function() UI.RequestDiplomacyMove(civ6ActionName, targetPlayerId) end)
        if ok then
            return true, nil, string.format("%s requested via UI (player %d)",
                subActionName, targetPlayerId)
        end
        lastError = err
        if Log then Log.Warn("RequestDiplomacyMove failed for %s: %s", subActionName, tostring(err)) end
    end

    if lastError then
        return false, ERROR_CODES.EXECUTION_FAILED,
            string.format("Diplomacy action %s failed: %s", subActionName, tostring(lastError))
    end

    return false, ERROR_CODES.EXECUTION_FAILED,
        string.format("No diplomacy API available for %s", subActionName)
end

-- True only when the setting is the boolean true: a string "true", a 1 or an
-- unset setting leaves the command turned off.
local function _diplomacySettingOn(name)
    return AlphaZeroConfig ~= nil and AlphaZeroConfig.Get(name) == true
end

-- Ask the acting player's diplomacy object one yes/no question about the
-- target. Fails closed: where the game cannot be asked (no diplomacy object,
-- no such method, the call raises, the answer is not a boolean) the answer is
-- (nil, message) and the caller refuses the command.
local function _askDiplomacy(playerId, method, targetPlayerId)
    local function cannot(why)
        return nil, string.format("Cannot ask the game %s(%s): %s",
            method, tostring(targetPlayerId), why)
    end
    local player = Players and Players[playerId]
    if not player or type(player.GetDiplomacy) ~= "function" then
        return cannot("no diplomacy for player " .. tostring(playerId))
    end
    local ok, diplomacy = pcall(player.GetDiplomacy, player)
    if not ok then return cannot("GetDiplomacy raised: " .. tostring(diplomacy)) end
    if type(diplomacy) ~= "table" and type(diplomacy) ~= "userdata" then
        return cannot("no diplomacy object")
    end
    local okMethod, fn = pcall(function() return diplomacy[method] end)
    if not okMethod or type(fn) ~= "function" then
        return cannot("the method is not available")
    end
    local okCall, answer = pcall(fn, diplomacy, targetPlayerId)
    if not okCall then return cannot("the call raised: " .. tostring(answer)) end
    if type(answer) ~= "boolean" then
        return cannot("the answer was not a boolean (" .. tostring(answer) .. ")")
    end
    return answer, nil
end

-- Peace and denouncing are for major civilizations only (not barbarians,
-- city-states or free cities). Asks Players[target]:IsMajor() fail-closed;
-- returns nil when the target may be acted on, else (code, message).
local function _refuseUnlessMajor(targetPlayerId)
    local function cannot(why)
        return ERROR_CODES.EXECUTION_FAILED, string.format(
            "Cannot ask the game IsMajor(%s): %s", tostring(targetPlayerId), why)
    end
    local target = Players and Players[targetPlayerId]
    if target == nil then return cannot("no such player") end
    local okMethod, fn = pcall(function() return target.IsMajor end)
    if not okMethod or type(fn) ~= "function" then
        return cannot("the method is not available")
    end
    local okCall, answer = pcall(fn, target)
    if not okCall then return cannot("the call raised: " .. tostring(answer)) end
    if type(answer) ~= "boolean" then
        return cannot("the answer was not a boolean (" .. tostring(answer) .. ")")
    end
    if not answer then
        return ERROR_CODES.OPERATION_REFUSED, string.format(
            "Refused: player %s is not a major civilization", tostring(targetPlayerId))
    end
    return nil
end

local function diplomacyDeclareWar(playerId, targetPlayerId)
    -- Refused unless the setting allows it: nothing in this mod may start a
    -- war by default (the same answer a move or an attack that would gets).
    if not _diplomacySettingOn("ALLOW_DECLARE_WAR") then
        return false, ERROR_CODES.WOULD_START_WAR,
            string.format("Refused: declaring war on player %s is turned off (ALLOW_DECLARE_WAR)",
                tostring(targetPlayerId))
    end
    return _diplomacyRequest(playerId, targetPlayerId, "DIPLOACTION_DECLARE_WAR", "DECLARE_WAR")
end

local function diplomacyMakePeace(playerId, targetPlayerId)
    -- Refused unless the setting allows it, and then only with a major
    -- civilization the game itself reports the pair at war with.
    if not _diplomacySettingOn("ALLOW_MAKE_PEACE") then
        return false, ERROR_CODES.DIPLOMACY_TURNED_OFF,
            string.format("Refused: making peace with player %s is turned off (ALLOW_MAKE_PEACE)",
                tostring(targetPlayerId))
    end
    local majorCode, majorWhy = _refuseUnlessMajor(targetPlayerId)
    if majorCode then return false, majorCode, majorWhy end
    local atWar, why = _askDiplomacy(playerId, "IsAtWarWith", targetPlayerId)
    if atWar == nil then return false, ERROR_CODES.EXECUTION_FAILED, why end
    if not atWar then
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("Refused: not at war with player %s, so there is no peace to make",
                tostring(targetPlayerId))
    end
    return _diplomacyRequest(playerId, targetPlayerId, "DIPLOACTION_MAKE_PEACE", "MAKE_PEACE")
end

local function diplomacyDenounce(playerId, targetPlayerId)
    -- Refused unless the setting allows it, and then only a major civilization
    -- the game reports as met and not at war.
    if not _diplomacySettingOn("ALLOW_DENOUNCE") then
        return false, ERROR_CODES.DIPLOMACY_TURNED_OFF,
            string.format("Refused: denouncing player %s is turned off (ALLOW_DENOUNCE)",
                tostring(targetPlayerId))
    end
    local majorCode, majorWhy = _refuseUnlessMajor(targetPlayerId)
    if majorCode then return false, majorCode, majorWhy end
    local met, whyMet = _askDiplomacy(playerId, "HasMet", targetPlayerId)
    if met == nil then return false, ERROR_CODES.EXECUTION_FAILED, whyMet end
    if not met then
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("Refused: player %s has not been met", tostring(targetPlayerId))
    end
    local atWar, whyWar = _askDiplomacy(playerId, "IsAtWarWith", targetPlayerId)
    if atWar == nil then return false, ERROR_CODES.EXECUTION_FAILED, whyWar end
    if atWar then
        return false, ERROR_CODES.OPERATION_REFUSED,
            string.format("Refused: at war with player %s; denouncing is for a player at peace",
                tostring(targetPlayerId))
    end
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

    if targetPlayerId == playerId then
        -- No diplomacy is aimed at oneself, whatever the sub-action (declaring
        -- war and the deferred ones included); checked first, nothing is sent.
        return false, ERROR_CODES.INVALID_TARGET,
            string.format("DIPLOMACY target player %s is the acting player", tostring(targetPlayerId))
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

-- The commands Check can answer for: the ones whose handlers ask the game
-- before they request.
local CHECKABLE = {
    [ACTION_TYPES.UNIT_MOVE] = true, [ACTION_TYPES.UNIT_ATTACK] = true,
    [ACTION_TYPES.UNIT_RANGED_ATTACK] = true, [ACTION_TYPES.UNIT_SKIP] = true,
    [ACTION_TYPES.UNIT_FORTIFY] = true, [ACTION_TYPES.UNIT_SLEEP] = true,
    [ACTION_TYPES.UNIT_FOUND_CITY] = true, [ACTION_TYPES.CITY_PRODUCE] = true,
}

-- What Execute would answer for this command, without requesting anything.
-- The exporter offers an action exactly when this says yes.
function AlphaZeroActionExecutor.Check(playerId, command)
    if not command or not command.action_type then
        return AlphaZeroActionExecutor.Execute(playerId, command)  -- refused there, nothing runs
    end
    if not CHECKABLE[command.action_type] then
        return false, ERROR_CODES.NOT_IMPLEMENTED,
            "No check for action type: " .. tostring(command and command.action_type)
    end
    local savedLog, savedDryRun = Log, dryRun
    dryRun, Log = true, nil  -- a check per candidate would flood the log
    local ok, success, errCode, message = pcall(AlphaZeroActionExecutor.Execute, playerId, command)
    dryRun, Log = savedDryRun, savedLog  -- a check inside a check stays a check
    if not ok then
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

-- ========== END ActionExecutor.lua ==========

-- ========== BEGIN Main.lua ==========
-- ============================================================================
-- Main.lua - Entry point for AlphaZero AI Bridge
-- ============================================================================
-- Coordinates all modules and handles game events to provide AI control.
-- Fixes: deduplicated events, proper polling handle, graceful shutdown, timeout.
-- ============================================================================

-- Load dependent modules via include() (required in AddUserInterfaces context).
-- Guarded so re-including this file (or loading alongside AlphaZeroMain.lua's
-- bundled copies) doesn't redefine each module's globals unnecessarily.
if not AlphaZeroConfig then include("Config") end
if not AlphaZeroLogger then include("Logger") end
if not AlphaZeroJson then include("JsonEncoder") end
if not AlphaZeroFileIO then include("FileIO") end
if not AlphaZeroStateExporter then include("StateExporter") end
if not AlphaZeroActionExecutor then include("ActionExecutor") end

AlphaZeroBridge = AlphaZeroBridge or {}


local Log = nil  -- Will be set after Logger initializes

-- Bridge state
local bridgeState = {
    initialized = false,
    aiPlayerId = nil,
    isMyTurn = false,
    waitingForCommand = false,
    lastHeartbeatTime = 0,
    actionsThisTurn = 0,
    lastError = nil,
    turnStartTime = 0,
    -- When the last command message was handled; the command window runs
    -- from the later of this and the turn's start.
    lastCommandTime = 0,
    -- Frames left before the state is exported after a command message; nil
    -- when no export is pending.
    exportAfterTicks = nil,
    -- An end_turn was carried out this turn: the turn is ending and a re-arm
    -- must not reopen it. Reset at the turn's start.
    turnEnding = false,
}

-- Polling state (use explicit callback reference for reliable removal)
local isPolling = false
local pollCallback = nil  -- Will be set to AlphaZeroBridge.PollCommands after definition
local frameCallback = nil -- The per-frame poll, where the context has frame updates
local hasFrames = false

-- Track registered events for graceful shutdown
local registeredEvents = {}

-- ----------------------------------------------------------------------------
-- Initialization
-- ----------------------------------------------------------------------------

-- Find the AI player (first human player we'll control)
local function findAIPlayer()
    -- Which rule decided is logged: a wrong player here makes every later
    -- check ask about the wrong player.
    local function decided(id, how)
        if Log then Log.Info("AI player %s: %s", tostring(id), how) end
        return id
    end
    local configuredId = AlphaZeroConfig and AlphaZeroConfig.Get("AI_PLAYER_ID")
    if configuredId then
        return decided(configuredId, "configured (AI_PLAYER_ID)")
    end

    -- The local player, as the game's own UI asks for it.
    local okLocal, localId = pcall(function() return Game.GetLocalPlayer() end)
    if okLocal and type(localId) == "number" and localId >= 0 then
        return decided(localId, "the game's local player")
    end

    -- Else the first living human among the players the game lists, falling
    -- back to walking the ids. A count is never used as an id: barbarians and
    -- city-states have high ids, and the human need not be among the first.
    local found = nil
    local function consider(id, player)
        if found == nil and player and player:IsHuman() and player:IsAlive() then
            found = id
        end
    end
    local listed = pcall(function()
        local players = Game.GetPlayers({ Alive = true })
        table.sort(players, function(a, b) return a:GetID() < b:GetID() end)
        for _, player in ipairs(players) do consider(player:GetID(), player) end
    end)
    if not listed then
        found = nil
        pcall(function()
            for id = 0, GameDefines.MAX_PLAYERS - 1 do consider(id, Players[id]) end
        end)
    end

    if found ~= nil then
        return decided(found, "the first living human player")
    end
    return decided(0, "fallback: no local player and no human found")
end

-- Drop a pending re-arm request (ExposedMembers.AlphaZeroRearm) without
-- acting on it, saying why. Used where the state is exported anyway (a turn
-- start) or the bridge starts over (initialisation, reset), so a request left
-- from before cannot fire later and export a second, stale state.
local function dropRearm(why)
    if not ExposedMembers or ExposedMembers.AlphaZeroRearm == nil then return end
    local nonce = ExposedMembers.AlphaZeroRearm
    ExposedMembers.AlphaZeroRearm = nil
    if Log then Log.Info("Re-arm dropped %s: %s", tostring(nonce), why) end
end

-- Whether the game says it is the bridge player's turn: the local player is
-- the bridge's player and that player's turn is active. Returns true, or false
-- and why not ("not our turn", "turn not active", "cannot ask"). Anything the
-- game cannot answer is "cannot ask", never a yes.
local function ourActiveTurn()
    local playerId = bridgeState.aiPlayerId
    if playerId == nil then return false, "cannot ask" end
    local okLocal, localId = pcall(function() return Game.GetLocalPlayer() end)
    if not okLocal or type(localId) ~= "number" then return false, "cannot ask" end
    if localId ~= playerId then return false, "not our turn" end
    local okActive, active = pcall(function()
        local player = Players[playerId]
        if not player or not player.IsTurnActive then error("no IsTurnActive") end
        return player:IsTurnActive()
    end)
    if not okActive then return false, "cannot ask" end
    if not active then return false, "turn not active" end
    return true
end

-- Initialize the bridge
function AlphaZeroBridge.Initialize()
    if bridgeState.initialized then
        return true
    end

    -- Initialize all modules
    if AlphaZeroConfig then
        AlphaZeroConfig.Init()
    end

    if AlphaZeroLogger then
        AlphaZeroLogger.Init()
        Log = AlphaZeroLogger.CreateLogger("Main")
    end

    if AlphaZeroJson then
        AlphaZeroJson.Init()
    end

    if AlphaZeroFileIO then
        AlphaZeroFileIO.Init()
    end

    if AlphaZeroStateExporter then
        AlphaZeroStateExporter.Init()
    end

    if AlphaZeroActionExecutor then
        AlphaZeroActionExecutor.Init()
    end

    -- Diagnostic: dump available APIs to Lua.log. Gated behind DEBUG log
    -- level -- this ran unconditionally on every init, spamming Lua.log and
    -- slowing the Python LogTailer. Only emit when LOG_LEVEL=DEBUG (numeric
    -- level 1; see the levels table in Config.lua's GetLogLevelNum()).
    if AlphaZeroConfig and AlphaZeroConfig.GetLogLevelNum and AlphaZeroConfig.GetLogLevelNum() <= 1 then
        local apis = {
            {"io", io},
            {"io.open", io and io.open},
            {"os", os},
            {"os.time", os and os.time},
            {"os.clock", os and os.clock},
            {"os.execute", os and os.execute},
            {"os.getenv", os and os.getenv},
            {"os.remove", os and os.remove},
            {"os.rename", os and os.rename},
            {"UI", UI},
            {"UI.GetAppSavegameDirectory", UI and UI.GetAppSavegameDirectory},
            {"UI.DataError", UI and UI.DataError},
            {"Modding", Modding},
            {"Modding.GetUserDataFolder", Modding and Modding.GetUserDataFolder},
            {"Game", Game},
            {"Players", Players},
            {"GameEvents", GameEvents},
            {"Events", Events},
            {"ContextPtr", ContextPtr},
            {"Network", Network},
            {"loadfile", loadfile},
            {"dofile", dofile},
            {"require", require},
            {"include", include},
        }
        print("[AlphaZero] === API AVAILABILITY DIAGNOSTIC ===")
        for _, entry in ipairs(apis) do
            local name, val = entry[1], entry[2]
            print(string.format("[AlphaZero] %s = %s", name, val and "AVAILABLE" or "nil"))
        end
        print("[AlphaZero] === END DIAGNOSTIC ===")
    end

    -- Validate configuration
    if AlphaZeroConfig and AlphaZeroConfig.Validate then
        local valid, errors = AlphaZeroConfig.Validate()
        if not valid then
            if Log then
                for _, err in ipairs(errors) do
                    Log.Error("Config error: %s", err)
                end
            end
            return false
        end
    end

    -- Find AI player
    bridgeState.aiPlayerId = findAIPlayer()
    if Log then Log.Info("AI Player ID: %d", bridgeState.aiPlayerId) end

    -- Clear old files, and a re-arm request from before this initialisation
    dropRearm("the bridge was initialised")
    if AlphaZeroFileIO then
        AlphaZeroFileIO.ClearAll()
        AlphaZeroFileIO.WriteHeartbeat()
    end

    -- Write the game's rule tables once per load, for the Python side to
    -- capture (GM-B4). The tables do not change during a game. Never fatal:
    -- the bridge works without the snapshot.
    if AlphaZeroGameInfo and AlphaZeroConfig and AlphaZeroConfig.IsEnabled("GAMEINFO_EXPORT") then
        local ok, written, err = pcall(AlphaZeroGameInfo.ExportAndWrite)
        if not ok or not written then
            if Log then
                Log.Warn("GameInfo snapshot not written: %s", tostring(ok and err or written))
            end
        end
    end

    bridgeState.initialized = true
    if Log then Log.Info("AlphaZero Bridge initialized successfully") end

    -- Loaded in the middle of our own active turn (a reload, or the mod added
    -- late): the turn-start event has already gone by, so start the turn here.
    if ourActiveTurn() then
        if Log then Log.Info("Loaded during our active turn: starting it") end
        AlphaZeroBridge.OnTurnStart(bridgeState.aiPlayerId)
    end

    return true
end

-- ----------------------------------------------------------------------------
-- Turn Management
-- ----------------------------------------------------------------------------

-- Called when the AI player's turn starts
function AlphaZeroBridge.OnTurnStart(playerId)
    if playerId ~= bridgeState.aiPlayerId then
        return
    end

    -- Re-entrance guard: prevent double-firing from multiple event sources
    if bridgeState.isMyTurn then
        return
    end

    if Log then Log.Info("Turn started for player %d", playerId) end
    dropRearm("the turn start exports the state")

    bridgeState.isMyTurn = true
    bridgeState.actionsThisTurn = 0
    bridgeState.waitingForCommand = true
    bridgeState.turnStartTime = (os and os.time) and os.time() or 0
    bridgeState.lastCommandTime = bridgeState.turnStartTime
    bridgeState.exportAfterTicks = nil
    bridgeState.exportAfterFrames = nil
    bridgeState.turnEnding = false

    -- Export and write current state
    local success, err = AlphaZeroStateExporter.ExportAndWrite(playerId)
    if not success then
        if Log then Log.Error("Failed to export state: %s", tostring(err)) end
        bridgeState.lastError = err
    end

    -- Start polling for commands
    AlphaZeroBridge.StartPolling()
end

-- Called when the AI player's turn ends
function AlphaZeroBridge.OnTurnEnd(playerId)
    if playerId ~= bridgeState.aiPlayerId then
        return
    end

    if Log then Log.Info("Turn ended for player %d (actions: %d)", playerId, bridgeState.actionsThisTurn) end

    bridgeState.isMyTurn = false
    bridgeState.waitingForCommand = false

    -- Stop polling
    AlphaZeroBridge.StopPolling()
end

-- ----------------------------------------------------------------------------
-- Command Polling
-- ----------------------------------------------------------------------------

-- Start polling for commands
function AlphaZeroBridge.StartPolling()
    if isPolling then
        return  -- Already polling
    end

    local started = false
    if Events and Events.GameCoreEventPublishComplete then
        -- Ensure pollCallback is set
        if not pollCallback then
            pollCallback = AlphaZeroBridge.PollCommands
        end
        Events.GameCoreEventPublishComplete.Add(pollCallback)
        started = true
    end

    -- Frames as well, where this context has them (the game's own panels use
    -- ContextPtr:SetUpdate). The event above is raised only after a batch of
    -- game events: in a game that has gone idle it never comes, so on it
    -- alone the state after a command would not be exported and no further
    -- command would be read.
    hasFrames = false
    if ContextPtr and ContextPtr.SetUpdate then
        if not frameCallback then
            frameCallback = function() AlphaZeroBridge.PollCommands("frame") end
        end
        hasFrames = pcall(function() ContextPtr:SetUpdate(frameCallback) end)
        started = started or hasFrames
    end

    if started then
        isPolling = true
    else
        if Log then Log.Warn("No timer mechanism available for polling") end
    end
end

-- Stop polling
function AlphaZeroBridge.StopPolling()
    if isPolling and Events and Events.GameCoreEventPublishComplete and pollCallback then
        Events.GameCoreEventPublishComplete.Remove(pollCallback)
    end
    if hasFrames and ContextPtr and ContextPtr.ClearUpdate then
        pcall(function() ContextPtr:ClearUpdate() end)
    end
    hasFrames = false
    isPolling = false
end

-- A re-arm request from the Python side (ExposedMembers.AlphaZeroRearm, a
-- nonce string): reopen a lapsed command window and export the state once.
-- Accepted only on our own active turn, with the window closed, no command
-- waiting to be read or for its state export, and no end_turn carried out;
-- the flag is cleared either way. It never asks the game to do anything: it
-- cannot end a turn. Each answer is logged and also printed as a machine line
-- AZ_REARM:<nonce>:accepted or AZ_REARM:<nonce>:ignored:<reason>. Returns
-- true when accepted.
local function checkRearm()
    if not ExposedMembers then return false end
    local nonce = ExposedMembers.AlphaZeroRearm
    if nonce == nil then return false end
    ExposedMembers.AlphaZeroRearm = nil

    local reason = nil
    if type(nonce) ~= "string" or nonce == "" then
        reason = "not a non-empty string"
        nonce = "-"
    elseif bridgeState.exportAfterTicks
        or (AlphaZeroFileIO and AlphaZeroFileIO.HasPendingCommand and AlphaZeroFileIO.HasPendingCommand()) then
        reason = "command in flight"
    elseif bridgeState.turnEnding then
        reason = "turn ending"
    elseif bridgeState.isMyTurn and bridgeState.waitingForCommand then
        reason = "window open"
    else
        local ours, why = ourActiveTurn()
        if not ours then reason = why end
    end
    if reason then
        if Log then Log.Warn("Re-arm ignored %s: %s", nonce, reason) end
        print(string.format("AZ_REARM:%s:ignored:%s", nonce, reason))
        return false
    end

    if Log then Log.Info("Re-arm accepted %s", nonce) end
    print(string.format("AZ_REARM:%s:accepted", nonce))
    if not bridgeState.isMyTurn then
        bridgeState.actionsThisTurn = 0
    end
    local now = (os and os.time) and os.time() or 0
    bridgeState.isMyTurn = true
    bridgeState.waitingForCommand = true
    bridgeState.turnStartTime = now
    bridgeState.lastCommandTime = now
    bridgeState.exportAfterTicks = nil
    bridgeState.exportAfterFrames = nil
    local exported, err = AlphaZeroStateExporter.ExportAndWrite(bridgeState.aiPlayerId)
    if not exported then
        if Log then Log.Error("Failed to export state after a re-arm: %s", tostring(err)) end
        bridgeState.lastError = err
    end
    AlphaZeroBridge.StartPolling()
    return true
end

-- Poll for and process commands
function AlphaZeroBridge.PollCommands(source)
    -- Before the early return below: a re-arm is what reopens a lapsed window.
    if checkRearm() then
        return
    end

    if not bridgeState.isMyTurn or not bridgeState.waitingForCommand then
        return
    end

    -- The state after a command message is exported a little later, once the
    -- game has carried the requests out. Until then no command is read: the
    -- next one should be chosen on the new state. The wait ends after
    -- STATE_EXPORT_DELAY_TICKS batches of game events, or, when the game
    -- raises none (it has gone idle), after STATE_EXPORT_MAX_FRAMES frames.
    if bridgeState.exportAfterTicks then
        if source == "frame" then
            if bridgeState.exportAfterFrames then
                bridgeState.exportAfterFrames = bridgeState.exportAfterFrames - 1
            end
        else
            bridgeState.exportAfterTicks = bridgeState.exportAfterTicks - 1
        end
        if bridgeState.exportAfterTicks > 0 and (bridgeState.exportAfterFrames or 1) > 0 then
            return
        end
        bridgeState.exportAfterTicks = nil
        bridgeState.exportAfterFrames = nil
        local exported, exportErr = AlphaZeroStateExporter.ExportAndWrite(bridgeState.aiPlayerId)
        if not exported then
            if Log then Log.Error("Failed to export state after a command: %s", tostring(exportErr)) end
            bridgeState.lastError = exportErr
        end
        return
    end

    -- Command timeout check: the window runs from the last command handled,
    -- so a turn of many commands is not cut off thirty seconds after it began.
    local timeout = (AlphaZeroConfig and AlphaZeroConfig.Get("COMMAND_TIMEOUT")) or 30.0
    local since = math.max(bridgeState.turnStartTime or 0, bridgeState.lastCommandTime or 0)
    local elapsed = ((os and os.time) and os.time() or 0) - since
    if elapsed > timeout then
        if Log then Log.Warn("Command timeout after %ds, stopping poll", elapsed) end
        bridgeState.waitingForCommand = false
        return
    end

    -- Write heartbeat periodically, every HEARTBEAT_INTERVAL seconds. A value
    -- that is not a positive number (a typo in the override file) falls back to
    -- one second: an error here would stop command polling.
    local heartbeatInterval = tonumber(AlphaZeroConfig and AlphaZeroConfig.Get("HEARTBEAT_INTERVAL"))
    if not heartbeatInterval or heartbeatInterval <= 0 then
        heartbeatInterval = 1.0
    end
    local now = (os and os.time) and os.time() or 0
    if now - bridgeState.lastHeartbeatTime >= heartbeatInterval then
        if AlphaZeroFileIO then
            AlphaZeroFileIO.WriteHeartbeat()
        end
        bridgeState.lastHeartbeatTime = now
    end

    -- Read commands
    local commandMessage, err = AlphaZeroFileIO.ReadCommands()
    if err then
        if Log then Log.Error("Error reading commands: %s", err) end
        return
    end

    if not commandMessage then
        return  -- No new commands
    end

    -- Process commands
    AlphaZeroBridge.ProcessCommands(commandMessage)
end

-- Commands that can answer what blocks the end of a turn.
local UNBLOCKING_ACTIONS = {
    research_tech = true, research_civic = true, city_produce = true,
    unit_skip = true, unit_fortify = true, unit_sleep = true,
}

-- Process a command message
function AlphaZeroBridge.ProcessCommands(commandMessage)
    if Log then Log.Info("Processing commands: msg_id=%s", tostring(commandMessage.msg_id)) end

    -- A cap on commands per turn, so a loop that keeps sending a command the
    -- game ignores stops. Past it only END_TURN is carried out, with one
    -- exception: while the game refuses to end the turn (a research,
    -- production or unit-orders choice is open), the commands that can answer
    -- that choice are still carried out, or the cap itself would trap the
    -- turn. That allowance ends at twice the cap.
    local commands = commandMessage.commands or {}
    local cap = tonumber(AlphaZeroConfig and AlphaZeroConfig.Get("MAX_COMMANDS_PER_TURN")) or 200
    local onlyEndTurn, onlyUnblocking = #commands > 0, #commands > 0
    for _, command in ipairs(commands) do
        if command.action_type ~= "end_turn" then
            onlyEndTurn = false
            if not UNBLOCKING_ACTIONS[command.action_type] then onlyUnblocking = false end
        end
    end

    -- The count the turn would reach with this message: a batch may not carry
    -- it past a limit either.
    local projected = bridgeState.actionsThisTurn + #commands
    local overCap = projected > cap and not onlyEndTurn
    if overCap and onlyUnblocking and projected <= 2 * cap then
        local blocker = nil
        if AlphaZeroStateExporter and AlphaZeroStateExporter.GetEndTurnBlocker then
            local ok, found, _, unknown = pcall(AlphaZeroStateExporter.GetEndTurnBlocker, bridgeState.aiPlayerId)
            blocker = ok and (found or unknown) or nil
        end
        if blocker then overCap = false end
    end

    local results, errorCode
    if overCap then
        local message = string.format(
            "%d commands already handled this turn (the cap is %d); only end_turn is accepted",
            bridgeState.actionsThisTurn, cap)
        if Log then Log.Warn(message) end
        results = { msg_id = commandMessage.msg_id, success = false, executed = {}, errors = { message } }
        errorCode = "COMMAND_CAP_REACHED"
    else
        -- Only a message that is carried out keeps the command window open.
        bridgeState.lastCommandTime = (os and os.time) and os.time() or 0
        results = AlphaZeroActionExecutor.ExecuteCommands(bridgeState.aiPlayerId, commandMessage)
        errorCode = "EXECUTION_ERROR"
        bridgeState.actionsThisTurn = bridgeState.actionsThisTurn + #commands
    end

    local ackErrorCode = nil
    local ackErrorMessage = nil
    if not results.success then
        ackErrorCode = errorCode
        ackErrorMessage = table.concat(results.errors, "; ")
    end

    -- Write acknowledgment
    local ackSuccess = AlphaZeroFileIO.WriteAck(
        commandMessage.msg_id or 0,
        results.success,
        ackErrorCode,
        ackErrorMessage,
        results.executed
    )

    if not ackSuccess then
        if Log then Log.Error("Failed to write ack") end
    end

    -- Check if turn ended
    local endTurnExecuted = false
    for _, exec in ipairs(results.executed) do
        if exec.action_type == "end_turn" and exec.success then
            endTurnExecuted = true
            break
        end
    end

    if endTurnExecuted then
        bridgeState.waitingForCommand = false
        bridgeState.turnEnding = true
    else
        -- Export the new state some frames from now, not in this one: the
        -- game has not carried the requests out yet.
        local delay = tonumber(AlphaZeroConfig and AlphaZeroConfig.Get("STATE_EXPORT_DELAY_TICKS")) or 2
        if delay < 1 then delay = 1 end
        bridgeState.exportAfterTicks = math.floor(delay)
        if hasFrames then
            local frames = tonumber(AlphaZeroConfig and AlphaZeroConfig.Get("STATE_EXPORT_MAX_FRAMES")) or 30
            if frames < 1 then frames = 1 end
            bridgeState.exportAfterFrames = math.floor(frames)
        else
            bridgeState.exportAfterFrames = nil
        end
    end
end

-- ----------------------------------------------------------------------------
-- Event Handlers
-- ----------------------------------------------------------------------------

-- Player turn started event
local function onPlayerTurnActivated(playerId, isFirstTime)
    if bridgeState.initialized then
        AlphaZeroBridge.OnTurnStart(playerId)
    end
end

-- Player turn ended event
local function onPlayerTurnDeactivated(playerId)
    if bridgeState.initialized then
        AlphaZeroBridge.OnTurnEnd(playerId)
    end
end

-- Game load complete
local function onLoadGameViewStateDone()
    if Log then Log.Info("Game view state loaded") end
    AlphaZeroBridge.Initialize()
end

-- Local player turn begin (single-player fallback)
local function onLocalPlayerTurnBegin()
    if not Game then return end
    local localPlayer = Game.GetLocalPlayer()
    if localPlayer == bridgeState.aiPlayerId then
        AlphaZeroBridge.OnTurnStart(localPlayer)
    end
end

-- Local player turn end (single-player fallback)
local function onLocalPlayerTurnEnd()
    if not Game then return end
    local localPlayer = Game.GetLocalPlayer()
    if localPlayer == bridgeState.aiPlayerId then
        AlphaZeroBridge.OnTurnEnd(localPlayer)
    end
end

-- Register event handlers (deduplicated: no GameEvents duplicates)
local function registerEvents()
    -- Clear any previous tracking
    registeredEvents = {}

    if Events then
        -- Primary turn events (UI context)
        if Events.PlayerTurnActivated then
            Events.PlayerTurnActivated.Add(onPlayerTurnActivated)
            table.insert(registeredEvents, {event = Events.PlayerTurnActivated, handler = onPlayerTurnActivated})
        end

        if Events.PlayerTurnDeactivated then
            Events.PlayerTurnDeactivated.Add(onPlayerTurnDeactivated)
            table.insert(registeredEvents, {event = Events.PlayerTurnDeactivated, handler = onPlayerTurnDeactivated})
        end

        if Events.LoadGameViewStateDone then
            Events.LoadGameViewStateDone.Add(onLoadGameViewStateDone)
            table.insert(registeredEvents, {event = Events.LoadGameViewStateDone, handler = onLoadGameViewStateDone})
        end

        -- Single-player fallback events
        if Events.LocalPlayerTurnBegin then
            Events.LocalPlayerTurnBegin.Add(onLocalPlayerTurnBegin)
            table.insert(registeredEvents, {event = Events.LocalPlayerTurnBegin, handler = onLocalPlayerTurnBegin})
        end

        if Events.LocalPlayerTurnEnd then
            Events.LocalPlayerTurnEnd.Add(onLocalPlayerTurnEnd)
            table.insert(registeredEvents, {event = Events.LocalPlayerTurnEnd, handler = onLocalPlayerTurnEnd})
        end
    end

    -- REMOVED: GameEvents.PlayerTurnStarted and GameEvents.PlayerTurnStartComplete
    -- These fire in gameplay context alongside the UI events above, causing triple-fire.
    -- The re-entrance guard in OnTurnStart provides additional safety.
end

-- Unregister all event listeners (for graceful shutdown)
local function unregisterEvents()
    for _, entry in ipairs(registeredEvents) do
        if entry.event and entry.handler then
            local ok = pcall(function() entry.event.Remove(entry.handler) end)
            if not ok and Log then
                Log.Debug("Could not remove event handler during shutdown")
            end
        end
    end
    registeredEvents = {}
end

-- ----------------------------------------------------------------------------
-- Debug/Status API
-- ----------------------------------------------------------------------------

-- Get bridge status
function AlphaZeroBridge.GetStatus()
    return {
        initialized = bridgeState.initialized,
        aiPlayerId = bridgeState.aiPlayerId,
        isMyTurn = bridgeState.isMyTurn,
        waitingForCommand = bridgeState.waitingForCommand,
        actionsThisTurn = bridgeState.actionsThisTurn,
        isPolling = isPolling,
        lastError = bridgeState.lastError,
    }
end

-- Manually trigger state export (for debugging)
function AlphaZeroBridge.ExportState()
    if bridgeState.aiPlayerId then
        return AlphaZeroStateExporter.ExportAndWrite(bridgeState.aiPlayerId)
    end
    return false, "Not initialized"
end

-- Manually trigger command poll (for debugging)
function AlphaZeroBridge.ManualPoll()
    AlphaZeroBridge.PollCommands()
end

-- Reset bridge state (graceful shutdown)
function AlphaZeroBridge.Reset()
    -- Unregister all event listeners
    unregisterEvents()

    -- Stop polling
    AlphaZeroBridge.StopPolling()

    -- Reset state
    bridgeState.initialized = false
    bridgeState.aiPlayerId = nil
    bridgeState.isMyTurn = false
    bridgeState.waitingForCommand = false
    bridgeState.actionsThisTurn = 0
    bridgeState.turnStartTime = 0
    bridgeState.exportAfterTicks = nil
    bridgeState.exportAfterFrames = nil
    bridgeState.turnEnding = false
    dropRearm("the bridge was reset")

    if AlphaZeroFileIO then
        AlphaZeroFileIO.ClearAll()
    end

    if Log then Log.Info("Bridge reset") end
end

-- ----------------------------------------------------------------------------
-- Entry Point
-- ----------------------------------------------------------------------------

-- Set up pollCallback reference
pollCallback = AlphaZeroBridge.PollCommands

-- Register events on load
registerEvents()

-- Try to initialize immediately if game is already loaded
if Game then
    AlphaZeroBridge.Initialize()
end

print("[AlphaZero] Main module loaded - Bridge ready")
print("[AlphaZero] Version: " .. (AlphaZeroConfig and AlphaZeroConfig.Get("VERSION") or "unknown"))

-- ========== END Main.lua ==========
