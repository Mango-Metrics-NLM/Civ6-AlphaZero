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

-- Path separator: Civ6 Lua sandbox strips package and os.getenv, so we
-- hardcode backslash. The mod only runs inside Civ6 (Windows/Steam).
local PATH_SEP = "\\"

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

    -- Player configuration
    AI_PLAYER_ID = nil,            -- Auto-detect first human player
    OBSERVE_ALL_PLAYERS = false,   -- Only observe AI player's visible tiles

    -- Map settings
    MAX_TILES_PER_STATE = 10000,   -- Limit tiles in state dump
    INCLUDE_FOG_OF_WAR = true,

    -- Safety settings
    MAX_ACTIONS_PER_TURN = 1000,   -- Prevent infinite action loops
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

-- ----------------------------------------------------------------------------
-- Public API
-- ----------------------------------------------------------------------------

-- Initialize configuration
function AlphaZeroConfig.Init()
    -- Start with defaults
    currentConfig = deepCopy(DEFAULT_CONFIG)

    -- Set derived paths using detected path separator
    local basePath = getDocumentsPath()
    print("[AlphaZero] Config: base path = " .. tostring(basePath))
    currentConfig.COMM_DIR = basePath .. PATH_SEP .. currentConfig.COMM_DIR_NAME
    print("[AlphaZero] Config: COMM_DIR = " .. tostring(currentConfig.COMM_DIR))
    currentConfig.STATE_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.STATE_FILE
    currentConfig.COMMANDS_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.COMMANDS_FILE
    currentConfig.ACK_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.ACK_FILE
    currentConfig.HEARTBEAT_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.HEARTBEAT_FILE
    currentConfig.ERROR_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.ERROR_FILE

    -- Try to load override configuration
    AlphaZeroConfig.LoadOverrides()

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

    local configPath = currentConfig.COMM_DIR .. PATH_SEP .. "alphazero_config.json"
    local ok, file = pcall(io.open, configPath, "r")
    if not ok or not file then return end

    local content = file:read("*all")
    file:close()

    -- Use JSON decoder (safe, no code execution)
    if AlphaZeroJson and content and #content > 0 then
        local overrides, err = AlphaZeroJson.Decode(content)
        if overrides and type(overrides) == 'table' then
            currentConfig = mergeTables(currentConfig, overrides)
            -- Recalculate derived paths if COMM_DIR changed
            if overrides.COMM_DIR then
                currentConfig.STATE_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.STATE_FILE
                currentConfig.COMMANDS_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.COMMANDS_FILE
                currentConfig.ACK_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.ACK_FILE
                currentConfig.HEARTBEAT_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.HEARTBEAT_FILE
                currentConfig.ERROR_PATH = currentConfig.COMM_DIR .. PATH_SEP .. currentConfig.ERROR_FILE
            end
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
    return PATH_SEP
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

    local f = io.open(logPath, "rb")
    if f then
        local size = f:seek("end")
        f:close()
        if size and size > MAX_LOG_SIZE then
            -- Rotate: remove old backup, rename current to .old
            os.remove(logPath .. ".old")
            os.rename(logPath, logPath .. ".old")
        end
    end
end

-- Write to log file (if enabled)
local function writeToFile(formatted)
    if not AlphaZeroConfig or not AlphaZeroConfig.IsEnabled("LOGGING") then
        return
    end

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

local function escapeString(str)
    str = string.gsub(str, '[\\"\n\r\t\b\f]', escapeChars)
    -- Escape control characters
    str = string.gsub(str, '[\x00-\x1f]', function(c)
        return string.format("\\u%04x", string.byte(c))
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

    -- Use integer format if whole number
    if math.floor(num) == num and math.abs(num) < 2^53 then
        return string.format("%d", num)
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

-- Path separator: hardcoded for Civ6 Windows/Steam (os.getenv stripped from sandbox)
local PATH_SEP = "\\"

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
        if PATH_SEP == "/" then
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
function AlphaZeroFileIO.Delete(path)
    local ok, err = os.remove(path)
    if not ok and fileExists(path) then
        if Log then Log.Warn("Could not delete file: %s", path) end
        return false, err
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
function AlphaZeroFileIO.WriteState(state)
    local path = AlphaZeroConfig and AlphaZeroConfig.Get("STATE_PATH")
    if not path then
        return false, "STATE_PATH not configured"
    end

    local ok, err = AlphaZeroFileIO.WriteJson(path, state, false)
    if ok and Log then
        Log.Info("State written: turn=%s, units=%d, cities=%d",
            tostring(state.turn_id),
            state.units and #state.units or 0,
            state.cities and #state.cities or 0)
    end
    return ok, err
end

-- Read commands from commands file
function AlphaZeroFileIO.ReadCommands()
    local path = AlphaZeroConfig and AlphaZeroConfig.Get("COMMANDS_PATH")
    if not path then
        return nil, "COMMANDS_PATH not configured"
    end

    -- Check if file changed
    if not AlphaZeroFileIO.HasChanged(path) then
        return nil, nil  -- No new commands
    end

    local data, err = AlphaZeroFileIO.ReadJson(path)
    if err then
        return nil, err
    end

    if data and Log then
        Log.Info("Commands received: msg_id=%s, count=%d",
            tostring(data.msg_id),
            data.commands and #data.commands or 0)
    end

    return data, nil
end

-- Write acknowledgment
function AlphaZeroFileIO.WriteAck(msgId, success, errorCode, errorMessage, executed)
    local path = AlphaZeroConfig and AlphaZeroConfig.Get("ACK_PATH")
    if not path then
        return false, "ACK_PATH not configured"
    end

    local ack = {
        msg_id = msgId,
        timestamp = os.time(),
        success = success,
        error_code = errorCode,
        error_message = errorMessage,
        executed = executed or {},
    }

    local ok, err = AlphaZeroFileIO.WriteJson(path, ack, false)
    if ok and Log then
        Log.Debug("Ack written: msg_id=%d, success=%s", msgId, tostring(success))
    end
    return ok, err
end

-- Write heartbeat
function AlphaZeroFileIO.WriteHeartbeat()
    local path = AlphaZeroConfig and AlphaZeroConfig.Get("HEARTBEAT_PATH")
    if not path then
        return false, "HEARTBEAT_PATH not configured"
    end

    local heartbeat = {
        timestamp = os.time(),
        lua_alive = true,
        turn = 0,
    }

    if Game then
        heartbeat.turn = Game.GetCurrentGameTurn() or 0
    end

    return AlphaZeroFileIO.WriteJson(path, heartbeat, false)
end

-- Write error to error file
function AlphaZeroFileIO.WriteError(code, message, context)
    local path = AlphaZeroConfig and AlphaZeroConfig.Get("ERROR_PATH")
    if not path then
        return false, "ERROR_PATH not configured"
    end

    local errorData = {
        timestamp = os.time(),
        code = code,
        message = message,
        context = context,
    }

    if Game then
        errorData.turn = Game.GetCurrentGameTurn() or 0
    end

    if not AlphaZeroJson then
        return false, "JSON encoder not available"
    end

    -- Append to error file
    local json, encodeErr = AlphaZeroJson.Encode(errorData, false)
    if not json then
        return false, encodeErr
    end

    return AlphaZeroFileIO.Append(path, json .. "\n")
end

-- Clear all communication files
function AlphaZeroFileIO.ClearAll()
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

    if Log then Log.Info("Cleared all communication files") end
end

-- Get path separator (for reuse by other modules)
function AlphaZeroFileIO.GetPathSep()
    return PATH_SEP
end

-- Initialize on load
AlphaZeroFileIO.Init()

print("[AlphaZero] FileIO module loaded")

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

-- Hex-grid neighbor offsets (Civ6 uses offset coordinates).
-- Neighbor positions differ based on whether the row (y) is even or odd.
local HEX_EVEN_ROW_OFFSETS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, 1}}
local HEX_ODD_ROW_OFFSETS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, -1}, {-1, -1}}

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

-- Get hex grid neighbors (Civ6 uses offset coordinates)
local function getHexNeighbors(x, y)
    local neighbors = {}
    local offsets = (y % 2 == 0) and HEX_EVEN_ROW_OFFSETS or HEX_ODD_ROW_OFFSETS
    for _, offset in ipairs(offsets) do
        table.insert(neighbors, {x + offset[1], y + offset[2]})
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

-- Get observable player IDs (own player + optionally all others)
local function getObservablePlayerIds(playerId)
    local playerIds = {playerId}
    local observeAll = AlphaZeroConfig and AlphaZeroConfig.Get("OBSERVE_ALL_PLAYERS")
    if observeAll and PlayerManager and PlayerManager.GetWasEverAliveCount then
        for i = 0, PlayerManager.GetWasEverAliveCount() - 1 do
            if i ~= playerId then
                table.insert(playerIds, i)
            end
        end
    end
    return playerIds
end

-- Get producible items from a GameInfo table (Units, Buildings, or Districts)
-- Returns items appended to the provided items/itemTypes tables
local function collectProducibleItems(buildQueue, gameInfoTable, typeField, itemCategory, items, itemTypes)
    if not gameInfoTable or not buildQueue or not buildQueue.CanProduce then
        return
    end
    for row in gameInfoTable() do
        local ok, canProd = pcall(function()
            return buildQueue:CanProduce(row.Index, true)
        end)
        if ok and canProd then
            local key = row[typeField] or row.Index
            table.insert(items, key)
            itemTypes[key] = itemCategory
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

    local visibilityPlayer = Players and Players[playerId]
    local playerVisibility = nil
    if visibilityPlayer and PlayersVisibility then
        playerVisibility = PlayersVisibility[playerId]
    end

    for x = 0, mapData.width - 1 do
        for y = 0, mapData.height - 1 do
            if tileCount >= maxTiles then
                break
            end

            local plot = Map.GetPlot(x, y)
            if plot then
                local isVisible = true
                local isRevealed = true

                if playerVisibility then
                    isVisible = playerVisibility:IsVisible(x, y)
                    isRevealed = playerVisibility:IsRevealed(x, y)
                end

                if isRevealed or not includeFog then
                    local tile = {
                        x = x,
                        y = y,
                        terrain = getTerrainTypeName(plot:GetTerrainType()),
                        features = {},
                        resource = getResourceTypeName(plot:GetResourceType()),
                        improvement = getImprovementTypeName(plot:GetImprovementType()),
                        owner_id = plot:GetOwner(),
                        is_visible = isVisible,
                        is_revealed = isRevealed,
                        yields = {},
                    }

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

                    -- Add yields if visible (with nil guard on YieldTypes)
                    if isVisible and YieldTypes then
                        tile.yields = {
                            food = plot:GetYield(YieldTypes.YIELD_FOOD or 0) or 0,
                            production = plot:GetYield(YieldTypes.YIELD_PRODUCTION or 0) or 0,
                            gold = plot:GetYield(YieldTypes.YIELD_GOLD or 0) or 0,
                            science = plot:GetYield(YieldTypes.YIELD_SCIENCE or 0) or 0,
                            culture = plot:GetYield(YieldTypes.YIELD_CULTURE or 0) or 0,
                            faith = plot:GetYield(YieldTypes.YIELD_FAITH or 0) or 0,
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

    local playerVisibility = PlayersVisibility and PlayersVisibility[playerId]
    local playerIds = getObservablePlayerIds(playerId)

    for _, pid in ipairs(playerIds) do
        local player = Players[pid]
        if player then
            local playerUnits = player:GetUnits()
            if playerUnits then
                for i, unit in playerUnits:Members() do
                    local x, y = unit:GetX(), unit:GetY()

                    -- Check visibility for non-own units
                    local canSee = (pid == playerId)
                    if not canSee and playerVisibility then
                        canSee = playerVisibility:IsVisible(x, y)
                    end

                    if canSee then
                        local typeInfo = getUnitTypeInfo(unit:GetUnitType())

                        local unitData = {
                            id = unit:GetID(),
                            type_name = typeInfo.type_name,
                            owner = pid,
                            x = x,
                            y = y,
                            hp = unit:GetDamage() and (100 - unit:GetDamage()) or 100,
                            max_hp = 100,
                            damage = unit:GetDamage() or 0,
                            movement = unit:GetMovementMovesRemaining() or 0,
                            max_movement = unit:GetMaxMoves() or 2,
                            combat_strength = unit:GetCombat() or 0,
                            ranged_strength = unit:GetRangedCombat() or 0,
                            range = unit:GetRange() or 0,
                            is_civilian = typeInfo.is_civilian,
                            can_attack = (unit:GetAttacksRemaining() or 0) > 0,
                            promotions = {},
                        }

                        -- Get promotions
                        local unitExp = unit:GetExperience()
                        if unitExp then
                            local promotions = unitExp:GetPromotions()
                            for _, promo in ipairs(promotions or {}) do
                                table.insert(unitData.promotions, promo)
                            end
                        end

                        table.insert(units, unitData)
                    end
                end
            end
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

    local playerVisibility = PlayersVisibility and PlayersVisibility[playerId]
    local playerIds = getObservablePlayerIds(playerId)

    for _, pid in ipairs(playerIds) do
        local player = Players[pid]
        if player then
            local playerCities = player:GetCities()
            if playerCities then
                for i, city in playerCities:Members() do
                    local x, y = city:GetX(), city:GetY()

                    -- Check visibility
                    local canSee = (pid == playerId)
                    if not canSee and playerVisibility then
                        canSee = playerVisibility:IsRevealed(x, y)
                    end

                    if canSee then
                        local cityData = {
                            id = city:GetID(),
                            name = city:GetName() or "Unknown",
                            owner = pid,
                            x = x,
                            y = y,
                            population = city:GetPopulation() or 1,
                            is_capital = city:IsCapital(),
                            defense_strength = city:GetStrengthValue() or 0,
                            buildings = {},
                            districts = {},
                            current_production = nil,
                            production_progress = 0,
                            available_production = {},
                        }

                        -- Only include detailed info for own cities
                        if pid == playerId then
                            local buildQueue = city:GetBuildQueue()
                            if buildQueue then
                                local currentProduction = buildQueue:CurrentlyBuilding()
                                if currentProduction then
                                    cityData.current_production = currentProduction
                                end
                                cityData.production_progress = buildQueue:GetTurnsLeft() or 0
                            end
                        end

                        table.insert(cities, cityData)
                    end
                end
            end
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

        -- Export researched civics
        if playerCulture.GetResearchedCivics then
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
        if playerTech.GetResearchingTech then
            playerData.current_research = playerTech:GetResearchingTech()
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
            local movesRemaining = unit:GetMovementMovesRemaining() or 0

            if movesRemaining > 0 then
                -- Get valid move targets using hex-grid neighbors
                local targets = {}
                local ux, uy = unit:GetX(), unit:GetY()
                local neighbors = getHexNeighbors(ux, uy)

                for _, neighbor in ipairs(neighbors) do
                    local tx, ty = neighbor[1], neighbor[2]
                    if tx >= 0 and ty >= 0 then
                        local plot = Map and Map.GetPlot(tx, ty)
                        if plot and not plot:IsImpassable() then
                            table.insert(targets, {tx, ty})
                        end
                    end
                end

                if #targets > 0 then
                    table.insert(actions, {
                        action = "move",
                        unit_id = unitId,
                        targets = targets,
                    })
                end

                -- Fortify action (available if unit has movement)
                if not unit:IsCivilian() then
                    table.insert(actions, {
                        action = "fortify",
                        unit_id = unitId,
                    })
                end

                -- Skip action (always available if has movement)
                table.insert(actions, {
                    action = "skip",
                    unit_id = unitId,
                })

                -- Found city (if settler)
                local typeInfo = getUnitTypeInfo(unit:GetUnitType())
                if typeInfo.type_name == "UNIT_SETTLER" or
                   (typeInfo.is_civilian and string.find(typeInfo.type_name, "SETTLER")) then
                    table.insert(actions, {
                        action = "found_city",
                        unit_id = unitId,
                    })
                end
            end

            -- Attack actions (check for enemy units in range)
            if (unit:GetAttacksRemaining() or 0) > 0 then
                local ux, uy = unit:GetX(), unit:GetY()
                local range = unit:GetRange() or 1
                local attackTargets = {}

                -- Get tiles in attack range
                local tilesInRange
                if range <= 1 then
                    tilesInRange = getHexNeighbors(ux, uy)
                else
                    tilesInRange = getHexTilesInRange(ux, uy, range)
                end

                for _, tilePos in ipairs(tilesInRange) do
                    local tx, ty = tilePos[1], tilePos[2]
                    if tx >= 0 and ty >= 0 then
                        local plot = Map and Map.GetPlot(tx, ty)
                        if plot then
                            -- Check for enemy units on this tile
                            local plotUnits = Units and Units.GetUnitsInPlot(plot)
                            if plotUnits then
                                for _, enemyUnit in plotUnits:Members() do
                                    if enemyUnit:GetOwner() ~= playerId then
                                        table.insert(attackTargets, {tx, ty})
                                        break
                                    end
                                end
                            end
                        end
                    end
                end

                if #attackTargets > 0 then
                    local attackAction = range > 1 and "ranged_attack" or "attack"
                    table.insert(actions, {
                        action = attackAction,
                        unit_id = unitId,
                        targets = attackTargets,
                    })
                end
            end
        end
    end

    -- City production actions (with actual available items)
    local playerCities = player:GetCities()
    if playerCities then
        for _, city in playerCities:Members() do
            local buildQueue = city:GetBuildQueue()
            if buildQueue then
                local needsProduction = not buildQueue:CurrentlyBuilding()
                -- Always export available production for each city
                local items = {}
                local itemTypes = {}

                if GameInfo then
                    collectProducibleItems(buildQueue, GameInfo.Units, "UnitType", "unit", items, itemTypes)
                    collectProducibleItems(buildQueue, GameInfo.Buildings, "BuildingType", "building", items, itemTypes)
                    collectProducibleItems(buildQueue, GameInfo.Districts, "DistrictType", "district", items, itemTypes)
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

-- Export full game state for a player
function AlphaZeroStateExporter.ExportState(playerId)
    local startTime = os.clock and os.clock() or 0

    local state = {
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

    -- Export components
    state.map = exportMap(playerId)
    state.units = exportUnits(playerId)
    state.cities = exportCities(playerId)
    state.player = exportPlayer(playerId)
    state.players = { state.player }  -- Include at least current player

    if AlphaZeroConfig and AlphaZeroConfig.IsEnabled("LEGAL_ACTIONS") then
        state.legal_actions = exportLegalActions(playerId)
    end

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

-- Export and write state to file
function AlphaZeroStateExporter.ExportAndWrite(playerId)
    local state = AlphaZeroStateExporter.ExportState(playerId)
    return AlphaZeroFileIO.WriteState(state)
end

-- Initialize on load
AlphaZeroStateExporter.Init()

print("[AlphaZero] StateExporter module loaded")

-- ========== END StateExporter.lua ==========

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

-- ========== END ActionExecutor.lua ==========

-- ========== BEGIN Main.lua ==========
-- ============================================================================
-- Main.lua - Entry point for AlphaZero AI Bridge
-- ============================================================================
-- Coordinates all modules and handles game events to provide AI control.
-- Fixes: deduplicated events, proper polling handle, graceful shutdown, timeout.
-- ============================================================================


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
}

-- Polling state (use explicit callback reference for reliable removal)
local isPolling = false
local pollCallback = nil  -- Will be set to AlphaZeroBridge.PollCommands after definition

-- Track registered events for graceful shutdown
local registeredEvents = {}

-- ----------------------------------------------------------------------------
-- Initialization
-- ----------------------------------------------------------------------------

-- Find the AI player (first human player we'll control)
local function findAIPlayer()
    local configuredId = AlphaZeroConfig and AlphaZeroConfig.Get("AI_PLAYER_ID")
    if configuredId then
        return configuredId
    end

    -- Find first human player
    if PlayerManager and PlayerManager.GetWasEverAliveCount then
        for i = 0, PlayerManager.GetWasEverAliveCount() - 1 do
            local player = Players and Players[i]
            if player and player:IsHuman() and player:IsAlive() then
                return i
            end
        end
    end

    return 0  -- Fallback to player 0
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

    -- Clear old files
    if AlphaZeroFileIO then
        AlphaZeroFileIO.ClearAll()
        AlphaZeroFileIO.WriteHeartbeat()
    end

    bridgeState.initialized = true
    if Log then Log.Info("AlphaZero Bridge initialized successfully") end

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

    bridgeState.isMyTurn = true
    bridgeState.actionsThisTurn = 0
    bridgeState.waitingForCommand = true
    bridgeState.turnStartTime = os.time()

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

    if Events and Events.GameCoreEventPublishComplete then
        -- Ensure pollCallback is set
        if not pollCallback then
            pollCallback = AlphaZeroBridge.PollCommands
        end
        Events.GameCoreEventPublishComplete.Add(pollCallback)
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
    isPolling = false
end

-- Poll for and process commands
function AlphaZeroBridge.PollCommands()
    if not bridgeState.isMyTurn or not bridgeState.waitingForCommand then
        return
    end

    -- Command timeout check
    local timeout = (AlphaZeroConfig and AlphaZeroConfig.Get("COMMAND_TIMEOUT")) or 30.0
    local elapsed = os.time() - bridgeState.turnStartTime
    if elapsed > timeout then
        if Log then Log.Warn("Command timeout after %ds, stopping poll", elapsed) end
        bridgeState.waitingForCommand = false
        return
    end

    -- Write heartbeat periodically
    local now = os.time()
    if now - bridgeState.lastHeartbeatTime >= 1 then
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

-- Process a command message
function AlphaZeroBridge.ProcessCommands(commandMessage)
    if Log then Log.Info("Processing commands: msg_id=%s", tostring(commandMessage.msg_id)) end

    local results = AlphaZeroActionExecutor.ExecuteCommands(
        bridgeState.aiPlayerId,
        commandMessage
    )

    -- Count actions
    bridgeState.actionsThisTurn = bridgeState.actionsThisTurn + #(commandMessage.commands or {})

    -- Write acknowledgment
    local ackSuccess = AlphaZeroFileIO.WriteAck(
        commandMessage.msg_id or 0,
        results.success,
        results.success and nil or "EXECUTION_ERROR",
        results.success and nil or table.concat(results.errors, "; "),
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
    else
        -- Export new state after actions
        AlphaZeroStateExporter.ExportAndWrite(bridgeState.aiPlayerId)
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
