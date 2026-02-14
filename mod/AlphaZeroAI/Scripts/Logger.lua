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

-- Cross-platform path separator
local PATH_SEP = package.config and package.config:sub(1, 1) or "/"

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
