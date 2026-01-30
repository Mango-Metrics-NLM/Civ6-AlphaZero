-- ============================================================================
-- Config.lua - Configuration management for AlphaZero AI Bridge
-- ============================================================================
-- Provides centralized configuration with sensible defaults and dynamic
-- overrides via environment or file-based configuration.
-- ============================================================================

AlphaZeroConfig = AlphaZeroConfig or {}

-- ----------------------------------------------------------------------------
-- Default Configuration Values
-- ----------------------------------------------------------------------------
-- These can be overridden by placing an alphazero_config.lua file in the
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

    -- Version
    VERSION = "1.0.0",
    PROTOCOL_VERSION = 1,
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

-- Get the base path for Civ6 documents
local function getDocumentsPath()
    -- Try to get from environment or use standard path
    local userProfile = os.getenv("USERPROFILE")
    if userProfile then
        return userProfile .. "\\Documents\\My Games\\Sid Meier's Civilization VI"
    end
    -- Fallback for non-Windows
    local home = os.getenv("HOME")
    if home then
        return home .. "/Documents/My Games/Sid Meier's Civilization VI"
    end
    return "."
end

-- ----------------------------------------------------------------------------
-- Public API
-- ----------------------------------------------------------------------------

-- Initialize configuration
function AlphaZeroConfig.Init()
    -- Start with defaults
    currentConfig = deepCopy(DEFAULT_CONFIG)

    -- Set derived paths
    local basePath = getDocumentsPath()
    currentConfig.COMM_DIR = basePath .. "\\" .. currentConfig.COMM_DIR_NAME
    currentConfig.STATE_PATH = currentConfig.COMM_DIR .. "\\" .. currentConfig.STATE_FILE
    currentConfig.COMMANDS_PATH = currentConfig.COMM_DIR .. "\\" .. currentConfig.COMMANDS_FILE
    currentConfig.ACK_PATH = currentConfig.COMM_DIR .. "\\" .. currentConfig.ACK_FILE
    currentConfig.HEARTBEAT_PATH = currentConfig.COMM_DIR .. "\\" .. currentConfig.HEARTBEAT_FILE
    currentConfig.ERROR_PATH = currentConfig.COMM_DIR .. "\\" .. currentConfig.ERROR_FILE

    -- Try to load override configuration
    AlphaZeroConfig.LoadOverrides()

    return currentConfig
end

-- Load configuration overrides from file
function AlphaZeroConfig.LoadOverrides()
    if not currentConfig then
        AlphaZeroConfig.Init()
        return
    end

    local configPath = currentConfig.COMM_DIR .. "\\alphazero_config.lua"
    local file = io.open(configPath, "r")
    if file then
        local content = file:read("*all")
        file:close()

        -- Try to load as Lua table
        local chunk, err = load("return " .. content)
        if chunk then
            local ok, overrides = pcall(chunk)
            if ok and type(overrides) == 'table' then
                currentConfig = mergeTables(currentConfig, overrides)
            end
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

-- Initialize on load
AlphaZeroConfig.Init()

print("[AlphaZero] Config module loaded - Version " .. AlphaZeroConfig.Get("VERSION"))
