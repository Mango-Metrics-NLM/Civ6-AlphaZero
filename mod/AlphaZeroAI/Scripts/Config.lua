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

    -- Polling intervals (in seconds)
    HEARTBEAT_INTERVAL = 1.0,      -- Write heartbeat every second

    -- Timeouts
    COMMAND_TIMEOUT = 30.0,        -- Wait 30s for command
    ACK_TIMEOUT = 5.0,             -- Wait 5s for ack read

    -- Feature flags
    ENABLE_LOGGING = true,
    LOG_LEVEL = "INFO",            -- DEBUG, INFO, WARN, ERROR
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
