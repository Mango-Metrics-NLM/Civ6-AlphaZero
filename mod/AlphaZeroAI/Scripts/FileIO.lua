-- ============================================================================
-- FileIO.lua - File I/O operations for AlphaZero AI Bridge
-- ============================================================================
-- Handles reading and writing JSON files for communication with Python.
-- Fixes: cross-platform mkdir, path traversal prevention, file size limits,
--        content-hash-based change detection.
-- ============================================================================

AlphaZeroFileIO = AlphaZeroFileIO or {}

local Log = nil  -- Will be set after Logger loads

-- Cross-platform path separator (safe for Civ6 Lua sandbox: package is nil)
local PATH_SEP = (os.getenv("WINDIR") ~= nil) and "\\" or "/"

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
    local safe, err = isPathSafe(path)
    if not safe then
        if Log then Log.Error("Security warning: %s: %s", err, path) end
        return
    end

    local safePath = shellEscape(path)
    if PATH_SEP == "/" then
        -- Unix/macOS
        os.execute('mkdir -p "' .. safePath .. '" 2>/dev/null')
    else
        -- Windows
        os.execute('mkdir "' .. safePath .. '" 2>nul')
    end
end

-- Check if file exists
local function fileExists(path)
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

    -- Load configurable max read size
    if AlphaZeroConfig then
        MAX_READ_BYTES = AlphaZeroConfig.Get("MAX_FILE_READ_BYTES") or MAX_READ_BYTES
    end

    -- Ensure communication directory exists
    local commDir = AlphaZeroConfig and AlphaZeroConfig.Get("COMM_DIR")
    if commDir then
        ensureDirectory(commDir)
        if Log then Log.Info("Comm directory: %s", commDir) end
    end
end

-- Read a file and return contents (with size limit)
function AlphaZeroFileIO.Read(path)
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
