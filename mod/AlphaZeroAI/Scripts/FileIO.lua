-- ============================================================================
-- FileIO.lua - File I/O operations for AlphaZero AI Bridge
-- ============================================================================
-- Handles reading and writing JSON files for communication with Python.
-- ============================================================================

AlphaZeroFileIO = AlphaZeroFileIO or {}

local Log = nil  -- Will be set after Logger loads

-- Track file modification times to detect changes
local fileModTimes = {}

-- ----------------------------------------------------------------------------
-- Helper Functions
-- ----------------------------------------------------------------------------

-- Ensure directory exists
local function ensureDirectory(path)
    -- Try to create directory (may fail silently if exists)
    local cmd = 'mkdir "' .. path .. '" 2>nul'
    os.execute(cmd)
end

-- Get file modification time (returns 0 if file doesn't exist)
local function getFileModTime(path)
    local file = io.open(path, "r")
    if not file then
        return 0
    end
    file:close()

    -- In Civ6 Lua, we can't get actual mod time, so we use file size as proxy
    local f = io.open(path, "rb")
    if f then
        local size = f:seek("end")
        f:close()
        return size or 0
    end
    return 0
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

    -- Ensure communication directory exists
    local commDir = AlphaZeroConfig.Get("COMM_DIR")
    if commDir then
        ensureDirectory(commDir)
        if Log then Log.Info("Comm directory: %s", commDir) end
    end
end

-- Read a file and return contents
function AlphaZeroFileIO.Read(path)
    local file, err = io.open(path, "r")
    if not file then
        if Log then Log.Debug("Could not open file for reading: %s", path) end
        return nil, "Could not open file: " .. tostring(err)
    end

    local content = file:read("*all")
    file:close()

    if Log then Log.Debug("Read %d bytes from %s", #content, path) end
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

-- Check if file has changed since last read
function AlphaZeroFileIO.HasChanged(path)
    local currentModTime = getFileModTime(path)
    local lastModTime = fileModTimes[path] or 0

    if currentModTime ~= lastModTime and currentModTime > 0 then
        fileModTimes[path] = currentModTime
        return true
    end
    return false
end

-- Reset change tracking for a file
function AlphaZeroFileIO.ResetChangeTracking(path)
    fileModTimes[path] = nil
end

-- Read JSON file and parse
function AlphaZeroFileIO.ReadJson(path)
    local content, readErr = AlphaZeroFileIO.Read(path)
    if not content then
        return nil, readErr
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
    local json, encodeErr = AlphaZeroJson.Encode(data, pretty)
    if not json then
        if Log then Log.Error("JSON encode error: %s", encodeErr) end
        return false, encodeErr
    end

    return AlphaZeroFileIO.Write(path, json)
end

-- Write state to state file
function AlphaZeroFileIO.WriteState(state)
    local path = AlphaZeroConfig.Get("STATE_PATH")
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
    local path = AlphaZeroConfig.Get("COMMANDS_PATH")
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
    local path = AlphaZeroConfig.Get("ACK_PATH")
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
    local path = AlphaZeroConfig.Get("HEARTBEAT_PATH")
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
    local path = AlphaZeroConfig.Get("ERROR_PATH")
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
        AlphaZeroConfig.Get("STATE_PATH"),
        AlphaZeroConfig.Get("COMMANDS_PATH"),
        AlphaZeroConfig.Get("ACK_PATH"),
        AlphaZeroConfig.Get("ERROR_PATH"),
    }

    for _, path in ipairs(files) do
        if path then
            AlphaZeroFileIO.Delete(path)
            AlphaZeroFileIO.ResetChangeTracking(path)
        end
    end

    if Log then Log.Info("Cleared all communication files") end
end

-- Initialize on load
AlphaZeroFileIO.Init()

print("[AlphaZero] FileIO module loaded")
