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
    if not os or not os.remove then return true, nil end
    local ok, err = pcall(os.remove, path)
    if not ok then
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
    return PATH_SEP
end

-- Initialize on load
AlphaZeroFileIO.Init()

print("[AlphaZero] FileIO module loaded (io=" .. (HAS_IO and "file" or "print") .. " mode)")

