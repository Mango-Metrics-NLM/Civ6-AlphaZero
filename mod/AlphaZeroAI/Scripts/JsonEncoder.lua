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
