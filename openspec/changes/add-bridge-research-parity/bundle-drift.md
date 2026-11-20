# Bundle drift reconciled by GM-B0

Before GM-B0 the mod's loaded bundle (`mod/AlphaZeroAI/Scripts/AlphaZeroMain.lua`) was maintained by hand
and had drifted from the module files beside it. GM-B0 made the module files equal to the bundle, because
the bundle is what ran in the game, so that generating the bundle changes no live behaviour.

This file keeps what that reconciliation changed in the module files, so nothing is lost:
lines marked `-` were in a module file and **never ran in the game**; lines marked `+` were only in
the bundle and did run. Each `-` block is a candidate to bring back deliberately in GM-B2, as a reviewed
change to live behaviour. Whitespace-only differences (a trailing blank line per module) are not listed.
Non-blank lines that differed: 24, in three modules. Bundle sha256 before and after: `e33177f9c33c11e1fcf6e2986bec3d97097868ef51fafce39f0caa7ba80622f6`.


**Status, 2026-11-20 (GM-B2):** both `-` blocks below that carried behaviour were brought back as reviewed
changes, each in its own commit with tests that run the Lua: the diplomacy error reporting in
`ActionExecutor.lua`, and the config-read heartbeat interval in `Main.lua` (with one addition: a value that
is not a positive number falls back to one second). The `Config.lua` lines were only ever in the bundle and
are unchanged.


## `Config.lua` (8 lines)

```diff
--- Config.lua (module file before)
+++ Config.lua (bundle section)
@@ -33,4 +33,6 @@
 
     -- Polling intervals (in seconds)
+    COMMAND_POLL_INTERVAL = 0.01,  -- 10ms
+    STATE_WRITE_DELAY = 0.001,     -- 1ms delay after write
     HEARTBEAT_INTERVAL = 1.0,      -- Write heartbeat every second
 
@@ -42,4 +44,7 @@
     ENABLE_LOGGING = true,
     LOG_LEVEL = "INFO",            -- DEBUG, INFO, WARN, ERROR
+    ENABLE_STATE_COMPRESSION = false,
+    ENABLE_DETAILED_UNITS = true,
+    ENABLE_DETAILED_CITIES = true,
     ENABLE_LEGAL_ACTIONS = true,
 
@@ -250,4 +255,8 @@
     end
 
+    if currentConfig.COMMAND_POLL_INTERVAL <= 0 then
+        table.insert(errors, "COMMAND_POLL_INTERVAL must be positive")
+    end
+
     if currentConfig.MAX_ACTIONS_PER_TURN <= 0 then
         table.insert(errors, "MAX_ACTIONS_PER_TURN must be positive")
```

## `ActionExecutor.lua` (13 lines)

```diff
--- ActionExecutor.lua (module file before)
+++ ActionExecutor.lua (bundle section)
@@ -661,31 +661,20 @@
     end
 
-    local lastError = nil
-
     -- Primary path: GetDiplomacy():DoAction(...). Civ6 expects a string or
     -- int action constant from DiplomaticActions table; pass civ6ActionName.
     if diplomacy.DoAction then
-        local ok, err = pcall(function() diplomacy:DoAction(civ6ActionName, targetPlayerId) end)
+        local ok = pcall(function() diplomacy:DoAction(civ6ActionName, targetPlayerId) end)
         if ok then
             return true, nil, string.format("%s against player %d", subActionName, targetPlayerId)
         end
-        lastError = err
-        if Log then Log.Warn("DoAction failed for %s: %s", subActionName, tostring(err)) end
     end
 
     -- Fallback: UI.RequestDiplomacyMove for older mod contexts.
     if UI and UI.RequestDiplomacyMove then
-        local ok, err = pcall(function() UI.RequestDiplomacyMove(civ6ActionName, targetPlayerId) end)
+        local ok = pcall(function() UI.RequestDiplomacyMove(civ6ActionName, targetPlayerId) end)
         if ok then
             return true, nil, string.format("%s requested via UI (player %d)",
                 subActionName, targetPlayerId)
         end
-        lastError = err
-        if Log then Log.Warn("RequestDiplomacyMove failed for %s: %s", subActionName, tostring(err)) end
-    end
-
-    if lastError then
-        return false, ERROR_CODES.EXECUTION_FAILED,
-            string.format("Diplomacy action %s failed: %s", subActionName, tostring(lastError))
     end
 
```

## `Main.lua` (3 lines)

```diff
--- Main.lua (module file before)
+++ Main.lua (bundle section)
@@ -258,7 +258,6 @@
 
     -- Write heartbeat periodically
-    local heartbeatInterval = (AlphaZeroConfig and AlphaZeroConfig.Get("HEARTBEAT_INTERVAL")) or 1.0
     local now = (os and os.time) and os.time() or 0
-    if now - bridgeState.lastHeartbeatTime >= heartbeatInterval then
+    if now - bridgeState.lastHeartbeatTime >= 1 then
         if AlphaZeroFileIO then
             AlphaZeroFileIO.WriteHeartbeat()
```
