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
                                        if districtName then
                                            table.insert(cityData.districts, districtName)
                                        end
                                    end
                                end)
                            end

                            local ok_bq, buildQueue = pcall(function() return city:GetBuildQueue() end)
                            if ok_bq and buildQueue then
                                local ok_cb, currentProduction = pcall(function() return buildQueue:CurrentlyBuilding() end)
                                if ok_cb and currentProduction then
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
                turns_left = safeCall(playerCulture, "GetTurnsLeftOnCurrentCivic"),
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

    -- Civics: no "can progress" call is known to work, so a civic is open when
    -- the player lacks it and has every prerequisite listed for it.
    local playerCulture = safeCall(player, "GetCulture")
    local civicTable = GameInfo and GameInfo.Civics
    local prereqTable = GameInfo and GameInfo.CivicPrereqs
    local okHasCivic, hasCivicMethod = pcall(function() return playerCulture.HasCivic end)
    if playerCulture and civicTable and prereqTable and okHasCivic and hasCivicMethod then
        local indexByType, prereqs = {}, {}
        for row in civicTable() do
            indexByType[row.CivicType] = row.Index
        end
        for row in prereqTable() do
            prereqs[row.Civic] = prereqs[row.Civic] or {}
            table.insert(prereqs[row.Civic], row.PrereqCivic)
        end

        local currentCivic = safeCall(playerCulture, "GetProgressingCivic")
        local items, names = {}, {}
        for row in civicTable() do
            if row.Index ~= currentCivic and not safeCall(playerCulture, "HasCivic", row.Index) then
                local open = true
                for _, prereqType in ipairs(prereqs[row.CivicType] or {}) do
                    local prereqIndex = indexByType[prereqType]
                    if prereqIndex ~= nil and not safeCall(playerCulture, "HasCivic", prereqIndex) then
                        open = false
                        break
                    end
                end
                if open then
                    table.insert(items, row.Index)
                    table.insert(names, row.CivicType)
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
    local ok_cities, playerCities = pcall(function() return player:GetCities() end)
    if ok_cities and playerCities then
        for _, city in playerCities:Members() do
            local ok_bq, buildQueue = pcall(function() return city:GetBuildQueue() end)
            if ok_bq and buildQueue then
                local ok_cb, currentlyBuilding = pcall(function() return buildQueue:CurrentlyBuilding() end)
                local needsProduction = ok_cb and not currentlyBuilding
                -- Always export available production for each city
                local items = {}
                local itemTypes = {}

                if GameInfo then
                    pcall(collectProducibleItems, buildQueue, GameInfo.Units, "UnitType", "unit", items, itemTypes)
                    pcall(collectProducibleItems, buildQueue, GameInfo.Buildings, "BuildingType", "building", items, itemTypes)
                    pcall(collectProducibleItems, buildQueue, GameInfo.Districts, "DistrictType", "district", items, itemTypes)
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

-- The notification that stops the player's turn from ending, if any.
-- Returns the name of its EndTurnBlockingTypes value (for example
-- "ENDTURN_BLOCKING_RESEARCH") and its message, or nil when nothing blocks or
-- when this Lua context has no NotificationManager.
function AlphaZeroStateExporter.GetEndTurnBlocker(playerId)
    if not NotificationManager then
        return nil, nil
    end

    local noBlocking = EndTurnBlockingTypes and EndTurnBlockingTypes.NO_ENDTURN_BLOCKING
    local ok, blockerName, message = pcall(function()
        local list = NotificationManager.GetList(playerId)
        if not list then
            return nil, nil
        end
        for _, notificationId in ipairs(list) do
            local entry = NotificationManager.Find(playerId, notificationId)
            if entry and not entry:IsDismissed() then
                local blocking = entry:GetEndTurnBlocking()
                if blocking and blocking ~= 0 and blocking ~= noBlocking then
                    local text = safeCall(entry, "GetMessage")
                    if type(text) ~= "string" or text == "" then
                        text = nil
                    end
                    return endTurnBlockingName(blocking), text
                end
            end
        end
        return nil, nil
    end)

    if ok then
        return blockerName, message
    end
    return nil, nil
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
    state.end_turn_blocking, state.end_turn_blocking_message =
        AlphaZeroStateExporter.GetEndTurnBlocker(playerId)

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
