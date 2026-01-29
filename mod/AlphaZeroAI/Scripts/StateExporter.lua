-- ============================================================================
-- StateExporter.lua - Game state serialization for AlphaZero AI Bridge
-- ============================================================================
-- Exports the current game state to JSON format for the Python agent.
-- ============================================================================

AlphaZeroStateExporter = AlphaZeroStateExporter or {}

local Log = nil  -- Will be set after Logger loads

-- Cache for expensive lookups
local unitTypeCache = {}
local buildingTypeCache = {}

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
    if featureTypeId == -1 then return nil end

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
    if resourceTypeId == -1 then return nil end

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
    if improvementTypeId == -1 then return nil end

    if GameInfo and GameInfo.Improvements then
        for row in GameInfo.Improvements() do
            if row.Index == improvementTypeId then
                return row.ImprovementType
            end
        end
    end
    return nil
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

    mapData.width = Map.GetGridSize()
    mapData.height = Map.GetGridSize()

    local maxTiles = AlphaZeroConfig.Get("MAX_TILES_PER_STATE") or 10000
    local tileCount = 0
    local includeFog = AlphaZeroConfig.Get("INCLUDE_FOG_OF_WAR")

    local visibilityPlayer = Players[playerId]
    local playerVisibility = nil
    if visibilityPlayer then
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

                    -- Add yields if visible
                    if isVisible then
                        tile.yields = {
                            food = plot:GetYield(YieldTypes.YIELD_FOOD) or 0,
                            production = plot:GetYield(YieldTypes.YIELD_PRODUCTION) or 0,
                            gold = plot:GetYield(YieldTypes.YIELD_GOLD) or 0,
                            science = plot:GetYield(YieldTypes.YIELD_SCIENCE) or 0,
                            culture = plot:GetYield(YieldTypes.YIELD_CULTURE) or 0,
                            faith = plot:GetYield(YieldTypes.YIELD_FAITH) or 0,
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

    local observeAll = AlphaZeroConfig.Get("OBSERVE_ALL_PLAYERS")
    local playerVisibility = PlayersVisibility and PlayersVisibility[playerId]

    -- Get all players to iterate
    local playerIds = {playerId}
    if observeAll then
        for i = 0, PlayerManager.GetWasEverAliveCount() - 1 do
            if i ~= playerId then
                table.insert(playerIds, i)
            end
        end
    end

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
                            can_attack = unit:GetAttacksRemaining() > 0,
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

    local observeAll = AlphaZeroConfig.Get("OBSERVE_ALL_PLAYERS")
    local playerVisibility = PlayersVisibility and PlayersVisibility[playerId]

    local playerIds = {playerId}
    if observeAll then
        for i = 0, PlayerManager.GetWasEverAliveCount() - 1 do
            if i ~= playerId then
                table.insert(playerIds, i)
            end
        end
    end

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

                            -- Get available production
                            -- Note: This is expensive, so we limit it
                            -- cityData.available_production = getAvailableProduction(city)
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

-- Export player data
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
    }

    local player = Players[playerId]
    if not player then
        return playerData
    end

    playerData.is_alive = player:IsAlive()
    playerData.is_human = player:IsHuman()

    -- Get civilization
    local config = PlayerConfigurations[playerId]
    if config then
        playerData.civilization = config:GetCivilizationTypeName() or "UNKNOWN"
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
    end

    local playerTech = player:GetTechs()
    if playerTech then
        playerData.science_per_turn = playerTech:GetScienceYield() or 0
        playerData.current_research = playerTech:GetResearchingTech()
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

-- Export legal actions
local function exportLegalActions(playerId)
    local actions = {}

    -- End turn is always available
    table.insert(actions, { action = "end_turn" })

    local player = Players[playerId]
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
                -- Get valid move targets
                local targets = {}
                local ux, uy = unit:GetX(), unit:GetY()

                -- Simple adjacent tile check (actual pathfinding is complex)
                local directions = {
                    {-1, 0}, {1, 0}, {0, -1}, {0, 1},
                    {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
                }

                for _, dir in ipairs(directions) do
                    local tx, ty = ux + dir[1], uy + dir[2]
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
            end

            -- Attack actions
            if unit:GetAttacksRemaining() > 0 then
                -- Get valid attack targets (simplified)
                local ux, uy = unit:GetX(), unit:GetY()
                local range = unit:GetRange() or 1

                -- Would need to check for enemy units in range
                -- This is a placeholder
            end
        end
    end

    -- City production actions
    local playerCities = player:GetCities()
    if playerCities then
        for _, city in playerCities:Members() do
            local buildQueue = city:GetBuildQueue()
            if buildQueue and not buildQueue:CurrentlyBuilding() then
                -- City needs production order
                table.insert(actions, {
                    action = "produce",
                    city_id = city:GetID(),
                    items = {},  -- Would list available items
                })
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
        state.era = Game.GetEras():GetCurrentEra() or 0
    end

    -- Get current player civilization name
    local config = PlayerConfigurations[playerId]
    if config then
        state.current_player = config:GetCivilizationTypeName() or ""
    end

    -- Export components
    state.map = exportMap(playerId)
    state.units = exportUnits(playerId)
    state.cities = exportCities(playerId)
    state.player = exportPlayer(playerId)
    state.players = { state.player }  -- Include at least current player

    if AlphaZeroConfig.IsEnabled("LEGAL_ACTIONS") then
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
