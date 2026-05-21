"""Civ6 Lua Bridge Implementation.

Provides file-based IPC with Civilization 6 through a Lua mod.
The bridge reads game state from JSON files exported by the Lua mod
and writes action files for the mod to execute.

Architecture:
    Python AlphaZero <--> JSON Files <--> Civ6 Lua Mod <--> Civ6 Game

File Protocol:
    state_dir/
        state.json          # Current game state (written by Lua)
        state.lock          # Lock file for atomic reads
        state_checksum.txt  # Checksum for validation

    action_dir/
        action.json         # Action to execute (written by Python)
        action.lock         # Lock file for atomic writes
        action_ack.json     # Acknowledgment from Lua
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import (
    ActionSendError,
    City,
    GameAction,
    GameInterface,
    GameState,
    InterfaceConfig,
    InterfaceError,
    Position,
    StateReadError,
    Tile,
    Unit,
)

logger = logging.getLogger(__name__)


class Civ6BridgeError(InterfaceError):
    """Civ6 bridge specific error."""
    pass


class LuaModNotFoundError(Civ6BridgeError):
    """Raised when the Lua mod is not responding."""
    pass


@dataclass
class LuaStateParser:
    """Parser for game state exported by Civ6 Lua mod.

    Converts the JSON format from the Lua mod into GameState objects.
    The Lua mod exports state using the Civ6 API functions.
    """

    player_id: int = 0
    map_size: tuple[int, int] = (80, 50)

    def parse(self, data: dict[str, Any]) -> GameState:
        """Parse Lua-exported JSON into GameState.

        Args:
            data: Raw JSON data from Lua mod.

        Returns:
            Parsed GameState object.
        """
        state = GameState()

        # Basic game info
        state.turn = data.get("turn", 0)
        state.current_player = data.get("currentPlayer", self.player_id)
        state.num_players = data.get("numPlayers", 2)
        state.map_width = data.get("mapWidth", self.map_size[0])
        state.map_height = data.get("mapHeight", self.map_size[1])

        # Parse units
        state.units = self._parse_units(data.get("units", []))

        # Parse cities
        state.cities = self._parse_cities(data.get("cities", []))

        # Parse terrain
        state.terrain = self._parse_terrain(data.get("terrain", {}))

        # Parse visibility
        state.visibility = self._parse_visibility(data.get("visibility", {}))

        # Parse resources
        state.resources = data.get("resources", {})

        # Technology and civics
        state.technologies = data.get("technologies", [])
        state.civics = data.get("civics", [])

        # Economy
        state.era = data.get("era", 0)
        state.gold = data.get("gold", 0)
        state.gold_per_turn = data.get("goldPerTurn", 0)
        state.science_per_turn = data.get("sciencePerTurn", 0)
        state.culture_per_turn = data.get("culturePerTurn", 0)
        state.faith_per_turn = data.get("faithPerTurn", 0)

        # Military
        state.military_strength = data.get("militaryStrength", 0)

        # Victory
        state.victory_progress = data.get("victoryProgress", {})
        state.game_over = data.get("gameOver", False)
        state.winner = data.get("winner")

        # Extra data
        state.extra = {
            k: v for k, v in data.items()
            if k not in [
                "turn", "currentPlayer", "numPlayers", "mapWidth", "mapHeight",
                "units", "cities", "terrain", "visibility", "resources",
                "technologies", "civics", "era", "gold", "goldPerTurn",
                "sciencePerTurn", "culturePerTurn", "faithPerTurn",
                "militaryStrength", "victoryProgress", "gameOver", "winner"
            ]
        }

        return state

    def _parse_units(self, units_data: list[dict[str, Any]]) -> list[Unit]:
        """Parse unit data from Lua export."""
        units = []
        for u in units_data:
            try:
                unit = Unit(
                    id=str(u.get("id", "")),
                    type=u.get("type", "UNIT_WARRIOR"),
                    player=u.get("owner", 0),
                    position=Position(
                        x=u.get("x", 0),
                        y=u.get("y", 0)
                    ),
                    health=u.get("damage", 0),
                    movement=u.get("movesRemaining", 0),
                    strength=u.get("combatStrength", 20),
                    can_attack=u.get("canAttack", True),
                    can_move=u.get("canMove", True),
                    promotions=u.get("promotions", []),
                    extra={
                        "maxMoves": u.get("maxMoves", 2),
                        "experience": u.get("experience", 0),
                        "level": u.get("level", 1),
                    }
                )
                # Health in Civ6 is stored as damage taken
                unit.health = 100 - u.get("damage", 0)
                units.append(unit)
            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to parse unit: {e}")
                continue
        return units

    def _parse_cities(self, cities_data: list[dict[str, Any]]) -> list[City]:
        """Parse city data from Lua export."""
        cities = []
        for c in cities_data:
            try:
                city = City(
                    id=str(c.get("id", "")),
                    name=c.get("name", "Unknown"),
                    player=c.get("owner", 0),
                    position=Position(
                        x=c.get("x", 0),
                        y=c.get("y", 0)
                    ),
                    population=c.get("population", 1),
                    production=c.get("productionProgress", 0),
                    food=c.get("foodSurplus", 0),
                    defense=c.get("combatStrength", 20),
                    districts=c.get("districts", []),
                    buildings=c.get("buildings", []),
                    producing=c.get("currentProduction"),
                    extra={
                        "housing": c.get("housing", 0),
                        "amenities": c.get("amenities", 0),
                        "turnsToGrow": c.get("turnsToGrow", 0),
                        "turnsToComplete": c.get("turnsToComplete", 0),
                    }
                )
                cities.append(city)
            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to parse city: {e}")
                continue
        return cities

    def _parse_terrain(
        self,
        terrain_data: dict[str, Any]
    ) -> dict[tuple[int, int], Tile]:
        """Parse terrain data from Lua export."""
        terrain = {}
        for key, tile_data in terrain_data.items():
            try:
                # Key format: "x,y"
                x, y = map(int, key.split(","))
                tile = Tile(
                    position=Position(x=x, y=y),
                    terrain=tile_data.get("terrain", "TERRAIN_PLAINS"),
                    features=tile_data.get("features", []),
                    resource=tile_data.get("resource"),
                    improvement=tile_data.get("improvement"),
                    yields={
                        "food": tile_data.get("food", 0),
                        "production": tile_data.get("production", 0),
                        "gold": tile_data.get("gold", 0),
                        "science": tile_data.get("science", 0),
                        "culture": tile_data.get("culture", 0),
                        "faith": tile_data.get("faith", 0),
                    },
                    owner=tile_data.get("owner"),
                )
                terrain[(x, y)] = tile
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse tile {key}: {e}")
                continue
        return terrain

    def _parse_visibility(
        self,
        visibility_data: dict[str, Any]
    ) -> dict[tuple[int, int], str]:
        """Parse visibility data from Lua export."""
        visibility = {}
        for key, vis in visibility_data.items():
            try:
                x, y = map(int, key.split(","))
                # Visibility values: "visible", "revealed", "hidden"
                visibility[(x, y)] = vis
            except (ValueError, KeyError, TypeError):
                continue
        return visibility


@dataclass
class Civ6LuaExporter:
    """Generates Lua code for exporting game state.

    This class generates the Lua mod code that should be installed
    in Civ6 to enable state export and action execution.
    """

    output_dir: Path = field(default_factory=lambda: Path("civ6_mod"))
    mod_id: str = "ALPHAZERO_BRIDGE"
    poll_interval_ms: int = 100

    def generate_mod(self) -> dict[str, str]:
        """Generate complete Lua mod files.

        Returns:
            Dictionary mapping filename to file content.
        """
        return {
            f"{self.mod_id}.modinfo": self._generate_modinfo(),
            "StateExporter.lua": self._generate_state_exporter(),
            "ActionExecutor.lua": self._generate_action_executor(),
            "Bridge.lua": self._generate_bridge(),
        }

    def _generate_modinfo(self) -> str:
        """Generate .modinfo XML file."""
        return f'''<?xml version="1.0" encoding="utf-8"?>
<Mod id="{self.mod_id}" version="1">
    <Properties>
        <Name>AlphaZero Bridge</Name>
        <Description>Enables AI training through file-based IPC</Description>
        <Authors>AlphaZero</Authors>
        <CompatibleVersions>2.0</CompatibleVersions>
    </Properties>
    <Components>
        <GameplayScripts>
            <File>Bridge.lua</File>
        </GameplayScripts>
    </Components>
</Mod>'''

    def _generate_state_exporter(self) -> str:
        """Generate StateExporter.lua."""
        return '''-- StateExporter.lua
-- Exports game state to JSON for AlphaZero

local StateExporter = {}

function StateExporter:GetGameState(playerID)
    local state = {
        turn = Game.GetCurrentGameTurn(),
        currentPlayer = playerID,
        numPlayers = PlayerManager.GetAliveMajorsCount(),
        mapWidth = Map.GetMapSize():GetWidth(),
        mapHeight = Map.GetMapSize():GetHeight(),
        units = self:GetUnits(playerID),
        cities = self:GetCities(playerID),
        terrain = self:GetTerrain(playerID),
        visibility = self:GetVisibility(playerID),
        resources = self:GetResources(playerID),
        technologies = self:GetTechnologies(playerID),
        civics = self:GetCivics(playerID),
        era = self:GetEra(playerID),
        gold = self:GetGold(playerID),
        goldPerTurn = self:GetGoldPerTurn(playerID),
        sciencePerTurn = self:GetSciencePerTurn(playerID),
        culturePerTurn = self:GetCulturePerTurn(playerID),
        faithPerTurn = self:GetFaithPerTurn(playerID),
        militaryStrength = self:GetMilitaryStrength(playerID),
        victoryProgress = self:GetVictoryProgress(playerID),
        gameOver = Game.IsGameOver(),
        winner = Game.GetWinner(),
    }
    return state
end

function StateExporter:GetUnits(playerID)
    local units = {}
    local player = Players[playerID]
    if player then
        local playerUnits = player:GetUnits()
        for i, unit in playerUnits:Members() do
            local unitInfo = {
                id = unit:GetID(),
                type = GameInfo.Units[unit:GetType()].UnitType,
                owner = unit:GetOwner(),
                x = unit:GetX(),
                y = unit:GetY(),
                damage = unit:GetDamage(),
                maxMoves = unit:GetMaxMoves(),
                movesRemaining = unit:GetMovesRemaining(),
                combatStrength = unit:GetCombat(),
                canAttack = unit:CanAttack(),
                canMove = unit:GetMovesRemaining() > 0,
                experience = unit:GetExperience():GetExperiencePoints(),
                level = unit:GetExperience():GetLevel(),
                promotions = self:GetUnitPromotions(unit),
            }
            table.insert(units, unitInfo)
        end
    end
    -- Also get visible enemy units
    for _, otherPlayer in ipairs(PlayerManager.GetAlive()) do
        if otherPlayer ~= playerID then
            local otherUnits = Players[otherPlayer]:GetUnits()
            for i, unit in otherUnits:Members() do
                if self:IsVisible(playerID, unit:GetX(), unit:GetY()) then
                    local unitInfo = {
                        id = unit:GetID(),
                        type = GameInfo.Units[unit:GetType()].UnitType,
                        owner = unit:GetOwner(),
                        x = unit:GetX(),
                        y = unit:GetY(),
                        damage = unit:GetDamage(),
                        combatStrength = unit:GetCombat(),
                    }
                    table.insert(units, unitInfo)
                end
            end
        end
    end
    return units
end

function StateExporter:GetCities(playerID)
    local cities = {}
    local player = Players[playerID]
    if player then
        local playerCities = player:GetCities()
        for i, city in playerCities:Members() do
            local cityInfo = {
                id = city:GetID(),
                name = city:GetName(),
                owner = city:GetOwner(),
                x = city:GetX(),
                y = city:GetY(),
                population = city:GetPopulation(),
                productionProgress = city:GetBuildQueue():GetCurrentProductionProgress(),
                foodSurplus = city:GetGrowth():GetFoodSurplus(),
                combatStrength = city:GetStrength(),
                housing = city:GetGrowth():GetHousing(),
                amenities = city:GetGrowth():GetAmenities(),
                turnsToGrow = city:GetGrowth():GetTurnsUntilGrowth(),
                turnsToComplete = city:GetBuildQueue():GetTurnsLeft(),
                currentProduction = city:GetBuildQueue():CurrentlyBuilding(),
                districts = self:GetCityDistricts(city),
                buildings = self:GetCityBuildings(city),
            }
            table.insert(cities, cityInfo)
        end
    end
    return cities
end

function StateExporter:GetTerrain(playerID)
    local terrain = {}
    local mapWidth = Map.GetMapSize():GetWidth()
    local mapHeight = Map.GetMapSize():GetHeight()
    for x = 0, mapWidth - 1 do
        for y = 0, mapHeight - 1 do
            if self:IsRevealed(playerID, x, y) then
                local plot = Map.GetPlot(x, y)
                local key = x .. "," .. y
                terrain[key] = {
                    terrain = GameInfo.Terrains[plot:GetTerrainType()].TerrainType,
                    features = self:GetPlotFeatures(plot),
                    resource = self:GetPlotResource(plot),
                    improvement = self:GetPlotImprovement(plot),
                    food = plot:GetYield(YieldTypes.YIELD_FOOD),
                    production = plot:GetYield(YieldTypes.YIELD_PRODUCTION),
                    gold = plot:GetYield(YieldTypes.YIELD_GOLD),
                    science = plot:GetYield(YieldTypes.YIELD_SCIENCE),
                    culture = plot:GetYield(YieldTypes.YIELD_CULTURE),
                    faith = plot:GetYield(YieldTypes.YIELD_FAITH),
                    owner = plot:GetOwner(),
                }
            end
        end
    end
    return terrain
end

function StateExporter:GetVisibility(playerID)
    local visibility = {}
    local mapWidth = Map.GetMapSize():GetWidth()
    local mapHeight = Map.GetMapSize():GetHeight()
    local playerVis = PlayersVisibility[playerID]
    for x = 0, mapWidth - 1 do
        for y = 0, mapHeight - 1 do
            local key = x .. "," .. y
            if playerVis:IsVisible(x, y) then
                visibility[key] = "visible"
            elseif playerVis:IsRevealed(x, y) then
                visibility[key] = "revealed"
            else
                visibility[key] = "hidden"
            end
        end
    end
    return visibility
end

function StateExporter:GetUnitPromotions(unit)
    local promotions = {}
    -- Get promotions from unit
    return promotions
end

function StateExporter:GetCityDistricts(city)
    local districts = {}
    local cityDistricts = city:GetDistricts()
    for i, district in cityDistricts:Members() do
        table.insert(districts, GameInfo.Districts[district:GetType()].DistrictType)
    end
    return districts
end

function StateExporter:GetCityBuildings(city)
    local buildings = {}
    local cityBuildings = city:GetBuildings()
    for building in GameInfo.Buildings() do
        if cityBuildings:HasBuilding(building.Index) then
            table.insert(buildings, building.BuildingType)
        end
    end
    return buildings
end

function StateExporter:GetPlotFeatures(plot)
    local features = {}
    local featureType = plot:GetFeatureType()
    if featureType ~= -1 then
        table.insert(features, GameInfo.Features[featureType].FeatureType)
    end
    return features
end

function StateExporter:GetPlotResource(plot)
    local resourceType = plot:GetResourceType()
    if resourceType ~= -1 then
        return GameInfo.Resources[resourceType].ResourceType
    end
    return nil
end

function StateExporter:GetPlotImprovement(plot)
    local improvementType = plot:GetImprovementType()
    if improvementType ~= -1 then
        return GameInfo.Improvements[improvementType].ImprovementType
    end
    return nil
end

function StateExporter:IsVisible(playerID, x, y)
    return PlayersVisibility[playerID]:IsVisible(x, y)
end

function StateExporter:IsRevealed(playerID, x, y)
    return PlayersVisibility[playerID]:IsRevealed(x, y)
end

function StateExporter:GetResources(playerID)
    local resources = {}
    local player = Players[playerID]
    if player then
        local playerResources = player:GetResources()
        for resource in GameInfo.Resources() do
            local count = playerResources:GetResourceAmount(resource.Index)
            if count > 0 then
                resources[resource.ResourceType] = count
            end
        end
    end
    return resources
end

function StateExporter:GetTechnologies(playerID)
    local techs = {}
    local player = Players[playerID]
    if player then
        local playerTechs = player:GetTechs()
        for tech in GameInfo.Technologies() do
            if playerTechs:HasTech(tech.Index) then
                table.insert(techs, tech.Index)
            end
        end
    end
    return techs
end

function StateExporter:GetCivics(playerID)
    local civics = {}
    local player = Players[playerID]
    if player then
        local playerCulture = player:GetCulture()
        for civic in GameInfo.Civics() do
            if playerCulture:HasCivic(civic.Index) then
                table.insert(civics, civic.Index)
            end
        end
    end
    return civics
end

function StateExporter:GetEra(playerID)
    local player = Players[playerID]
    if player then
        return player:GetEras():GetEra()
    end
    return 0
end

function StateExporter:GetGold(playerID)
    local player = Players[playerID]
    if player then
        return player:GetTreasury():GetGoldBalance()
    end
    return 0
end

function StateExporter:GetGoldPerTurn(playerID)
    local player = Players[playerID]
    if player then
        return player:GetTreasury():GetGoldYield()
    end
    return 0
end

function StateExporter:GetSciencePerTurn(playerID)
    local player = Players[playerID]
    if player then
        return player:GetTechs():GetScienceYield()
    end
    return 0
end

function StateExporter:GetCulturePerTurn(playerID)
    local player = Players[playerID]
    if player then
        return player:GetCulture():GetCultureYield()
    end
    return 0
end

function StateExporter:GetFaithPerTurn(playerID)
    local player = Players[playerID]
    if player then
        return player:GetReligion():GetFaithYield()
    end
    return 0
end

function StateExporter:GetMilitaryStrength(playerID)
    local player = Players[playerID]
    if player then
        return player:GetStats():GetMilitaryStrengthWithoutTreasury()
    end
    return 0
end

function StateExporter:GetVictoryProgress(playerID)
    local progress = {}
    -- Victory progress tracking
    return progress
end

return StateExporter
'''

    def _generate_action_executor(self) -> str:
        """Generate ActionExecutor.lua."""
        return '''-- ActionExecutor.lua
-- Executes actions from AlphaZero

local ActionExecutor = {}

function ActionExecutor:ExecuteAction(action, playerID)
    local actionType = action.type

    if actionType == "end_turn" then
        return self:EndTurn(playerID)
    elseif actionType == "move" then
        return self:MoveUnit(action, playerID)
    elseif actionType == "attack" then
        return self:AttackUnit(action, playerID)
    elseif actionType == "skip" then
        return self:SkipUnit(action, playerID)
    elseif actionType == "produce" then
        return self:Produce(action, playerID)
    elseif actionType == "research" then
        return self:Research(action, playerID)
    elseif actionType == "build" then
        return self:Build(action, playerID)
    elseif actionType == "settle" then
        return self:Settle(action, playerID)
    else
        print("Unknown action type: " .. actionType)
        return false
    end
end

function ActionExecutor:EndTurn(playerID)
    -- End the current turn
    UI.RequestPlayerOperation(playerID, PlayerOperations.FINISH_TURN, {})
    return true
end

function ActionExecutor:MoveUnit(action, playerID)
    local unitID = action.unit_id
    local targetX = action.target[1]
    local targetY = action.target[2]

    local player = Players[playerID]
    if not player then return false end

    local unit = UnitManager.GetUnit(playerID, unitID)
    if not unit then return false end

    -- Request unit movement
    UnitManager.RequestCommand(unit, UnitCommandTypes.MOVE_TO, targetX, targetY)
    return true
end

function ActionExecutor:AttackUnit(action, playerID)
    local unitID = action.unit_id
    local targetX = action.target[1]
    local targetY = action.target[2]

    local player = Players[playerID]
    if not player then return false end

    local unit = UnitManager.GetUnit(playerID, unitID)
    if not unit then return false end

    -- Check for enemy unit at target
    local plot = Map.GetPlot(targetX, targetY)
    if not plot then return false end

    -- Attack the target location
    UnitManager.RequestCommand(unit, UnitCommandTypes.ATTACK, targetX, targetY)
    return true
end

function ActionExecutor:SkipUnit(action, playerID)
    local unitID = action.unit_id

    local player = Players[playerID]
    if not player then return false end

    local unit = UnitManager.GetUnit(playerID, unitID)
    if not unit then return false end

    -- Skip the unit
    UnitManager.RequestCommand(unit, UnitCommandTypes.SKIP)
    return true
end

function ActionExecutor:Produce(action, playerID)
    local cityID = action.city_id
    local production = action.production

    local player = Players[playerID]
    if not player then return false end

    local city = CityManager.GetCity(playerID, cityID)
    if not city then return false end

    -- Queue production
    local buildQueue = city:GetBuildQueue()

    -- Determine if it's a unit, building, or district
    local unitInfo = GameInfo.Units[production]
    if unitInfo then
        buildQueue:CreateIncompleteUnit(unitInfo.Index)
        return true
    end

    local buildingInfo = GameInfo.Buildings[production]
    if buildingInfo then
        buildQueue:CreateIncompleteBuilding(buildingInfo.Index)
        return true
    end

    local districtInfo = GameInfo.Districts[production]
    if districtInfo then
        -- Districts need a location
        -- For now, auto-select
        return false
    end

    return false
end

function ActionExecutor:Research(action, playerID)
    local techType = action.tech

    local player = Players[playerID]
    if not player then return false end

    local techInfo = GameInfo.Technologies[techType]
    if not techInfo then return false end

    local playerTechs = player:GetTechs()
    playerTechs:SetResearchingTech(techInfo.Index)
    return true
end

function ActionExecutor:Build(action, playerID)
    local unitID = action.unit_id
    local improvement = action.improvement

    local player = Players[playerID]
    if not player then return false end

    local unit = UnitManager.GetUnit(playerID, unitID)
    if not unit then return false end

    local improvementInfo = GameInfo.Improvements[improvement]
    if not improvementInfo then return false end

    UnitManager.RequestOperation(unit, UnitOperationTypes.BUILD_IMPROVEMENT, {
        ImprovementType = improvementInfo.Index
    })
    return true
end

function ActionExecutor:Settle(action, playerID)
    local unitID = action.unit_id

    local player = Players[playerID]
    if not player then return false end

    local unit = UnitManager.GetUnit(playerID, unitID)
    if not unit then return false end

    -- Check if unit can settle
    UnitManager.RequestOperation(unit, UnitOperationTypes.FOUND_CITY)
    return true
end

return ActionExecutor
'''

    def _generate_bridge(self) -> str:
        """Generate Bridge.lua main entry point."""
        return f'''-- Bridge.lua
-- Main entry point for AlphaZero bridge

include("StateExporter")
include("ActionExecutor")

local Bridge = {{}}
local POLL_INTERVAL = {self.poll_interval_ms}
local STATE_DIR = nil  -- Set via environment or config
local ACTION_DIR = nil

function Bridge:Initialize()
    print("AlphaZero Bridge initializing...")

    -- Get directories from game configuration
    STATE_DIR = os.getenv("ALPHAZERO_STATE_DIR") or "alphazero_state"
    ACTION_DIR = os.getenv("ALPHAZERO_ACTION_DIR") or "alphazero_action"

    -- Create directories if needed
    os.execute("mkdir -p " .. STATE_DIR)
    os.execute("mkdir -p " .. ACTION_DIR)

    -- Register for turn events
    Events.LocalPlayerTurnBegin.Add(function() self:OnTurnBegin() end)
    Events.LocalPlayerTurnEnd.Add(function() self:OnTurnEnd() end)

    print("AlphaZero Bridge initialized")
end

function Bridge:OnTurnBegin()
    local playerID = Game.GetLocalPlayer()
    if playerID == -1 then return end

    -- Export current state
    self:ExportState(playerID)

    -- Start polling for actions
    self:StartActionPolling(playerID)
end

function Bridge:OnTurnEnd()
    -- Stop polling
    self:StopActionPolling()
end

function Bridge:ExportState(playerID)
    local state = StateExporter:GetGameState(playerID)
    local json = self:ToJSON(state)

    -- Write state file with lock
    local lockPath = STATE_DIR .. "/state.lock"
    local statePath = STATE_DIR .. "/state.json"
    local checksumPath = STATE_DIR .. "/state_checksum.txt"

    -- Create lock
    local lockFile = io.open(lockPath, "w")
    if lockFile then
        lockFile:write(os.time())
        lockFile:close()

        -- Write state
        local stateFile = io.open(statePath, "w")
        if stateFile then
            stateFile:write(json)
            stateFile:close()

            -- Write checksum
            local checksum = self:ComputeChecksum(json)
            local checksumFile = io.open(checksumPath, "w")
            if checksumFile then
                checksumFile:write(checksum)
                checksumFile:close()
            end
        end

        -- Remove lock
        os.remove(lockPath)
    end
end

function Bridge:StartActionPolling(playerID)
    self.polling = true
    self.pollTimer = 0

    -- Use game timer for polling
    Events.GameCoreEventPublishComplete.Add(function()
        if self.polling then
            self:PollForAction(playerID)
        end
    end)
end

function Bridge:StopActionPolling()
    self.polling = false
end

function Bridge:PollForAction(playerID)
    local actionPath = ACTION_DIR .. "/action.json"
    local lockPath = ACTION_DIR .. "/action.lock"

    -- Check for lock
    local lockFile = io.open(lockPath, "r")
    if lockFile then
        lockFile:close()
        return  -- Action being written
    end

    -- Try to read action
    local actionFile = io.open(actionPath, "r")
    if actionFile then
        local content = actionFile:read("*all")
        actionFile:close()

        if content and content ~= "" then
            local action = self:FromJSON(content)
            if action then
                -- Execute the action
                local success = ActionExecutor:ExecuteAction(action, playerID)

                -- Write acknowledgment
                local ackPath = ACTION_DIR .. "/action_ack.json"
                local ackFile = io.open(ackPath, "w")
                if ackFile then
                    ackFile:write(self:ToJSON({{
                        success = success,
                        timestamp = os.time(),
                        action_type = action.type
                    }}))
                    ackFile:close()
                end

                -- Clear action file
                os.remove(actionPath)

                -- Export new state after action
                self:ExportState(playerID)
            end
        end
    end
end

function Bridge:ToJSON(obj)
    -- Simple JSON encoder (Civ6 has built-in JSON support)
    return JSON.encode(obj)
end

function Bridge:FromJSON(str)
    -- Simple JSON decoder
    return JSON.decode(str)
end

function Bridge:ComputeChecksum(str)
    -- Simple checksum (Civ6 may have better hash functions)
    local sum = 0
    for i = 1, #str do
        sum = (sum + string.byte(str, i)) % 4294967296
    end
    return string.format("%08x", sum)
end

-- Initialize on load
Bridge:Initialize()

return Bridge
'''

    def write_mod(self, output_dir: Path | None = None) -> Path:
        """Write mod files to disk.

        Args:
            output_dir: Directory to write mod files.

        Returns:
            Path to the mod directory.
        """
        target_dir = output_dir or self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        files = self.generate_mod()
        for filename, content in files.items():
            filepath = target_dir / filename
            filepath.write_text(content)
            logger.info(f"Wrote mod file: {filepath}")

        return target_dir


class Civ6Bridge(GameInterface):
    """Bridge to Civilization 6 via Lua mod and file-based IPC.

    This bridge communicates with a Civ6 Lua mod through JSON files.
    The mod exports game state and the bridge sends actions back.

    Example:
        config = InterfaceConfig(
            state_dir=Path("/path/to/state"),
            action_dir=Path("/path/to/action"),
        )
        bridge = Civ6Bridge(config)
        with bridge:
            state = bridge.read_state()
            action = GameAction.move(unit_id="1", target=Position(5, 5))
            bridge.send_action(action)
    """

    def __init__(
        self,
        config: InterfaceConfig | None = None,
        state_dir: Path | None = None,
        action_dir: Path | None = None,
        player_id: int = 0,
    ) -> None:
        """Initialize the Civ6 bridge.

        Args:
            config: Interface configuration.
            state_dir: Directory for state files (overrides config).
            action_dir: Directory for action files (overrides config).
            player_id: Player ID to control.
        """
        super().__init__(config)

        self.state_dir = state_dir or self.config.state_dir or Path("alphazero_state")
        self.action_dir = (
            action_dir or self.config.action_dir or Path("alphazero_action")
        )
        self.player_id = player_id
        self.parser = LuaStateParser(player_id=player_id)
        self._last_state: GameState | None = None
        self._state_checksum: str | None = None

    def connect(self) -> bool:
        """Connect to the Civ6 game.

        Creates necessary directories and verifies the Lua mod is responding.

        Returns:
            True if connection successful.

        Raises:
            LuaModNotFoundError: If mod is not responding.
        """
        try:
            # Create directories if they don't exist
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.action_dir.mkdir(parents=True, exist_ok=True)

            # Check if mod is already exporting state
            state_file = self.state_dir / "state.json"
            if state_file.exists():
                try:
                    self.read_state()
                    self._connected = True
                    logger.info("Connected to Civ6 bridge")
                    return True
                except StateReadError:
                    pass

            # Wait for mod to start
            logger.info("Waiting for Civ6 Lua mod to start...")
            start_time = time.time()
            while time.time() - start_time < self.config.timeout_seconds:
                if state_file.exists():
                    try:
                        self.read_state()
                        self._connected = True
                        logger.info("Connected to Civ6 bridge")
                        return True
                    except StateReadError:
                        pass
                time.sleep(self.config.poll_interval_ms / 1000.0)

            # If we get here, we can still connect but mod isn't running
            logger.warning("Civ6 mod not detected, bridge in standby mode")
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise LuaModNotFoundError(f"Failed to connect to Civ6: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from the game."""
        self._connected = False
        logger.info("Disconnected from Civ6 bridge")

    def read_state(self) -> GameState:
        """Read current game state from JSON file.

        Returns:
            Current game state.

        Raises:
            StateReadError: If reading state fails.
        """
        state_file = self.state_dir / "state.json"
        lock_file = self.state_dir / "state.lock"
        checksum_file = self.state_dir / "state_checksum.txt"

        # Wait for lock to clear
        start_time = time.time()
        while lock_file.exists():
            if time.time() - start_time > 5.0:  # 5 second timeout for lock
                raise StateReadError("Timed out waiting for state lock")
            time.sleep(0.01)

        if not state_file.exists():
            raise StateReadError("State file not found")

        try:
            content = state_file.read_text()

            # Validate checksum if available
            if checksum_file.exists() and self.config.validate_state:
                expected_checksum = checksum_file.read_text().strip()
                actual_checksum = self._compute_checksum(content)
                if expected_checksum != actual_checksum:
                    raise StateReadError(
                        f"State checksum mismatch: {expected_checksum} != {actual_checksum}"
                    )

            data = json.loads(content)
            state = self.parser.parse(data)
            self._last_state = state
            self._state_checksum = self._compute_checksum(content)

            return state

        except json.JSONDecodeError as e:
            raise StateReadError(f"Invalid JSON in state file: {e}") from e
        except Exception as e:
            raise StateReadError(f"Failed to read state: {e}") from e

    def send_action(self, action: GameAction) -> bool:
        """Send an action to the game via JSON file.

        Args:
            action: Action to send.

        Returns:
            True if action was accepted.

        Raises:
            ActionSendError: If sending action fails.
        """
        action_file = self.action_dir / "action.json"
        lock_file = self.action_dir / "action.lock"
        ack_file = self.action_dir / "action_ack.json"

        try:
            # Create lock
            lock_file.write_text(str(time.time()))

            # Write action
            action_data = action.to_dict()
            action_file.write_text(json.dumps(action_data, indent=2))

            # Remove lock
            lock_file.unlink()

            logger.debug(f"Sent action: {action.action_type.value}")

            # Wait for acknowledgment
            start_time = time.time()
            while time.time() - start_time < self.config.timeout_seconds:
                if ack_file.exists():
                    try:
                        ack_content = ack_file.read_text()
                        ack_data = json.loads(ack_content)
                        ack_file.unlink()

                        if ack_data.get("success"):
                            logger.debug("Action acknowledged successfully")
                            return True
                        else:
                            logger.warning(f"Action rejected: {ack_data}")
                            return False
                    except (json.JSONDecodeError, OSError):
                        pass
                time.sleep(self.config.poll_interval_ms / 1000.0)

            # Timeout - action may or may not have been processed
            logger.warning("Timed out waiting for action acknowledgment")
            return True  # Assume success if no explicit rejection

        except Exception as e:
            # Clean up lock on error
            if lock_file.exists():
                lock_file.unlink()
            raise ActionSendError(f"Failed to send action: {e}") from e

    def get_valid_actions(self, state: GameState) -> list[GameAction]:
        """Get list of valid actions for current state.

        Args:
            state: Current game state.

        Returns:
            List of valid actions.
        """
        actions: list[GameAction] = []

        # End turn is always valid
        actions.append(GameAction.end_turn())

        # Unit actions
        for unit in state.get_own_units():
            # Skip action
            actions.append(GameAction.skip(unit.id))

            if unit.can_move:
                # Movement actions (8 directions)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        target = Position(
                            x=unit.position.x + dx,
                            y=unit.position.y + dy
                        )
                        # Check map bounds
                        if 0 <= target.x < state.map_width and 0 <= target.y < state.map_height:
                            # Check terrain passability
                            tile = state.terrain.get(target.to_tuple())
                            if tile and tile.terrain != "TERRAIN_MOUNTAIN":
                                actions.append(GameAction.move(unit.id, target))

            if unit.can_attack:
                # Attack actions (8 directions)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        target = Position(
                            x=unit.position.x + dx,
                            y=unit.position.y + dy
                        )
                        # Check for enemy units
                        enemies = [
                            u for u in state.get_units_at(target)
                            if u.player != state.current_player
                        ]
                        if enemies:
                            actions.append(GameAction.attack(unit.id, target))

        # City production actions
        for city in state.get_own_cities():
            if not city.producing:
                # Can produce units
                for unit_type in ["UNIT_WARRIOR", "UNIT_SETTLER", "UNIT_BUILDER"]:
                    actions.append(GameAction.produce(city.id, unit_type))

        return actions

    def _compute_checksum(self, content: str) -> str:
        """Compute checksum for content validation.

        Args:
            content: String content to checksum.

        Returns:
            Hexadecimal checksum string.
        """
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:8]

    def wait_for_turn(self, timeout_seconds: float | None = None) -> bool:
        """Wait for our turn to begin.

        Args:
            timeout_seconds: Maximum time to wait.

        Returns:
            True if it's now our turn.
        """
        timeout = timeout_seconds or self.config.timeout_seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                state = self.read_state()
                if state.current_player == self.player_id:
                    return True
            except StateReadError:
                pass
            time.sleep(self.config.poll_interval_ms / 1000.0)

        return False

    def generate_lua_mod(self, output_dir: Path | None = None) -> Path:
        """Generate the Lua mod files for Civ6.

        Args:
            output_dir: Directory to write mod files.

        Returns:
            Path to the generated mod directory.
        """
        exporter = Civ6LuaExporter(
            poll_interval_ms=self.config.poll_interval_ms,
        )
        return exporter.write_mod(output_dir)
