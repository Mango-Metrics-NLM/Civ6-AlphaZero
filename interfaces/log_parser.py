"""Civ6 Log File Parser.

Provides passive observation of Civilization 6 games by parsing log files.
This is useful for learning from human gameplay without active game control.

Log Locations:
    Windows: %LOCALAPPDATA%\\Firaxis Games\\Sid Meier's Civilization VI\\Logs
    Linux: ~/.local/share/Aspyr/Sid Meier's Civilization VI/Logs

Log Files:
    - Lua.log: Lua script output and errors
    - Database.log: Database queries and game data
    - AI_Log.log: AI decision making
    - Gameplay.log: Game events and actions
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .base import (
    GameAction,
    GameState,
    InterfaceConfig,
    InterfaceError,
    Position,
)

logger = logging.getLogger(__name__)


class LogParseError(InterfaceError):
    """Error parsing log file."""
    pass


class LogEventType(Enum):
    """Types of log events."""
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    UNIT_MOVE = "unit_move"
    UNIT_ATTACK = "unit_attack"
    UNIT_CREATED = "unit_created"
    UNIT_DESTROYED = "unit_destroyed"
    CITY_FOUNDED = "city_founded"
    CITY_CAPTURED = "city_captured"
    CITY_PRODUCTION = "city_production"
    TECH_RESEARCHED = "tech_researched"
    CIVIC_COMPLETED = "civic_completed"
    WAR_DECLARED = "war_declared"
    PEACE_MADE = "peace_made"
    GREAT_PERSON = "great_person"
    WONDER_BUILT = "wonder_built"
    UNKNOWN = "unknown"


@dataclass
class LogEntry:
    """A parsed log entry."""
    timestamp: datetime
    event_type: LogEventType
    player_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""

    def __str__(self) -> str:
        return f"[{self.timestamp}] {self.event_type.value}: {self.data}"


@dataclass
class MovementLog:
    """Parsed unit movement event."""
    unit_id: str
    unit_type: str
    player_id: int
    from_position: Position
    to_position: Position
    timestamp: datetime

    @classmethod
    def from_log_entry(cls, entry: LogEntry) -> MovementLog | None:
        """Create from log entry if applicable."""
        if entry.event_type != LogEventType.UNIT_MOVE:
            return None

        data = entry.data
        try:
            return cls(
                unit_id=data.get("unit_id", ""),
                unit_type=data.get("unit_type", ""),
                player_id=data.get("player_id", 0),
                from_position=Position(
                    x=data.get("from_x", 0),
                    y=data.get("from_y", 0)
                ),
                to_position=Position(
                    x=data.get("to_x", 0),
                    y=data.get("to_y", 0)
                ),
                timestamp=entry.timestamp,
            )
        except (KeyError, TypeError):
            return None


@dataclass
class CombatLog:
    """Parsed combat event."""
    attacker_id: str
    attacker_type: str
    attacker_player: int
    defender_id: str
    defender_type: str
    defender_player: int
    position: Position
    attacker_damage: int
    defender_damage: int
    defender_destroyed: bool
    timestamp: datetime

    @classmethod
    def from_log_entry(cls, entry: LogEntry) -> CombatLog | None:
        """Create from log entry if applicable."""
        if entry.event_type != LogEventType.UNIT_ATTACK:
            return None

        data = entry.data
        try:
            return cls(
                attacker_id=data.get("attacker_id", ""),
                attacker_type=data.get("attacker_type", ""),
                attacker_player=data.get("attacker_player", 0),
                defender_id=data.get("defender_id", ""),
                defender_type=data.get("defender_type", ""),
                defender_player=data.get("defender_player", 0),
                position=Position(
                    x=data.get("x", 0),
                    y=data.get("y", 0)
                ),
                attacker_damage=data.get("attacker_damage", 0),
                defender_damage=data.get("defender_damage", 0),
                defender_destroyed=data.get("defender_destroyed", False),
                timestamp=entry.timestamp,
            )
        except (KeyError, TypeError):
            return None


class Civ6LogParser:
    """Parser for Civilization 6 log files.

    Watches and parses log files to extract game events for learning
    from human gameplay.

    Example:
        parser = Civ6LogParser(log_dir=Path("/path/to/logs"))
        parser.start_watching()

        for event in parser.events():
            if event.event_type == LogEventType.UNIT_MOVE:
                movement = MovementLog.from_log_entry(event)
                # Process movement...

        parser.stop_watching()
    """

    # Log file patterns for different event types
    PATTERNS = {
        "turn_start": re.compile(
            r"\[(\d+:\d+:\d+)\].*Turn (\d+) has started for player (\d+)"
        ),
        "turn_end": re.compile(
            r"\[(\d+:\d+:\d+)\].*Player (\d+) ended their turn"
        ),
        "unit_move": re.compile(
            r"\[(\d+:\d+:\d+)\].*Unit (\w+) \((\d+)\) moved from \((\d+),(\d+)\) to \((\d+),(\d+)\)"
        ),
        "unit_attack": re.compile(
            r"\[(\d+:\d+:\d+)\].*Unit (\w+) \((\d+)\) attacked (\w+) at \((\d+),(\d+)\)"
        ),
        "unit_created": re.compile(
            r"\[(\d+:\d+:\d+)\].*Unit (\w+) created for player (\d+) at \((\d+),(\d+)\)"
        ),
        "unit_destroyed": re.compile(
            r"\[(\d+:\d+:\d+)\].*Unit (\w+) \((\d+)\) was destroyed"
        ),
        "city_founded": re.compile(
            r"\[(\d+:\d+:\d+)\].*City (\w+) founded by player (\d+) at \((\d+),(\d+)\)"
        ),
        "city_production": re.compile(
            r"\[(\d+:\d+:\d+)\].*City (\w+) began producing (\w+)"
        ),
        "tech_researched": re.compile(
            r"\[(\d+:\d+:\d+)\].*Player (\d+) completed research: (\w+)"
        ),
    }

    def __init__(
        self,
        config: InterfaceConfig | None = None,
        log_dir: Path | None = None,
        log_file: str = "Gameplay.log",
        poll_interval_ms: int = 100,
    ) -> None:
        """Initialize the log parser.

        Args:
            config: Interface configuration.
            log_dir: Directory containing log files.
            log_file: Name of the log file to parse.
            poll_interval_ms: Polling interval for file changes.
        """
        self.config = config or InterfaceConfig()

        # Determine log directory
        if log_dir:
            self.log_dir = log_dir
        elif self.config.log_dir:
            self.log_dir = self.config.log_dir
        else:
            self.log_dir = self._find_log_directory()

        self.log_file = log_file
        self.poll_interval_ms = poll_interval_ms

        self._watching = False
        self._file_position = 0
        self._event_queue: list[LogEntry] = []
        self._callbacks: list[Callable[[LogEntry], None]] = []
        self._current_turn = 0
        self._current_player = 0

    def _find_log_directory(self) -> Path:
        """Find the Civ6 log directory based on platform."""
        import platform

        system = platform.system()

        if system == "Windows":
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            return Path(local_app_data) / "Firaxis Games" / "Sid Meier's Civilization VI" / "Logs"
        elif system == "Linux":
            home = Path.home()
            return home / ".local" / "share" / "Aspyr" / "Sid Meier's Civilization VI" / "Logs"
        elif system == "Darwin":  # macOS
            home = Path.home()
            return home / "Library" / "Application Support" / "Civilization VI" / "Logs"
        else:
            return Path(".")

    @property
    def log_path(self) -> Path:
        """Full path to log file."""
        return self.log_dir / self.log_file

    def start_watching(self) -> None:
        """Start watching the log file for new entries."""
        if not self.log_path.exists():
            logger.warning(f"Log file not found: {self.log_path}")

        # Start from end of file
        if self.log_path.exists():
            self._file_position = self.log_path.stat().st_size

        self._watching = True
        logger.info(f"Started watching: {self.log_path}")

    def stop_watching(self) -> None:
        """Stop watching the log file."""
        self._watching = False
        logger.info("Stopped watching log file")

    def events(self) -> Iterator[LogEntry]:
        """Iterate over new log events.

        Yields:
            LogEntry for each new event.
        """
        while self._watching:
            # Check for new content
            new_events = self._read_new_entries()
            yield from new_events

            if not new_events:
                time.sleep(self.poll_interval_ms / 1000.0)

    def add_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """Add a callback for new events.

        Args:
            callback: Function to call with each new event.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """Remove a callback.

        Args:
            callback: Callback to remove.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def parse_line(self, line: str) -> LogEntry | None:
        """Parse a single log line.

        Args:
            line: Raw log line.

        Returns:
            LogEntry if line matches a pattern.
        """
        line = line.strip()
        if not line:
            return None

        for event_name, pattern in self.PATTERNS.items():
            match = pattern.search(line)
            if match:
                return self._create_entry(event_name, match, line)

        return None

    def parse_file(self, filepath: Path | None = None) -> list[LogEntry]:
        """Parse an entire log file.

        Args:
            filepath: Path to log file (defaults to configured path).

        Returns:
            List of parsed log entries.
        """
        path = filepath or self.log_path
        if not path.exists():
            return []

        entries = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                entry = self.parse_line(line)
                if entry:
                    entries.append(entry)

        return entries

    def get_game_replay(
        self,
        filepath: Path | None = None
    ) -> list[tuple[GameState, GameAction]]:
        """Extract state-action pairs from log file.

        This is useful for supervised learning from human games.

        Args:
            filepath: Path to log file.

        Returns:
            List of (state, action) pairs.
        """
        entries = self.parse_file(filepath)
        replay: list[tuple[GameState, GameAction]] = []

        current_state = GameState()

        for entry in entries:
            # Update state based on event
            if entry.event_type == LogEventType.TURN_START:
                current_state.turn = entry.data.get("turn", current_state.turn)
                current_state.current_player = entry.data.get("player_id", 0)

            elif entry.event_type == LogEventType.UNIT_MOVE:
                # Extract action
                action = GameAction.move(
                    unit_id=entry.data.get("unit_id", ""),
                    target=Position(
                        x=entry.data.get("to_x", 0),
                        y=entry.data.get("to_y", 0)
                    ),
                )
                replay.append((self._copy_state(current_state), action))

                # Update state
                self._apply_movement(current_state, entry.data)

            elif entry.event_type == LogEventType.UNIT_ATTACK:
                action = GameAction.attack(
                    unit_id=entry.data.get("attacker_id", ""),
                    target=Position(
                        x=entry.data.get("x", 0),
                        y=entry.data.get("y", 0)
                    ),
                )
                replay.append((self._copy_state(current_state), action))

                # Update state
                self._apply_combat(current_state, entry.data)

            elif entry.event_type == LogEventType.CITY_PRODUCTION:
                action = GameAction.produce(
                    city_id=entry.data.get("city_name", ""),
                    production_type=entry.data.get("production", ""),
                )
                replay.append((self._copy_state(current_state), action))

        return replay

    def get_movement_data(
        self,
        filepath: Path | None = None
    ) -> list[MovementLog]:
        """Extract all movement events from log.

        Args:
            filepath: Path to log file.

        Returns:
            List of movement events.
        """
        entries = self.parse_file(filepath)
        movements = []

        for entry in entries:
            movement = MovementLog.from_log_entry(entry)
            if movement:
                movements.append(movement)

        return movements

    def get_combat_data(
        self,
        filepath: Path | None = None
    ) -> list[CombatLog]:
        """Extract all combat events from log.

        Args:
            filepath: Path to log file.

        Returns:
            List of combat events.
        """
        entries = self.parse_file(filepath)
        combats = []

        for entry in entries:
            combat = CombatLog.from_log_entry(entry)
            if combat:
                combats.append(combat)

        return combats

    def _read_new_entries(self) -> list[LogEntry]:
        """Read new entries since last check."""
        if not self.log_path.exists():
            return []

        entries = []
        try:
            current_size = self.log_path.stat().st_size

            if current_size < self._file_position:
                # File was truncated (new game?)
                self._file_position = 0

            if current_size > self._file_position:
                with open(self.log_path, encoding="utf-8", errors="ignore") as f:
                    f.seek(self._file_position)
                    for line in f:
                        entry = self.parse_line(line)
                        if entry:
                            entries.append(entry)
                            # Call callbacks
                            for callback in self._callbacks:
                                try:
                                    callback(entry)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")

                    self._file_position = f.tell()

        except Exception as e:
            logger.error(f"Error reading log file: {e}")

        return entries

    def _create_entry(
        self,
        event_name: str,
        match: re.Match,
        raw_line: str
    ) -> LogEntry:
        """Create a LogEntry from a regex match."""
        groups = match.groups()

        # Parse timestamp
        time_str = groups[0] if groups else "00:00:00"
        try:
            timestamp = datetime.strptime(time_str, "%H:%M:%S")
            # Add today's date
            today = datetime.now().date()
            timestamp = datetime.combine(today, timestamp.time())
        except ValueError:
            timestamp = datetime.now()

        # Map event type
        event_type_map = {
            "turn_start": LogEventType.TURN_START,
            "turn_end": LogEventType.TURN_END,
            "unit_move": LogEventType.UNIT_MOVE,
            "unit_attack": LogEventType.UNIT_ATTACK,
            "unit_created": LogEventType.UNIT_CREATED,
            "unit_destroyed": LogEventType.UNIT_DESTROYED,
            "city_founded": LogEventType.CITY_FOUNDED,
            "city_production": LogEventType.CITY_PRODUCTION,
            "tech_researched": LogEventType.TECH_RESEARCHED,
        }
        event_type = event_type_map.get(event_name, LogEventType.UNKNOWN)

        # Extract data based on event type
        data = self._extract_event_data(event_name, groups)

        return LogEntry(
            timestamp=timestamp,
            event_type=event_type,
            player_id=data.get("player_id"),
            data=data,
            raw_line=raw_line,
        )

    def _extract_event_data(
        self,
        event_name: str,
        groups: tuple
    ) -> dict[str, Any]:
        """Extract event-specific data from regex groups."""
        data: dict[str, Any] = {}

        if event_name == "turn_start" and len(groups) >= 3:
            data["turn"] = int(groups[1])
            data["player_id"] = int(groups[2])

        elif event_name == "turn_end" and len(groups) >= 2:
            data["player_id"] = int(groups[1])

        elif event_name == "unit_move" and len(groups) >= 7:
            data["unit_type"] = groups[1]
            data["unit_id"] = groups[2]
            data["from_x"] = int(groups[3])
            data["from_y"] = int(groups[4])
            data["to_x"] = int(groups[5])
            data["to_y"] = int(groups[6])

        elif event_name == "unit_attack" and len(groups) >= 6:
            data["attacker_type"] = groups[1]
            data["attacker_id"] = groups[2]
            data["defender_type"] = groups[3]
            data["x"] = int(groups[4])
            data["y"] = int(groups[5])

        elif event_name == "unit_created" and len(groups) >= 5:
            data["unit_type"] = groups[1]
            data["player_id"] = int(groups[2])
            data["x"] = int(groups[3])
            data["y"] = int(groups[4])

        elif event_name == "unit_destroyed" and len(groups) >= 3:
            data["unit_type"] = groups[1]
            data["unit_id"] = groups[2]

        elif event_name == "city_founded" and len(groups) >= 5:
            data["city_name"] = groups[1]
            data["player_id"] = int(groups[2])
            data["x"] = int(groups[3])
            data["y"] = int(groups[4])

        elif event_name == "city_production" and len(groups) >= 3:
            data["city_name"] = groups[1]
            data["production"] = groups[2]

        elif event_name == "tech_researched" and len(groups) >= 3:
            data["player_id"] = int(groups[1])
            data["tech"] = groups[2]

        return data

    def _copy_state(self, state: GameState) -> GameState:
        """Create a deep copy of game state."""
        import copy
        return copy.deepcopy(state)

    def _apply_movement(self, state: GameState, data: dict[str, Any]) -> None:
        """Apply movement to state."""
        unit_id = data.get("unit_id", "")
        to_x = data.get("to_x", 0)
        to_y = data.get("to_y", 0)

        for unit in state.units:
            if unit.id == unit_id:
                unit.position = Position(x=to_x, y=to_y)
                break

    def _apply_combat(self, state: GameState, data: dict[str, Any]) -> None:
        """Apply combat results to state."""
        # Simplified combat - just update positions
        pass
