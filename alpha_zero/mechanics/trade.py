"""Trade Route System for Civ6-AlphaZero.

This module implements a comprehensive trade system including:
- Trade route management and pathfinding
- Yield calculations based on distance and destination
- Trade unit tracking and automation
- Trade agreement negotiation

All mechanics are designed to be:
- Fully configurable via dataclasses
- Testable in isolation
- Compatible with the existing encoder system
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TradeRouteType(Enum):
    """Types of trade routes."""

    DOMESTIC = auto()  # Between own cities
    INTERNATIONAL = auto()  # To foreign cities
    CITY_STATE = auto()  # To city-states


class TradeGoodType(Enum):
    """Types of trade goods."""

    FOOD = auto()
    PRODUCTION = auto()
    GOLD = auto()
    SCIENCE = auto()
    CULTURE = auto()
    FAITH = auto()
    TOURISM = auto()


class TraderStatus(Enum):
    """Status of a trade unit."""

    IDLE = auto()
    TRAVELING = auto()
    TRADING = auto()
    RETURNING = auto()


@dataclass
class TradeYields:
    """Yields from a trade route.

    Attributes:
        food: Food per turn
        production: Production per turn
        gold: Gold per turn
        science: Science per turn
        culture: Culture per turn
        faith: Faith per turn
    """

    food: float = 0.0
    production: float = 0.0
    gold: float = 0.0
    science: float = 0.0
    culture: float = 0.0
    faith: float = 0.0

    def __add__(self, other: TradeYields) -> TradeYields:
        """Add two yield objects."""
        return TradeYields(
            food=self.food + other.food,
            production=self.production + other.production,
            gold=self.gold + other.gold,
            science=self.science + other.science,
            culture=self.culture + other.culture,
            faith=self.faith + other.faith,
        )

    def __mul__(self, multiplier: float) -> TradeYields:
        """Scale yields by a multiplier."""
        return TradeYields(
            food=self.food * multiplier,
            production=self.production * multiplier,
            gold=self.gold * multiplier,
            science=self.science * multiplier,
            culture=self.culture * multiplier,
            faith=self.faith * multiplier,
        )

    def total(self) -> float:
        """Get total of all yields."""
        return self.food + self.production + self.gold + self.science + self.culture + self.faith

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "food": self.food,
            "production": self.production,
            "gold": self.gold,
            "science": self.science,
            "culture": self.culture,
            "faith": self.faith,
        }


@dataclass
class TradeRouteConfig:
    """Configuration for trade route calculations.

    Attributes:
        base_gold_per_tile: Base gold yield per tile distance
        destination_bonus_multiplier: Multiplier for destination bonuses
        domestic_food_multiplier: Food multiplier for domestic routes
        domestic_production_multiplier: Production multiplier for domestic routes
        international_gold_multiplier: Gold multiplier for international routes
        district_bonus_per_district: Bonus per district at destination
        trading_post_bonus: Bonus for having trading post
        max_route_length: Maximum tiles a route can span
        route_duration_turns: Base turns for a trade route
    """

    base_gold_per_tile: float = 0.5
    destination_bonus_multiplier: float = 1.0
    domestic_food_multiplier: float = 1.5
    domestic_production_multiplier: float = 1.5
    international_gold_multiplier: float = 2.0
    district_bonus_per_district: float = 0.5
    trading_post_bonus: float = 1.0
    max_route_length: int = 30
    route_duration_turns: int = 20


@dataclass
class TradeRoute:
    """Represents an active trade route.

    Attributes:
        route_id: Unique identifier for this route
        trader_id: ID of the trader unit on this route
        origin_city: Name of the origin city
        destination_city: Name of the destination city
        origin_player: Player ID of the origin
        destination_player: Player ID of the destination
        route_type: Type of trade route
        distance: Distance in tiles
        turns_remaining: Turns until route completes
        yields_per_turn: Calculated yields for this route
        path: List of tile positions along the route
        trading_posts: Set of cities with trading posts
    """

    route_id: str
    trader_id: str
    origin_city: str
    destination_city: str
    origin_player: int
    destination_player: int
    route_type: TradeRouteType
    distance: int
    turns_remaining: int
    yields_per_turn: TradeYields = field(default_factory=TradeYields)
    path: list[tuple[int, int]] = field(default_factory=list)
    trading_posts: set[str] = field(default_factory=set)

    @property
    def is_domestic(self) -> bool:
        """Check if this is a domestic route."""
        return self.origin_player == self.destination_player

    @property
    def is_active(self) -> bool:
        """Check if route is still active."""
        return self.turns_remaining > 0


@dataclass
class Trader:
    """Represents a trade unit.

    Attributes:
        trader_id: Unique identifier
        player_id: Owner player ID
        position: Current position (x, y)
        home_city: Name of the home city
        status: Current status
        current_route: Active route ID if trading
        movement_remaining: Movement points this turn
        max_movement: Maximum movement points
        level: Experience level (affects bonuses)
    """

    trader_id: str
    player_id: int
    position: tuple[int, int]
    home_city: str
    status: TraderStatus = TraderStatus.IDLE
    current_route: str | None = None
    movement_remaining: int = 3
    max_movement: int = 3
    level: int = 1


@dataclass
class TradeAgreement:
    """Bilateral trade agreement between players.

    Attributes:
        agreement_id: Unique identifier
        player_a: First player ID
        player_b: Second player ID
        gold_per_turn_a_to_b: Gold flow from A to B
        gold_per_turn_b_to_a: Gold flow from B to A
        resources_a_to_b: Resources A provides to B
        resources_b_to_a: Resources B provides to A
        turns_remaining: Duration remaining
        open_borders: Whether agreement includes open borders
    """

    agreement_id: str
    player_a: int
    player_b: int
    gold_per_turn_a_to_b: int = 0
    gold_per_turn_b_to_a: int = 0
    resources_a_to_b: list[str] = field(default_factory=list)
    resources_b_to_a: list[str] = field(default_factory=list)
    turns_remaining: int = 30
    open_borders: bool = False


class TradeManager:
    """Manages all trade-related mechanics.

    This class handles:
    - Trade route creation and management
    - Yield calculations
    - Trader unit tracking
    - Trade agreements

    Example:
        >>> config = TradeRouteConfig(base_gold_per_tile=0.6)
        >>> manager = TradeManager(config)
        >>> route = manager.create_route(
        ...     trader_id="trader_1",
        ...     origin="Rome",
        ...     destination="Alexandria",
        ...     origin_player=0,
        ...     destination_player=1,
        ...     distance=15
        ... )
        >>> yields = manager.calculate_yields(route, game_state)
    """

    def __init__(
        self,
        config: TradeRouteConfig | None = None,
        pathfinder: Callable[[tuple[int, int], tuple[int, int], Any], list[tuple[int, int]]]
        | None = None,
    ) -> None:
        """Initialize trade manager.

        Args:
            config: Trade route configuration
            pathfinder: Optional function for finding paths between cities
        """
        self.config = config or TradeRouteConfig()
        self._pathfinder = pathfinder
        self._routes: dict[str, TradeRoute] = {}
        self._traders: dict[str, Trader] = {}
        self._agreements: dict[str, TradeAgreement] = {}
        self._trading_posts: dict[int, set[str]] = {}  # player_id -> set of city names
        self._route_counter = 0

        logger.debug(f"TradeManager initialized with config: {self.config}")

    def create_route(
        self,
        trader_id: str,
        origin: str,
        destination: str,
        origin_player: int,
        destination_player: int,
        distance: int,
        destination_districts: int = 0,
        destination_wonders: int = 0,
    ) -> TradeRoute:
        """Create a new trade route.

        Args:
            trader_id: ID of the trader unit
            origin: Name of origin city
            destination: Name of destination city
            origin_player: Player ID of origin
            destination_player: Player ID of destination
            distance: Distance in tiles
            destination_districts: Number of districts at destination
            destination_wonders: Number of wonders at destination

        Returns:
            Created TradeRoute object
        """
        self._route_counter += 1
        route_id = f"route_{self._route_counter}"

        # Determine route type
        if origin_player == destination_player:
            route_type = TradeRouteType.DOMESTIC
        else:
            route_type = TradeRouteType.INTERNATIONAL

        # Calculate base yields
        yields = self._calculate_base_yields(
            route_type=route_type,
            distance=distance,
            destination_districts=destination_districts,
            destination_wonders=destination_wonders,
            origin_player=origin_player,
            destination=destination,
        )

        route = TradeRoute(
            route_id=route_id,
            trader_id=trader_id,
            origin_city=origin,
            destination_city=destination,
            origin_player=origin_player,
            destination_player=destination_player,
            route_type=route_type,
            distance=distance,
            turns_remaining=self.config.route_duration_turns,
            yields_per_turn=yields,
            trading_posts=self._trading_posts.get(origin_player, set()).copy(),
        )

        self._routes[route_id] = route

        # Update trader status
        if trader_id in self._traders:
            self._traders[trader_id].status = TraderStatus.TRADING
            self._traders[trader_id].current_route = route_id

        logger.info(
            f"Created trade route {route_id}: {origin} -> {destination} "
            f"(type={route_type.name}, distance={distance}, yields={yields.total():.1f})",
        )

        return route

    def _calculate_base_yields(
        self,
        route_type: TradeRouteType,
        distance: int,
        destination_districts: int,
        destination_wonders: int,
        origin_player: int,
        destination: str,
    ) -> TradeYields:
        """Calculate base yields for a trade route.

        Args:
            route_type: Type of route
            distance: Distance in tiles
            destination_districts: Districts at destination
            destination_wonders: Wonders at destination
            origin_player: Player who owns the route
            destination: Destination city name

        Returns:
            Calculated yields
        """
        yields = TradeYields()

        # Base gold from distance
        base_gold = distance * self.config.base_gold_per_tile

        # District bonuses
        district_bonus = destination_districts * self.config.district_bonus_per_district

        # Trading post bonus
        has_trading_post = destination in self._trading_posts.get(origin_player, set())
        trading_post_bonus = self.config.trading_post_bonus if has_trading_post else 0.0

        if route_type == TradeRouteType.DOMESTIC:
            # Domestic routes provide food and production
            yields.food = 1.0 * self.config.domestic_food_multiplier
            yields.production = 1.0 * self.config.domestic_production_multiplier
            yields.gold = base_gold
        else:
            # International routes provide more gold
            yields.gold = base_gold * self.config.international_gold_multiplier

        # Apply district and trading post bonuses
        yields.gold += district_bonus + trading_post_bonus

        # Wonder bonuses
        yields.culture += destination_wonders * 0.5
        yields.science += destination_wonders * 0.5

        return yields

    def process_turn(self) -> dict[int, TradeYields]:
        """Process all trade routes for a turn.

        Returns:
            Dictionary mapping player_id to total yields from trade
        """
        player_yields: dict[int, TradeYields] = {}
        completed_routes: list[str] = []

        for route_id, route in self._routes.items():
            # Accumulate yields for origin player
            if route.origin_player not in player_yields:
                player_yields[route.origin_player] = TradeYields()

            player_yields[route.origin_player] = (
                player_yields[route.origin_player] + route.yields_per_turn
            )

            # Decrement route duration
            route.turns_remaining -= 1

            if route.turns_remaining <= 0:
                completed_routes.append(route_id)
                # Add trading post to destination
                if route.origin_player not in self._trading_posts:
                    self._trading_posts[route.origin_player] = set()
                self._trading_posts[route.origin_player].add(route.destination_city)

        # Remove completed routes
        for route_id in completed_routes:
            route = self._routes.pop(route_id)
            # Update trader status
            if route.trader_id in self._traders:
                self._traders[route.trader_id].status = TraderStatus.IDLE
                self._traders[route.trader_id].current_route = None

            logger.info(f"Trade route {route_id} completed")

        # Process trade agreements
        completed_agreements: list[str] = []
        for agreement_id, agreement in self._agreements.items():
            # Apply gold transfers
            if agreement.player_a not in player_yields:
                player_yields[agreement.player_a] = TradeYields()
            if agreement.player_b not in player_yields:
                player_yields[agreement.player_b] = TradeYields()

            player_yields[agreement.player_a].gold -= agreement.gold_per_turn_a_to_b
            player_yields[agreement.player_a].gold += agreement.gold_per_turn_b_to_a
            player_yields[agreement.player_b].gold -= agreement.gold_per_turn_b_to_a
            player_yields[agreement.player_b].gold += agreement.gold_per_turn_a_to_b

            agreement.turns_remaining -= 1
            if agreement.turns_remaining <= 0:
                completed_agreements.append(agreement_id)

        for agreement_id in completed_agreements:
            self._agreements.pop(agreement_id)
            logger.info(f"Trade agreement {agreement_id} expired")

        return player_yields

    def register_trader(
        self,
        trader_id: str,
        player_id: int,
        position: tuple[int, int],
        home_city: str,
    ) -> Trader:
        """Register a new trade unit.

        Args:
            trader_id: Unique identifier
            player_id: Owner player
            position: Starting position
            home_city: Home city name

        Returns:
            Created Trader object
        """
        trader = Trader(
            trader_id=trader_id,
            player_id=player_id,
            position=position,
            home_city=home_city,
        )
        self._traders[trader_id] = trader
        logger.debug(f"Registered trader {trader_id} for player {player_id}")
        return trader

    def get_available_destinations(
        self,
        trader_id: str,
        cities: dict[str, tuple[int, int, int]],  # name -> (x, y, player_id)
    ) -> list[tuple[str, TradeRouteType, int]]:
        """Get available trade destinations for a trader.

        Args:
            trader_id: ID of the trader
            cities: Dictionary of city data (name -> (x, y, player_id))

        Returns:
            List of (city_name, route_type, distance) tuples
        """
        if trader_id not in self._traders:
            return []

        trader = self._traders[trader_id]
        destinations: list[tuple[str, TradeRouteType, int]] = []

        for city_name, (x, y, player_id) in cities.items():
            # Skip home city
            if city_name == trader.home_city:
                continue

            # Calculate Manhattan distance (simplified pathfinding)
            distance = abs(x - trader.position[0]) + abs(y - trader.position[1])

            if distance > self.config.max_route_length:
                continue

            # Determine route type
            if player_id == trader.player_id:
                route_type = TradeRouteType.DOMESTIC
            else:
                route_type = TradeRouteType.INTERNATIONAL

            destinations.append((city_name, route_type, distance))

        return sorted(destinations, key=lambda x: x[2])  # Sort by distance

    def create_agreement(
        self,
        player_a: int,
        player_b: int,
        gold_a_to_b: int = 0,
        gold_b_to_a: int = 0,
        resources_a_to_b: list[str] | None = None,
        resources_b_to_a: list[str] | None = None,
        duration: int = 30,
        open_borders: bool = False,
    ) -> TradeAgreement:
        """Create a trade agreement between two players.

        Args:
            player_a: First player ID
            player_b: Second player ID
            gold_a_to_b: Gold per turn from A to B
            gold_b_to_a: Gold per turn from B to A
            resources_a_to_b: Resources A provides
            resources_b_to_a: Resources B provides
            duration: Agreement duration in turns
            open_borders: Include open borders

        Returns:
            Created agreement
        """
        agreement_id = f"agreement_{player_a}_{player_b}_{len(self._agreements)}"

        agreement = TradeAgreement(
            agreement_id=agreement_id,
            player_a=player_a,
            player_b=player_b,
            gold_per_turn_a_to_b=gold_a_to_b,
            gold_per_turn_b_to_a=gold_b_to_a,
            resources_a_to_b=resources_a_to_b or [],
            resources_b_to_a=resources_b_to_a or [],
            turns_remaining=duration,
            open_borders=open_borders,
        )

        self._agreements[agreement_id] = agreement
        logger.info(f"Created trade agreement {agreement_id}")
        return agreement

    def get_player_trade_summary(self, player_id: int) -> dict[str, Any]:
        """Get summary of all trade for a player.

        Args:
            player_id: Player to summarize

        Returns:
            Dictionary with trade statistics
        """
        routes = [r for r in self._routes.values() if r.origin_player == player_id]
        traders = [t for t in self._traders.values() if t.player_id == player_id]
        agreements = [
            a
            for a in self._agreements.values()
            if a.player_a == player_id or a.player_b == player_id
        ]

        total_yields = TradeYields()
        for route in routes:
            total_yields = total_yields + route.yields_per_turn

        return {
            "active_routes": len(routes),
            "domestic_routes": len([r for r in routes if r.is_domestic]),
            "international_routes": len([r for r in routes if not r.is_domestic]),
            "traders_total": len(traders),
            "traders_idle": len([t for t in traders if t.status == TraderStatus.IDLE]),
            "traders_trading": len([t for t in traders if t.status == TraderStatus.TRADING]),
            "trading_posts": len(self._trading_posts.get(player_id, set())),
            "trade_agreements": len(agreements),
            "total_yields": total_yields.to_dict(),
        }

    def cancel_route(self, route_id: str) -> bool:
        """Cancel an active trade route.

        Args:
            route_id: ID of route to cancel

        Returns:
            True if successfully cancelled
        """
        if route_id not in self._routes:
            return False

        route = self._routes.pop(route_id)

        if route.trader_id in self._traders:
            self._traders[route.trader_id].status = TraderStatus.IDLE
            self._traders[route.trader_id].current_route = None

        logger.info(f"Cancelled trade route {route_id}")
        return True

    def get_route(self, route_id: str) -> TradeRoute | None:
        """Get a trade route by ID."""
        return self._routes.get(route_id)

    def get_trader(self, trader_id: str) -> Trader | None:
        """Get a trader by ID."""
        return self._traders.get(trader_id)

    def get_all_routes(self, player_id: int | None = None) -> list[TradeRoute]:
        """Get all trade routes, optionally filtered by player.

        Args:
            player_id: Optional player filter

        Returns:
            List of trade routes
        """
        if player_id is None:
            return list(self._routes.values())
        return [r for r in self._routes.values() if r.origin_player == player_id]

    def encode_for_state(self, player_id: int) -> dict[str, Any]:
        """Encode trade state for neural network input.

        Args:
            player_id: Player perspective

        Returns:
            Dictionary of encoded trade features
        """
        routes = self.get_all_routes(player_id)
        summary = self.get_player_trade_summary(player_id)

        return {
            "num_routes": len(routes),
            "domestic_routes": summary["domestic_routes"],
            "international_routes": summary["international_routes"],
            "traders_idle": summary["traders_idle"],
            "trading_posts": summary["trading_posts"],
            "trade_gold_per_turn": summary["total_yields"]["gold"],
            "trade_food_per_turn": summary["total_yields"]["food"],
            "trade_production_per_turn": summary["total_yields"]["production"],
        }

    def reset(self) -> None:
        """Reset all trade state."""
        self._routes.clear()
        self._traders.clear()
        self._agreements.clear()
        self._trading_posts.clear()
        self._route_counter = 0
        logger.debug("TradeManager reset")
