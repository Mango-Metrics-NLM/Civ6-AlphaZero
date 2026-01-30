"""Extended trade route tests for Phase 4 test coverage.

Comprehensive tests covering:
- TradeYields arithmetic operations
- Trade route creation and properties
- Trader management
- Turn processing
"""

from __future__ import annotations

import pytest

from alpha_zero.mechanics.trade import (
    TradeManager,
    TradeRoute,
    TradeRouteConfig,
    TradeRouteType,
    TraderStatus,
    TradeYields,
)

# =============================================================================
# TradeYields Tests
# =============================================================================


class TestTradeYields:
    """Tests for TradeYields dataclass operations."""

    def test_yields_creation(self) -> None:
        """Test creating yields with all values."""
        yields = TradeYields(
            gold=10.0,
            food=5.0,
            production=3.0,
            science=2.0,
            culture=1.0,
            faith=0.5,
        )
        assert yields.gold == 10.0
        assert yields.food == 5.0
        assert yields.production == 3.0

    def test_yields_add(self) -> None:
        """Test adding two yield objects."""
        yields1 = TradeYields(gold=10.0, food=5.0, production=3.0)
        yields2 = TradeYields(gold=5.0, food=2.0, production=1.0)

        result = yields1 + yields2

        assert result.gold == 15.0
        assert result.food == 7.0
        assert result.production == 4.0

    def test_yields_add_identity(self) -> None:
        """Test adding zero yields preserves original."""
        yields = TradeYields(gold=10.0, food=5.0)
        zero_yields = TradeYields()

        result = yields + zero_yields

        assert result.gold == yields.gold
        assert result.food == yields.food

    def test_yields_add_commutativity(self) -> None:
        """Test addition is commutative (a + b = b + a)."""
        yields1 = TradeYields(gold=10.0, food=5.0)
        yields2 = TradeYields(gold=3.0, food=2.0)

        result1 = yields1 + yields2
        result2 = yields2 + yields1

        assert result1.gold == result2.gold
        assert result1.food == result2.food

    def test_yields_mul(self) -> None:
        """Test scalar multiplication."""
        yields = TradeYields(gold=10.0, food=5.0, production=2.0)

        result = yields * 2.0

        assert result.gold == 20.0
        assert result.food == 10.0
        assert result.production == 4.0

    def test_yields_mul_by_zero(self) -> None:
        """Test multiplication by zero."""
        yields = TradeYields(gold=10.0, food=5.0)

        result = yields * 0.0

        assert result.gold == 0.0
        assert result.food == 0.0

    def test_yields_mul_by_fraction(self) -> None:
        """Test multiplication by fraction."""
        yields = TradeYields(gold=10.0, food=4.0)

        result = yields * 0.5

        assert result.gold == 5.0
        assert result.food == 2.0

    def test_yields_total(self) -> None:
        """Test total() returns sum of all yield types."""
        yields = TradeYields(
            gold=10.0,
            food=5.0,
            production=3.0,
            science=2.0,
            culture=1.0,
            faith=0.5,
        )

        total = yields.total()
        assert total == 21.5

    def test_yields_to_dict(self) -> None:
        """Test conversion to dictionary."""
        yields = TradeYields(gold=10.0, food=5.0)

        result = yields.to_dict()

        assert isinstance(result, dict)
        assert result["gold"] == 10.0
        assert result["food"] == 5.0


# =============================================================================
# TradeRoute Tests
# =============================================================================


class TestTradeRoute:
    """Tests for TradeRoute dataclass properties."""

    def test_is_domestic(self) -> None:
        """Test is_domestic property for same-player routes."""
        route = TradeRoute(
            route_id="route_1",
            trader_id="trader_1",
            origin_city="CityA",
            destination_city="CityB",
            origin_player=0,
            destination_player=0,  # Same player
            route_type=TradeRouteType.DOMESTIC,
            distance=5,
            turns_remaining=10,
            yields_per_turn=TradeYields(gold=5.0),
        )

        assert route.is_domestic is True

    def test_is_international(self) -> None:
        """Test is_domestic is False for different-player routes."""
        route = TradeRoute(
            route_id="route_1",
            trader_id="trader_1",
            origin_city="CityA",
            destination_city="CityB",
            origin_player=0,
            destination_player=1,  # Different player
            route_type=TradeRouteType.INTERNATIONAL,
            distance=10,
            turns_remaining=10,
            yields_per_turn=TradeYields(gold=10.0),
        )

        assert route.is_domestic is False

    def test_is_active_positive_turns(self) -> None:
        """Test is_active when turns_remaining > 0."""
        route = TradeRoute(
            route_id="route_1",
            trader_id="trader_1",
            origin_city="CityA",
            destination_city="CityB",
            origin_player=0,
            destination_player=1,
            route_type=TradeRouteType.INTERNATIONAL,
            distance=5,
            turns_remaining=5,
            yields_per_turn=TradeYields(gold=5.0),
        )

        assert route.is_active is True

    def test_is_active_zero_turns(self) -> None:
        """Test is_active when turns_remaining = 0."""
        route = TradeRoute(
            route_id="route_1",
            trader_id="trader_1",
            origin_city="CityA",
            destination_city="CityB",
            origin_player=0,
            destination_player=1,
            route_type=TradeRouteType.INTERNATIONAL,
            distance=5,
            turns_remaining=0,
            yields_per_turn=TradeYields(gold=5.0),
        )

        assert route.is_active is False


# =============================================================================
# TradeManager Tests
# =============================================================================


class TestTradeManager:
    """Tests for TradeManager."""

    @pytest.fixture
    def manager(self) -> TradeManager:
        """Create a trade manager."""
        return TradeManager()

    def test_manager_creation(self) -> None:
        """Test creating a trade manager."""
        manager = TradeManager()
        assert manager.config is not None
        assert manager.config.base_gold_per_tile == 0.5

    def test_manager_with_custom_config(self) -> None:
        """Test creating manager with custom config."""
        config = TradeRouteConfig(
            base_gold_per_tile=1.0,
            max_route_length=50,
        )
        manager = TradeManager(config=config)

        assert manager.config.base_gold_per_tile == 1.0
        assert manager.config.max_route_length == 50

    def test_register_trader(self, manager: TradeManager) -> None:
        """Test registering a new trader."""
        trader = manager.register_trader(
            trader_id="trader_1",
            player_id=0,
            position=(5, 5),
            home_city="Capital",
        )

        assert trader is not None
        assert trader.trader_id == "trader_1"
        assert trader.player_id == 0
        assert trader.status == TraderStatus.IDLE
        assert trader.home_city == "Capital"

    def test_register_multiple_traders(self, manager: TradeManager) -> None:
        """Test registering multiple traders."""
        trader1 = manager.register_trader(
            trader_id="trader_1",
            player_id=0,
            position=(0, 0),
            home_city="Capital",
        )
        trader2 = manager.register_trader(
            trader_id="trader_2",
            player_id=0,
            position=(5, 5),
            home_city="CityA",
        )

        assert trader1.trader_id != trader2.trader_id
        assert manager.get_trader("trader_1") is trader1
        assert manager.get_trader("trader_2") is trader2

    def test_get_trader_nonexistent(self, manager: TradeManager) -> None:
        """Test getting non-existent trader returns None."""
        result = manager.get_trader("nonexistent")
        assert result is None

    def test_create_route(self, manager: TradeManager) -> None:
        """Test creating a trade route."""
        manager.register_trader(
            trader_id="trader_1",
            player_id=0,
            position=(0, 0),
            home_city="Capital",
        )

        route = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=5,
        )

        assert route is not None
        assert route.origin_city == "Capital"
        assert route.destination_city == "CityA"
        assert route.is_domestic is True

    def test_create_international_route(self, manager: TradeManager) -> None:
        """Test creating an international trade route."""
        manager.register_trader(
            trader_id="trader_1",
            player_id=0,
            position=(0, 0),
            home_city="Capital",
        )

        route = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="ForeignCity",
            origin_player=0,
            destination_player=1,
            distance=10,
        )

        assert route is not None
        assert route.route_type == TradeRouteType.INTERNATIONAL
        assert route.is_domestic is False

    def test_route_id_increments(self, manager: TradeManager) -> None:
        """Test that route IDs increment."""
        manager.register_trader("trader_1", 0, (0, 0), "Capital")
        manager.register_trader("trader_2", 0, (0, 0), "Capital")

        route1 = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=5,
        )
        route2 = manager.create_route(
            trader_id="trader_2",
            origin="Capital",
            destination="CityB",
            origin_player=0,
            destination_player=0,
            distance=5,
        )

        assert route1.route_id != route2.route_id

    def test_get_route(self, manager: TradeManager) -> None:
        """Test getting a route by ID."""
        manager.register_trader("trader_1", 0, (0, 0), "Capital")

        route = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=5,
        )

        retrieved = manager.get_route(route.route_id)
        assert retrieved is route

    def test_get_route_nonexistent(self, manager: TradeManager) -> None:
        """Test getting non-existent route returns None."""
        result = manager.get_route("nonexistent")
        assert result is None

    def test_cancel_route(self, manager: TradeManager) -> None:
        """Test canceling a route."""
        manager.register_trader("trader_1", 0, (0, 0), "Capital")

        route = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=5,
        )
        route_id = route.route_id

        success = manager.cancel_route(route_id)

        assert success is True
        assert manager.get_route(route_id) is None

    def test_cancel_route_nonexistent(self, manager: TradeManager) -> None:
        """Test canceling non-existent route returns False."""
        success = manager.cancel_route("nonexistent_route")
        assert success is False

    def test_process_turn_empty(self, manager: TradeManager) -> None:
        """Test process_turn handles no routes gracefully."""
        yields = manager.process_turn()
        assert isinstance(yields, dict)

    def test_process_turn_with_route(self, manager: TradeManager) -> None:
        """Test process_turn accumulates yields."""
        manager.register_trader("trader_1", 0, (0, 0), "Capital")

        route = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=5,
        )

        initial_turns = route.turns_remaining
        yields = manager.process_turn()

        # Route duration should decrement
        assert route.turns_remaining == initial_turns - 1

        # Should have some yields
        assert 0 in yields or len(yields) > 0

    def test_reset(self, manager: TradeManager) -> None:
        """Test resetting the manager clears all state."""
        manager.register_trader("trader_1", 0, (0, 0), "Capital")
        manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=5,
        )

        manager.reset()

        # All state should be cleared
        assert len(manager._routes) == 0
        assert len(manager._traders) == 0
        assert len(manager._agreements) == 0


# =============================================================================
# Yield Calculation Tests
# =============================================================================


class TestYieldCalculation:
    """Tests for yield calculation logic."""

    def test_domestic_route_has_food_and_production(self) -> None:
        """Test domestic routes include food and production."""
        manager = TradeManager()
        manager.register_trader("trader_1", 0, (0, 0), "Capital")

        route = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,  # Domestic
            distance=5,
        )

        # Domestic routes should have food and production
        assert route.yields_per_turn.food > 0
        assert route.yields_per_turn.production > 0

    def test_international_route_gold_multiplier(self) -> None:
        """Test international routes have higher gold yields."""
        manager = TradeManager()
        manager.register_trader("trader_1", 0, (0, 0), "Capital")
        manager.register_trader("trader_2", 0, (0, 0), "Capital")

        domestic = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=10,
        )
        international = manager.create_route(
            trader_id="trader_2",
            origin="Capital",
            destination="ForeignCity",
            origin_player=0,
            destination_player=1,
            distance=10,
        )

        # International should have higher gold (2x multiplier by default)
        assert international.yields_per_turn.gold > domestic.yields_per_turn.gold

    def test_longer_distance_more_gold(self) -> None:
        """Test longer routes have more gold."""
        manager = TradeManager()
        manager.register_trader("trader_1", 0, (0, 0), "Capital")
        manager.register_trader("trader_2", 0, (0, 0), "Capital")

        short = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=1,
            distance=5,
        )
        long = manager.create_route(
            trader_id="trader_2",
            origin="Capital",
            destination="CityB",
            origin_player=0,
            destination_player=1,
            distance=15,
        )

        assert long.yields_per_turn.gold > short.yields_per_turn.gold

    def test_district_bonus(self) -> None:
        """Test districts increase yields."""
        manager = TradeManager()
        manager.register_trader("trader_1", 0, (0, 0), "Capital")
        manager.register_trader("trader_2", 0, (0, 0), "Capital")

        no_districts = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=5,
            destination_districts=0,
        )
        with_districts = manager.create_route(
            trader_id="trader_2",
            origin="Capital",
            destination="CityB",
            origin_player=0,
            destination_player=0,
            distance=5,
            destination_districts=5,
        )

        assert with_districts.yields_per_turn.gold > no_districts.yields_per_turn.gold


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestTradeEdgeCases:
    """Tests for edge cases in trade mechanics."""

    def test_zero_distance_route(self) -> None:
        """Test creating route with zero distance."""
        manager = TradeManager()
        manager.register_trader("trader_1", 0, (0, 0), "Capital")

        route = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="Capital",
            origin_player=0,
            destination_player=0,
            distance=0,
        )

        # Should still create route
        assert route is not None
        assert route.yields_per_turn.gold >= 0

    def test_route_completion(self) -> None:
        """Test route is removed when it completes."""
        config = TradeRouteConfig(route_duration_turns=2)
        manager = TradeManager(config=config)

        manager.register_trader("trader_1", 0, (0, 0), "Capital")

        route = manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=5,
        )
        route_id = route.route_id

        # Process until route completes
        manager.process_turn()  # 2 -> 1
        assert manager.get_route(route_id) is not None

        manager.process_turn()  # 1 -> 0 (completes)
        assert manager.get_route(route_id) is None

    def test_trader_status_changes(self) -> None:
        """Test trader status changes with route."""
        manager = TradeManager()
        trader = manager.register_trader("trader_1", 0, (0, 0), "Capital")

        assert trader.status == TraderStatus.IDLE

        manager.create_route(
            trader_id="trader_1",
            origin="Capital",
            destination="CityA",
            origin_player=0,
            destination_player=0,
            distance=5,
        )

        assert trader.status == TraderStatus.TRADING
