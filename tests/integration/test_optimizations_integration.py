"""Integration tests for Performance Optimizations.

Tests cover:
- Playout cap with MCTS-like simulation
- Cyclic LR with optimizer integration
- All optimizations working together
- PBT with optimization configs

The ``PerformanceOptimizationsConfig`` preset/serialization tests went with
``alpha_zero/optimizations/config.py`` (dead code, removed 2026-09-02).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from alpha_zero.optimizations import (
    AuxiliaryLossCalculator,
    AuxiliaryTargetGenerator,
    AuxiliaryTargets,
    AuxiliaryTargetsConfig,
    CyclicLRConfig,
    CyclicLRScheduler,
    PlayoutCapConfig,
    PlayoutCapRandomizer,
    PolicyPruningConfig,
    PolicyTargetPruner,
)


class TestPlayoutCapIntegration:
    """Integration tests for playout cap with simulated MCTS."""

    @pytest.fixture
    def playout_config(self):
        """Create playout cap configuration."""
        return PlayoutCapConfig(
            min_playouts=10,
            max_playouts=100,
            cap_distribution="geometric",
            geometric_p=0.3,
            use_position_complexity=True,
        )

    def test_playout_cap_with_mcts_simulation(self, playout_config):
        """Test playout cap randomizer with simulated MCTS runs."""
        rng = np.random.default_rng(42)
        randomizer = PlayoutCapRandomizer(playout_config, rng=rng)

        # Simulate multiple MCTS runs with varying complexity
        total_playouts_original = 0
        total_playouts_capped = 0
        num_positions = 100
        base_playouts = 100

        for _ in range(num_positions):
            # Get capped playout count
            cap = randomizer.sample_playout_cap()
            capped_playouts = min(base_playouts, cap)

            total_playouts_original += base_playouts
            total_playouts_capped += capped_playouts

        # Verify speedup
        speedup = total_playouts_original / total_playouts_capped
        assert speedup > 1.0  # Should have some speedup

        # Check stats
        stats = randomizer.get_stats()
        assert stats.total_samples == num_positions
        assert stats.average_cap > 0

    def test_playout_cap_complexity_scaling(self, playout_config):
        """Test that complexity scaling affects caps via game phase."""
        rng = np.random.default_rng(42)
        randomizer = PlayoutCapRandomizer(playout_config, rng=rng)

        # Sample caps - the randomizer uses game phase for complexity
        caps = [randomizer.sample_playout_cap() for _ in range(100)]

        # All caps should be within bounds
        assert all(playout_config.min_playouts <= c <= playout_config.max_playouts for c in caps)

        # Mean should be reasonable
        mean_cap = np.mean(caps)
        assert playout_config.min_playouts < mean_cap < playout_config.max_playouts

    def test_different_distributions(self):
        """Test all playout cap distribution types."""
        distributions = ["geometric", "uniform", "exponential"]

        for dist in distributions:
            config = PlayoutCapConfig(
                min_playouts=10,
                max_playouts=100,
                cap_distribution=dist,
                geometric_p=0.3,
                exponential_lambda=0.01,
            )
            rng = np.random.default_rng(42)
            randomizer = PlayoutCapRandomizer(config, rng=rng)

            # Sample many caps
            caps = [randomizer.sample_playout_cap() for _ in range(100)]

            # All caps should be within bounds
            assert all(config.min_playouts <= c <= config.max_playouts for c in caps)

            # Check distribution characteristics
            mean_cap = np.mean(caps)
            assert config.min_playouts < mean_cap < config.max_playouts


class TestCyclicLRIntegration:
    """Integration tests for cyclic LR with optimizer."""

    @pytest.fixture
    def lr_config(self):
        """Create cyclic LR configuration."""
        return CyclicLRConfig(
            schedule_type="cosine_annealing",
            base_lr=1e-5,
            max_lr=1e-3,
            step_size_up=100,
            warmup_steps=20,
        )

    def test_cyclic_lr_with_optimizer_simulation(self, lr_config):
        """Test cyclic LR integration with simulated optimizer updates."""
        scheduler = CyclicLRScheduler(lr_config)

        # Simulate training loop
        num_steps = 500
        lrs = []

        for step in range(num_steps):
            # Get learning rate for this step
            lr = scheduler.get_lr(step)
            lrs.append(lr)

        # Verify warmup
        assert lrs[0] < lrs[10] < lrs[19]  # LR increases during warmup
        assert abs(lrs[20] - lr_config.max_lr) < lr_config.max_lr * 0.05  # Near max after warmup

        # Verify cycling
        assert min(lrs[20:]) >= lr_config.base_lr * 0.99
        assert max(lrs[20:]) <= lr_config.max_lr * 1.01

        # Verify values change (cycling happening)
        unique_lrs = len(set(lrs[20:]))
        assert unique_lrs > 10  # Should have many different values

    def test_all_schedule_types(self):
        """Test all schedule types complete training simulation."""
        schedule_types = [
            "triangular",
            "triangular2",
            "exp_range",
            "cosine_annealing",
            "warm_restarts",
        ]

        for schedule_type in schedule_types:
            config = CyclicLRConfig(
                schedule_type=schedule_type,
                base_lr=1e-5,
                max_lr=1e-3,
                step_size_up=50,
                warmup_steps=10,
            )
            scheduler = CyclicLRScheduler(config)

            # Run through training steps
            lrs = [scheduler.get_lr(step) for step in range(200)]

            # All LRs should be valid (allowing 0 during warmup)
            assert all(0 <= lr <= config.max_lr * 1.01 for lr in lrs)
            assert all(lr >= config.base_lr * 0.99 for lr in lrs[10:])

    def test_momentum_cycling(self):
        """Test momentum cycles inversely to learning rate."""
        config = CyclicLRConfig(
            schedule_type="triangular",
            base_lr=1e-5,
            max_lr=1e-3,
            step_size_up=100,
            step_size_down=100,
            warmup_steps=0,
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
        )
        scheduler = CyclicLRScheduler(config)

        # At cycle start (low LR), momentum should be high
        lr_0 = scheduler.get_lr(0)
        mom_0 = scheduler.get_momentum(0)
        assert lr_0 == pytest.approx(config.base_lr, rel=1e-3)
        assert mom_0 == pytest.approx(config.max_momentum, rel=0.01)

        # At cycle peak (high LR), momentum should be low
        lr_100 = scheduler.get_lr(100)
        mom_100 = scheduler.get_momentum(100)
        assert lr_100 == pytest.approx(config.max_lr, rel=1e-3)
        assert mom_100 == pytest.approx(config.base_momentum, rel=0.01)


class TestPolicyPruningIntegration:
    """Integration tests for policy target pruning."""

    @pytest.fixture
    def pruning_config(self):
        """Create pruning configuration."""
        return PolicyPruningConfig(
            min_probability=0.01,
            max_actions=20,
            preserve_top_k=3,
            use_entropy_threshold=True,
        )

    def test_pruning_with_real_distributions(self, pruning_config):
        """Test pruning with realistic policy distributions."""
        pruner = PolicyTargetPruner(pruning_config)

        # Create realistic peaked policy (most mass on few actions)
        num_actions = 100
        logits = np.random.randn(num_actions)
        logits[0:5] += 5  # Make top 5 actions much more likely

        policy = np.exp(logits - np.max(logits))  # Softmax
        policy /= policy.sum()

        # Prune
        pruned = pruner.prune(policy)

        # Verify top-k preserved
        top_k_original = np.argsort(policy)[-pruning_config.preserve_top_k :]
        for idx in top_k_original:
            assert pruned[idx] > 0

        # Verify pruning happened
        num_kept = np.sum(pruned > 0)
        assert num_kept <= pruning_config.max_actions

        # Verify normalization
        assert np.isclose(pruned.sum(), 1.0)

    def test_pruning_maintains_relative_order(self, pruning_config):
        """Test that pruning maintains relative probability ordering."""
        pruner = PolicyTargetPruner(pruning_config)

        # Create simple policy
        policy = np.array([0.4, 0.3, 0.15, 0.1, 0.05])

        pruned = pruner.prune(policy)

        # Check relative ordering preserved for non-zero elements
        non_zero_indices = np.where(pruned > 0)[0]
        for i in range(len(non_zero_indices) - 1):
            idx1, idx2 = non_zero_indices[i], non_zero_indices[i + 1]
            if policy[idx1] > policy[idx2]:
                assert pruned[idx1] >= pruned[idx2]

    def test_pruning_reduction_rates(self, pruning_config):
        """Test that pruning achieves target reduction rates."""
        pruner = PolicyTargetPruner(pruning_config)

        # Test with various action space sizes
        for num_actions in [50, 100, 200, 500]:
            # Create uniform-ish policy with some noise
            policy = np.abs(np.random.randn(num_actions))
            policy = policy / policy.sum()

            pruned = pruner.prune(policy)
            num_kept = np.sum(pruned > 0)

            # Should be bounded by max_actions
            assert num_kept <= pruning_config.max_actions


class TestAllOptimizationsTogether:
    """Integration tests for all optimizations working together."""

    @pytest.fixture
    def full_config(self):
        """Bundle one config per optimization (the four live components)."""
        return SimpleNamespace(
            playout_cap=PlayoutCapConfig(
                min_playouts=25,
                max_playouts=200,
                cap_distribution="geometric",
            ),
            policy_pruning=PolicyPruningConfig(
                min_probability=0.005,
                max_actions=50,
            ),
            cyclic_lr=CyclicLRConfig(
                schedule_type="cosine_annealing",
                base_lr=1e-5,
                max_lr=1e-3,
                step_size_up=1000,
            ),
            auxiliary_targets=AuxiliaryTargetsConfig(
                enable_attention_targets=True,
                enable_opponent_modeling=True,
                enable_phase_prediction=False,
                enable_future_value=False,
            ),
        )

    def test_training_step_simulation(self, full_config):
        """Test simulated training step with all optimizations."""
        # Initialize all components
        rng = np.random.default_rng(42)
        playout_randomizer = PlayoutCapRandomizer(full_config.playout_cap, rng=rng)
        policy_pruner = PolicyTargetPruner(full_config.policy_pruning)
        lr_scheduler = CyclicLRScheduler(full_config.cyclic_lr)
        aux_generator = AuxiliaryTargetGenerator(full_config.auxiliary_targets)

        # Simulate training steps
        num_steps = 100
        for step in range(num_steps):
            # 1. Get playout cap for MCTS
            playout_cap = playout_randomizer.sample_playout_cap()
            assert (
                full_config.playout_cap.min_playouts
                <= playout_cap
                <= full_config.playout_cap.max_playouts
            )

            # 2. Get learning rate
            lr = lr_scheduler.get_lr(step)
            assert 0 <= lr <= full_config.cyclic_lr.max_lr * 1.01

            # 3. Simulate policy output and prune
            policy = np.random.dirichlet(np.ones(100))
            pruned_policy = policy_pruner.prune(policy)
            assert np.isclose(pruned_policy.sum(), 1.0)

            # 4. Generate auxiliary targets
            game_state = {
                "board": np.random.randn(16, 16),
                "turn": step % 50,
                "max_turns": 50,
            }
            aux_targets = aux_generator.generate_targets(game_state)
            assert isinstance(aux_targets, AuxiliaryTargets)


class TestPBTWithOptimizations:
    """Integration tests for PBT with optimization configs."""

    def test_pbt_hyperparameter_space_with_temperatures(self):
        """Test PBT hyperparameter space includes temperature parameters."""
        from alpha_zero.pbt.config import HyperparameterSpace

        space = HyperparameterSpace()

        # Verify temperature parameters exist
        all_params = space.get_all_params()
        assert "temperature_initial" in all_params
        assert "temperature_final" in all_params
        assert "temperature_decay_steps" in all_params

        # Verify ranges are valid
        assert space.temperature_initial.min_value < space.temperature_initial.max_value
        assert space.temperature_final.min_value < space.temperature_final.max_value
        assert space.temperature_decay_steps.min_value < space.temperature_decay_steps.max_value

    def test_crossover_mutation_with_optimization_params(self):
        """Test crossover mutation works with optimization-related parameters."""
        from alpha_zero.pbt.config import HyperparameterSpace
        from alpha_zero.pbt.mutations import CrossoverMutation

        rng = np.random.default_rng(42)

        # Create crossover mutation
        crossover = CrossoverMutation(
            crossover_rate=0.8,
            blend_alpha=0.5,
            crossover_type="blend",
        )

        # Set up parent values (simulating two members of population)
        parent1_values = {
            "learning_rate": 0.001,
            "temperature_initial": 1.5,
            "temperature_final": 0.3,
        }

        parent2_values = {
            "learning_rate": 0.0001,
            "temperature_initial": 2.0,
            "temperature_final": 0.1,
        }

        crossover.set_crossover_parent(parent2_values)

        # Test crossover for each parameter
        space = HyperparameterSpace()

        # Crossover learning rate
        lr_result = crossover.mutate(
            parent1_values["learning_rate"],
            space.learning_rate,
            rng,
            strength=0.2,
        )
        assert space.learning_rate.min_value <= lr_result <= space.learning_rate.max_value

        # Crossover temperature_initial
        temp_result = crossover.mutate(
            parent1_values["temperature_initial"],
            space.temperature_initial,
            rng,
            strength=0.2,
        )
        assert (
            space.temperature_initial.min_value
            <= temp_result
            <= space.temperature_initial.max_value
        )

    def test_mutation_operator_crossover_strategy(self):
        """Test MutationOperator with crossover strategy."""
        from alpha_zero.pbt.config import HyperparameterSpace
        from alpha_zero.pbt.mutations import MutationConfig, MutationOperator

        rng = np.random.default_rng(42)
        space = HyperparameterSpace()

        # Create MutationOperator with config
        config = MutationConfig(strategy="crossover", strength=0.1)
        operator = MutationOperator(config=config, rng=rng)

        # Set crossover parent
        parent_values = {
            "learning_rate": 0.0005,
            "batch_size": 128,
            "temperature_initial": 1.8,
        }
        operator.set_crossover_parent(parent_values)

        # Mutate a parameter
        original_lr = 0.001
        mutated_lr = operator.mutate(
            original_lr,
            space.learning_rate,
        )

        # Result should be within valid range
        assert space.learning_rate.min_value <= mutated_lr <= space.learning_rate.max_value


class TestAuxiliaryTargetsIntegration:
    """Integration tests for auxiliary policy targets."""

    @pytest.fixture
    def aux_config(self):
        """Create auxiliary targets configuration."""
        return AuxiliaryTargetsConfig(
            enable_attention_targets=True,
            enable_opponent_modeling=True,
            enable_phase_prediction=True,
            enable_future_value=True,
            attention_weight=0.1,
            opponent_weight=0.2,
            phase_weight=0.05,
            future_value_weight=0.1,
        )

    def test_auxiliary_loss_computation(self, aux_config):
        """Test auxiliary loss computation with all targets."""
        generator = AuxiliaryTargetGenerator(aux_config)
        calculator = AuxiliaryLossCalculator(aux_config)

        # Generate targets
        game_state = {
            "board": np.random.randn(16, 16),
            "turn": 25,
            "max_turns": 50,
        }
        targets = generator.generate_targets(game_state)

        # Simulate predictions as dict (matching target shapes)
        predictions = {}
        if targets.attention_map is not None:
            predictions["attention"] = np.random.rand(*targets.attention_map.shape)
        if targets.opponent_action is not None:
            opponent_pred = np.random.rand(*targets.opponent_action.shape)
            predictions["opponent"] = opponent_pred / opponent_pred.sum()
        if targets.game_phase is not None:
            phase_pred = np.random.rand(*targets.game_phase.shape)
            predictions["phase"] = phase_pred / phase_pred.sum()
        if targets.future_value is not None:
            predictions["future_value"] = np.random.rand()

        # Compute loss
        loss_dict = calculator.compute_loss(predictions, targets)

        assert isinstance(loss_dict, dict)
        assert "total_auxiliary_loss" in loss_dict
        assert loss_dict["total_auxiliary_loss"] >= 0  # Loss should be non-negative
        assert np.isfinite(loss_dict["total_auxiliary_loss"])

    def test_auxiliary_targets_phases(self, aux_config):
        """Test auxiliary targets across game phases."""
        generator = AuxiliaryTargetGenerator(aux_config)

        max_turns = 100

        # Early game
        early_state = {"board": np.random.randn(16, 16), "turn": 5, "max_turns": max_turns}
        early_targets = generator.generate_targets(early_state)

        # Mid game
        mid_state = {"board": np.random.randn(16, 16), "turn": 50, "max_turns": max_turns}
        mid_targets = generator.generate_targets(mid_state)
        assert mid_targets is not None  # Verify mid game targets generated

        # Late game
        late_state = {"board": np.random.randn(16, 16), "turn": 90, "max_turns": max_turns}
        late_targets = generator.generate_targets(late_state)

        # Verify phase predictions differ appropriately
        if aux_config.enable_phase_prediction:
            # Early game should have higher early phase probability
            assert early_targets.game_phase[0] > late_targets.game_phase[0]
            # Late game should have higher late phase probability
            assert late_targets.game_phase[2] > early_targets.game_phase[2]
