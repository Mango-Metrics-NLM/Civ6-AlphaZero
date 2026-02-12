"""Domain-agnostic protocols for the Strategic Planning SDK.

This module defines the structural-subtyping contracts (protocols) that all SDK
components program against.  Implementations satisfy a protocol simply by
providing the required attributes and methods -- no base-class inheritance is
needed.

Every protocol is decorated with :func:`~typing.runtime_checkable` so that
``isinstance`` checks work at runtime (with the usual caveat that only method
*existence*, not *signatures*, is verified by ``isinstance``).

Layer 1 -- depends only on Layer 0 (types, exceptions).

Example::

    from src.core.protocols import StrategicEnvironment

    class MyEnv:
        # Implement all StrategicEnvironment methods ...
        ...

    assert isinstance(MyEnv(), StrategicEnvironment)
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.core.exceptions import ProtocolViolationError
from src.core.types import ObservationSpec

__all__ = [
    "CheckpointableProtocol",
    "MCTSProtocol",
    "NetworkProtocol",
    "RewardShaper",
    "SelectionPolicy",
    "Serializable",
    "StateEncoder",
    "StorageBackend",
    "StrategicEnvironment",
    "TrainerProtocol",
    "assert_valid_environment",
    "check_protocol_compliance",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Serializable
# ---------------------------------------------------------------------------


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be round-tripped through dictionaries.

    Any class that provides ``to_dict`` and the classmethod ``from_dict``
    satisfies this protocol, enabling transparent JSON / YAML / MessagePack
    serialisation without requiring SDK base classes.

    Example::

        @dataclass
        class MyConfig:
            value: int = 10

            def to_dict(self) -> dict[str, Any]:
                return {"value": self.value}

            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> "MyConfig":
                return cls(**data)

        assert isinstance(MyConfig(), Serializable)
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialise the object to a plain dictionary.

        Returns:
            A JSON-serialisable dictionary representation.  All values must
            be primitives, lists, or nested dicts (no numpy arrays or
            tensors).

        Postconditions:
            - ``type(result)`` is ``dict``.
            - The result is JSON-serialisable.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Serializable:
        """Reconstruct an instance from a dictionary.

        Args:
            data: Dictionary previously produced by ``to_dict()``.

        Returns:
            A new instance whose state matches the serialised data.

        Preconditions:
            - *data* was produced by ``to_dict()`` (or is structurally
              equivalent).
        """
        ...


# ---------------------------------------------------------------------------
# 2. StrategicEnvironment
# ---------------------------------------------------------------------------


@runtime_checkable
class StrategicEnvironment(Protocol):
    """Core environment protocol for strategic planning domains.

    This is the central abstraction through which the SDK interacts with
    any turn-based or step-based strategic environment.  Implementations
    must support cloning (for MCTS tree expansion) and provide a boolean
    action mask at every step.

    The observation dictionary returned by ``reset``, ``step``, and
    ``get_observation`` **must** contain at least:

    * ``"spatial"`` -- :class:`NDArray[np.float32]` matching
      ``observation_spec.spatial_shape``.
    * ``"vector"`` -- :class:`NDArray[np.float32]` of length
      ``observation_spec.vector_dim``.

    Optionally it may also contain ``"action_mask"`` and arbitrary
    domain-specific keys.

    Example::

        env: StrategicEnvironment = MyChessEnv()
        obs = env.reset(seed=42)
        while not env.is_terminal():
            mask = env.get_valid_actions()
            action = int(np.argmax(mask))  # dummy policy
            obs, reward, terminated, truncated, info = env.step(action)
    """

    @property
    def observation_spec(self) -> ObservationSpec:
        """Specification describing observation tensor shapes.

        Returns:
            An :class:`~src.core.types.ObservationSpec` instance.
        """
        ...

    @property
    def action_space_size(self) -> int:
        """Total number of discrete actions in the flat action space.

        Returns:
            A positive integer.
        """
        ...

    def reset(
        self, *, seed: int | None = None
    ) -> dict[str, NDArray[np.float32]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional random seed for reproducibility.  When
                ``None``, the environment may use any internal source
                of randomness.

        Returns:
            The initial observation dictionary.

        Postconditions:
            - ``len(self.get_valid_actions()) >= 1`` -- at least one
              legal action exists.
            - ``not self.is_terminal()`` -- the fresh state is not
              terminal.
            - ``obs["spatial"].shape == self.observation_spec.spatial_shape``.
            - ``obs["vector"].shape == (self.observation_spec.vector_dim,)``.
        """
        ...

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        """Execute an action and advance the environment by one step.

        Args:
            action: A flat action index in ``[0, action_space_size)``.

        Returns:
            A 5-tuple ``(obs, reward, terminated, truncated, info)`` where:

            * *obs* -- the new observation dictionary.
            * *reward* -- scalar reward for the acting player.
            * *terminated* -- whether the episode ended naturally.
            * *truncated* -- whether the episode was cut short (e.g. step
              limit).
            * *info* -- auxiliary information (domain-specific).

        Preconditions:
            - ``not self.is_terminal()`` -- cannot step a finished episode.
            - ``self.get_valid_actions()[action]`` is ``True`` -- the action
              must be legal.

        Postconditions:
            - ``-1.0 <= reward <= 1.0``.
            - Observation tensors conform to ``observation_spec``.
        """
        ...

    def get_valid_actions(self) -> NDArray[np.bool_]:
        """Return a boolean mask over the flat action space.

        Returns:
            A boolean array of shape ``(action_space_size,)`` where
            ``True`` indicates a legal action.

        Postconditions:
            - ``result.shape == (self.action_space_size,)``.
            - At least one element is ``True`` unless ``self.is_terminal()``
              is ``True``.
        """
        ...

    def get_observation(self) -> dict[str, NDArray[np.float32]]:
        """Return the current observation without advancing the state.

        Returns:
            Observation dictionary with at least ``"spatial"`` and
            ``"vector"`` keys.
        """
        ...

    def is_terminal(self) -> bool:
        """Check whether the current episode has ended.

        Returns:
            ``True`` if the episode is over (either terminated or
            truncated), ``False`` otherwise.
        """
        ...

    def clone(self) -> StrategicEnvironment:
        """Create a deep copy of the environment and its internal state.

        This is used by MCTS to expand search trees without mutating
        the original environment.

        Returns:
            A new :class:`StrategicEnvironment` instance whose state is
            identical to ``self`` but fully independent.

        Postconditions:
            - Modifications to the clone do **not** affect the original.
        """
        ...

    def get_symmetries(
        self,
        observation: NDArray[np.float32],
        policy: NDArray[np.float32],
    ) -> list[tuple[NDArray[np.float32], NDArray[np.float32]]]:
        """Return symmetric equivalents of an observation-policy pair.

        Symmetry augmentation increases the effective size of the
        training dataset by exploiting domain symmetries (e.g. board
        rotations and reflections).

        Args:
            observation: A spatial observation tensor (e.g. board state).
            policy: The MCTS-improved policy distribution over actions.

        Returns:
            A list of ``(augmented_obs, augmented_policy)`` tuples.
            The list **always** includes the original ``(observation,
            policy)`` pair as its first element.

        Postconditions:
            - ``len(result) >= 1``.
            - ``result[0] == (observation, policy)`` (identity transform).
        """
        ...


# ---------------------------------------------------------------------------
# 3. StateEncoder
# ---------------------------------------------------------------------------


@runtime_checkable
class StateEncoder(Protocol):
    """Protocol for transforming raw game state into tensor features.

    A :class:`StateEncoder` bridges the gap between an environment's
    native state representation (often a nested dictionary) and the
    fixed-shape tensors expected by the neural network.

    Example::

        encoder: StateEncoder = MyCiv6Encoder()
        obs = encoder.encode(raw_game_state)
        spatial = obs["spatial"]  # shape matches encoder.observation_spec.spatial_shape
    """

    @property
    def observation_spec(self) -> ObservationSpec:
        """Specification describing the output tensor shapes.

        Returns:
            An :class:`~src.core.types.ObservationSpec` whose
            ``spatial_shape`` and ``vector_dim`` match the arrays
            produced by :meth:`encode`.
        """
        ...

    def encode(
        self, raw_state: dict[str, Any]
    ) -> dict[str, NDArray[np.float32]]:
        """Encode a raw game state into spatial and vector tensors.

        Args:
            raw_state: Domain-specific game state dictionary.

        Returns:
            A dictionary with **at least** the following keys:

            * ``"spatial"`` -- :class:`NDArray[np.float32]` of shape
              ``observation_spec.spatial_shape``.
            * ``"vector"`` -- :class:`NDArray[np.float32]` of shape
              ``(observation_spec.vector_dim,)``.

        Postconditions:
            - ``"spatial" in result and "vector" in result``.
            - Shapes conform to ``self.observation_spec``.
        """
        ...

    def decode_action(self, action_index: int) -> dict[str, Any]:
        """Convert a flat action index to a human-readable description.

        Args:
            action_index: Integer in ``[0, action_space_size)``.

        Returns:
            A dictionary describing the action in domain terms
            (e.g. ``{"type": "move", "unit": 3, "direction": "north"}``).
        """
        ...


# ---------------------------------------------------------------------------
# 4. NetworkProtocol
# ---------------------------------------------------------------------------


@runtime_checkable
class NetworkProtocol(Protocol):
    """Protocol for neural networks with dual policy/value heads.

    Implementations wrap a deep neural network (typically PyTorch) that
    accepts spatial and vector features and produces:

    * **policy logits** (or probabilities) over the action space, and
    * a scalar **value estimate** for the current state.

    ``Any`` is used for tensor types to avoid a hard dependency on
    PyTorch at the protocol level.

    Example::

        net: NetworkProtocol = MyResNet(board_size=16, num_actions=356)
        logits, value = net.forward(spatial_tensor, vector_tensor)
        probs, value = net.predict(spatial_tensor, vector_tensor)
    """

    def forward(
        self,
        spatial: Any,
        vector: Any,
        action_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        """Training-mode forward pass.

        Args:
            spatial: Spatial feature tensor of shape ``(B, C, H, W)``.
            vector: Vector feature tensor of shape ``(B, V)``.
            action_mask: Optional boolean mask of shape ``(B, A)``
                indicating valid actions.  When provided,
                implementations should mask out invalid actions before
                computing the output distribution.

        Returns:
            A tuple ``(policy_logits, value)`` where:

            * *policy_logits* has shape ``(B, A)`` (``A`` is the action
              space size).
            * *value* has shape ``(B, 1)`` with values in ``[-1, 1]``.

        Postconditions:
            - ``policy_logits.shape[-1] == action_space_size``.
            - ``value.shape[-1] == 1``.
        """
        ...

    def predict(
        self,
        spatial: Any,
        vector: Any,
        action_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        """Inference-mode forward pass (no gradient tracking).

        Identical to :meth:`forward` except:

        * Gradients are disabled (e.g. ``torch.no_grad()``).
        * Policy output is a **probability distribution** (sums to 1)
          rather than raw logits.

        Args:
            spatial: Spatial feature tensor of shape ``(B, C, H, W)``.
            vector: Vector feature tensor of shape ``(B, V)``.
            action_mask: Optional boolean mask of shape ``(B, A)``.

        Returns:
            A tuple ``(policy_probs, value)`` where:

            * *policy_probs* has shape ``(B, A)`` and sums to ~1.0
              along the last dimension.
            * *value* has shape ``(B, 1)`` with values in ``[-1, 1]``.
        """
        ...


# ---------------------------------------------------------------------------
# 5. MCTSProtocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MCTSProtocol(Protocol):
    """Protocol for Monte Carlo Tree Search implementations.

    Any MCTS variant (vanilla PUCT, Gumbel AlphaZero, etc.) that
    exposes ``search`` and ``select_action`` satisfies this protocol.

    Example::

        mcts: MCTSProtocol = GumbelMCTS(network, config)
        action_probs, root_value = mcts.search(env, num_simulations=800)
        action = mcts.select_action(action_probs, temperature=0.5)
    """

    def search(
        self,
        env: StrategicEnvironment,
        num_simulations: int | None = None,
    ) -> tuple[NDArray[np.float32], float]:
        """Run tree search from the current environment state.

        Args:
            env: Environment positioned at the root state.  The
                implementation may call ``env.clone()`` internally
                but must **not** mutate *env* itself.
            num_simulations: Number of simulations to perform.  If
                ``None``, the implementation uses its configured
                default.

        Returns:
            A tuple ``(action_probs, root_value)`` where:

            * *action_probs* is a float array of shape
              ``(env.action_space_size,)`` summing to ~1.0.
            * *root_value* is the estimated value of the root state.

        Invariants:
            - ``abs(sum(action_probs) - 1.0) < 1e-5``.
            - *env* is not mutated.
        """
        ...

    def select_action(
        self,
        action_probs: NDArray[np.float32],
        temperature: float = 1.0,
        action_mask: NDArray[np.bool_] | None = None,
    ) -> int:
        """Select a single action from a probability distribution.

        When ``temperature == 0`` the action with the highest
        probability is chosen (argmax).  When ``temperature > 0``
        actions are sampled proportionally to
        ``probs^(1/temperature)``.

        Args:
            action_probs: Probability distribution over actions.
            temperature: Sampling temperature.  ``0`` means greedy,
                ``1`` means proportional sampling.
            action_mask: Optional boolean mask.  When provided, only
                actions where the mask is ``True`` may be selected.

        Returns:
            A scalar action index.

        Postconditions:
            - ``0 <= result < len(action_probs)``.
            - If *action_mask* is provided,
              ``action_mask[result]`` is ``True``.
        """
        ...


# ---------------------------------------------------------------------------
# 6. SelectionPolicy
# ---------------------------------------------------------------------------


@runtime_checkable
class SelectionPolicy(Protocol):
    """Strategy for selecting a child node during MCTS tree traversal.

    Different selection policies (PUCT, Gumbel, etc.) implement this
    protocol to provide pluggable exploration/exploitation trade-offs.

    Example::

        policy: SelectionPolicy = PUCTPolicy(c_puct=1.5)
        child_idx = policy.select_child(
            parent_visit_count=100,
            children_priors=priors,
            children_values=q_values,
            children_visit_counts=visits,
            valid_mask=mask,
        )
    """

    def select_child(
        self,
        parent_visit_count: int,
        children_priors: NDArray[np.float32],
        children_values: NDArray[np.float32],
        children_visit_counts: NDArray[np.int64],
        valid_mask: NDArray[np.bool_],
    ) -> int:
        """Select the index of the child to expand next.

        Args:
            parent_visit_count: Total visit count of the parent node.
            children_priors: Prior probabilities for each child,
                shape ``(num_children,)``.
            children_values: Mean action-value (Q) for each child,
                shape ``(num_children,)``.
            children_visit_counts: Visit count for each child,
                shape ``(num_children,)``.
            valid_mask: Boolean mask indicating which children are
                valid candidates, shape ``(num_children,)``.

        Returns:
            Index of the selected child.

        Preconditions:
            - ``np.any(valid_mask)`` is ``True`` -- at least one child
              must be valid.

        Postconditions:
            - ``valid_mask[result]`` is ``True``.
            - ``0 <= result < len(children_priors)``.
        """
        ...


# ---------------------------------------------------------------------------
# 7. RewardShaper
# ---------------------------------------------------------------------------


@runtime_checkable
class RewardShaper(Protocol):
    """Optional domain-specific reward shaping transformation.

    Reward shapers modify the raw environment reward to provide denser
    or more informative learning signals.  The shaped reward must remain
    within ``[-1, 1]`` to preserve training stability.

    Example::

        shaper: RewardShaper = ExplorationBonusShaper(bonus=0.01)
        shaped = shaper.shape_reward(
            raw_reward=0.0,
            observation=obs,
            action=42,
            info={"cities_founded": 2},
        )
    """

    def shape_reward(
        self,
        raw_reward: float,
        observation: dict[str, NDArray[np.float32]],
        action: int,
        info: dict[str, Any],
    ) -> float:
        """Transform a raw reward into a shaped reward.

        Args:
            raw_reward: The unmodified reward from the environment's
                ``step`` method.
            observation: The observation produced by the same ``step``
                call.
            action: The action that was taken.
            info: The auxiliary info dict from ``step``.

        Returns:
            The shaped reward.

        Postconditions:
            - ``-1.0 <= result <= 1.0``.
        """
        ...


# ---------------------------------------------------------------------------
# 8. CheckpointableProtocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CheckpointableProtocol(Protocol):
    """Protocol for objects whose state can be persisted and restored.

    This covers neural network weights, optimiser state, replay buffers,
    or any other component that needs save/load semantics.

    Example::

        model: CheckpointableProtocol = MyNetwork()
        model.save("/tmp/checkpoint.pt")
        model.load("/tmp/checkpoint.pt", device="cuda:0")
    """

    def save(self, path: str) -> None:
        """Persist the object's state to a filesystem path.

        Args:
            path: Destination path (file or directory, depending on
                the implementation).

        Postconditions:
            - A file (or directory) exists at *path* after return.
        """
        ...

    def load(self, path: str, device: str = "cpu") -> None:
        """Restore state from a previously saved checkpoint.

        Args:
            path: Source path produced by a prior ``save`` call.
            device: Target device for loaded tensors (e.g. ``"cpu"``,
                ``"cuda:0"``).

        Preconditions:
            - A valid checkpoint exists at *path*.
        """
        ...


# ---------------------------------------------------------------------------
# 9. TrainerProtocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol for training loop implementations.

    A trainer wraps the gradient-update logic and checkpoint management
    for one or more neural networks.

    Example::

        trainer: TrainerProtocol = AlphaZeroTrainer(network, config)
        metrics = trainer.train_step(batch)
        trainer.save_checkpoint("/tmp/step_1000.pt")
    """

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute a single gradient-update step.

        Args:
            batch: A dictionary containing training tensors.  Expected
                keys vary by implementation but typically include
                ``"spatial"``, ``"vector"``, ``"action_mask"``,
                ``"policy"``, and ``"value"``.

        Returns:
            A dictionary of scalar metrics (e.g.
            ``{"total_loss": 0.42, "policy_loss": 0.3, "value_loss": 0.12}``).
        """
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save the full training state (model weights + optimiser).

        Args:
            path: Destination filesystem path.
        """
        ...

    def load_checkpoint(self, path: str) -> None:
        """Restore training state from a checkpoint.

        Args:
            path: Source path produced by a prior ``save_checkpoint`` call.
        """
        ...


# ---------------------------------------------------------------------------
# 10. StorageBackend
# ---------------------------------------------------------------------------


@runtime_checkable
class StorageBackend(Protocol):
    """Abstract key-value storage backend.

    Provides a uniform interface over local filesystems, cloud object
    stores (GCS, S3), or in-memory stores.  All values are opaque
    ``bytes``, leaving serialisation to the caller.

    Example::

        storage: StorageBackend = GCSBackend(bucket="my-bucket")
        storage.save("models/v1.pt", model_bytes)
        data = storage.load("models/v1.pt")
        assert storage.exists("models/v1.pt")
    """

    def save(self, key: str, data: bytes) -> None:
        """Persist *data* under the given *key*.

        Args:
            key: A string key (slash-separated path components are
                conventional).
            data: Raw bytes to store.

        Postconditions:
            - ``self.exists(key)`` is ``True`` after return.
        """
        ...

    def load(self, key: str) -> bytes:
        """Retrieve the data previously stored under *key*.

        Args:
            key: The key to load.

        Returns:
            The raw bytes associated with *key*.

        Preconditions:
            - ``self.exists(key)`` is ``True``.

        Raises:
            KeyError: If *key* does not exist.
        """
        ...

    def exists(self, key: str) -> bool:
        """Check whether *key* has been stored.

        Args:
            key: The key to probe.

        Returns:
            ``True`` if data exists under *key*, ``False`` otherwise.
        """
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys that start with *prefix*.

        Args:
            prefix: A string prefix to filter by.  An empty string
                matches all keys.

        Returns:
            A list of matching keys in unspecified order.
        """
        ...

    def delete(self, key: str) -> None:
        """Remove the data stored under *key*.

        Args:
            key: The key to delete.

        Postconditions:
            - ``self.exists(key)`` is ``False`` after return.
            - Deleting a non-existent key is a silent no-op.
        """
        ...


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def assert_valid_environment(env: StrategicEnvironment) -> None:
    """Validate that *env* satisfies all :class:`StrategicEnvironment` contracts.

    Performs a full behavioural contract check by exercising ``reset``,
    ``step``, ``clone``, ``get_observation``, ``get_valid_actions``,
    ``is_terminal``, and ``get_symmetries``.

    This function is intended for registration-time validation or test
    harnesses, **not** for hot-path runtime checks.

    Args:
        env: The environment instance to validate.

    Raises:
        ProtocolViolationError: If any contract check fails.  The
            exception's ``missing_methods`` attribute lists the specific
            violations discovered.

    Example::

        from src.core.protocols import assert_valid_environment

        env = MyEnv()
        assert_valid_environment(env)  # raises if contracts are broken
    """
    violations: list[str] = []

    # ------------------------------------------------------------------
    # Structural check: ensure required methods/properties exist
    # ------------------------------------------------------------------
    structural_missing = check_protocol_compliance(env, StrategicEnvironment)
    if structural_missing:
        raise ProtocolViolationError(
            f"{type(env).__name__} is missing StrategicEnvironment methods: "
            f"{', '.join(structural_missing)}",
            env_class=type(env),
            missing_methods=structural_missing,
        )

    # ------------------------------------------------------------------
    # Property checks
    # ------------------------------------------------------------------
    try:
        obs_spec = env.observation_spec
        if not isinstance(obs_spec, ObservationSpec):
            violations.append(
                f"observation_spec returned {type(obs_spec).__name__}, "
                f"expected ObservationSpec"
            )
    except Exception as exc:
        violations.append(f"observation_spec raised {type(exc).__name__}: {exc}")

    try:
        action_size = env.action_space_size
        if not isinstance(action_size, int) or action_size < 1:
            violations.append(
                f"action_space_size must be a positive int, got {action_size!r}"
            )
    except Exception as exc:
        violations.append(
            f"action_space_size raised {type(exc).__name__}: {exc}"
        )

    if violations:
        raise ProtocolViolationError(
            f"{type(env).__name__} property violations: "
            + "; ".join(violations),
            env_class=type(env),
            missing_methods=violations,
        )

    # ------------------------------------------------------------------
    # reset() contract
    # ------------------------------------------------------------------
    try:
        obs = env.reset(seed=0)
    except Exception as exc:
        raise ProtocolViolationError(
            f"{type(env).__name__}.reset(seed=0) raised "
            f"{type(exc).__name__}: {exc}",
            env_class=type(env),
            missing_methods=["reset"],
        ) from exc

    if not isinstance(obs, dict):
        violations.append(f"reset() returned {type(obs).__name__}, expected dict")
    else:
        if "spatial" not in obs:
            violations.append("reset() observation missing 'spatial' key")
        else:
            expected_spatial = env.observation_spec.spatial_shape
            actual_spatial = obs["spatial"].shape
            if actual_spatial != expected_spatial:
                violations.append(
                    f"reset() spatial shape {actual_spatial} != "
                    f"observation_spec.spatial_shape {expected_spatial}"
                )
        if "vector" not in obs:
            violations.append("reset() observation missing 'vector' key")
        else:
            expected_vector_dim = env.observation_spec.vector_dim
            actual_vector = obs["vector"].shape
            if actual_vector != (expected_vector_dim,):
                violations.append(
                    f"reset() vector shape {actual_vector} != "
                    f"expected ({expected_vector_dim},)"
                )

    # ------------------------------------------------------------------
    # is_terminal() after reset
    # ------------------------------------------------------------------
    try:
        if env.is_terminal():
            violations.append("is_terminal() is True immediately after reset()")
    except Exception as exc:
        violations.append(f"is_terminal() raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # get_valid_actions() after reset
    # ------------------------------------------------------------------
    try:
        valid = env.get_valid_actions()
        if valid.shape != (env.action_space_size,):
            violations.append(
                f"get_valid_actions() shape {valid.shape} != "
                f"({env.action_space_size},)"
            )
        if not np.any(valid):
            violations.append(
                "get_valid_actions() has no True entries after reset()"
            )
    except Exception as exc:
        violations.append(
            f"get_valid_actions() raised {type(exc).__name__}: {exc}"
        )

    # ------------------------------------------------------------------
    # get_observation()
    # ------------------------------------------------------------------
    try:
        current_obs = env.get_observation()
        if not isinstance(current_obs, dict):
            violations.append(
                f"get_observation() returned {type(current_obs).__name__}, "
                f"expected dict"
            )
    except Exception as exc:
        violations.append(
            f"get_observation() raised {type(exc).__name__}: {exc}"
        )

    # ------------------------------------------------------------------
    # step() contract -- take one valid action
    # ------------------------------------------------------------------
    try:
        valid_actions = env.get_valid_actions()
        action = int(np.argmax(valid_actions))
        result = env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            violations.append(
                f"step() returned {type(result).__name__} of length "
                f"{len(result) if isinstance(result, tuple) else 'N/A'}, "
                f"expected 5-tuple"
            )
        else:
            step_obs, reward, terminated, truncated, info = result
            if not isinstance(step_obs, dict):
                violations.append(
                    f"step() obs is {type(step_obs).__name__}, expected dict"
                )
            if not isinstance(reward, (int, float)):
                violations.append(
                    f"step() reward is {type(reward).__name__}, expected float"
                )
            elif not (-1.0 <= reward <= 1.0):
                violations.append(
                    f"step() reward {reward} outside [-1.0, 1.0]"
                )
            if not isinstance(terminated, bool):
                violations.append(
                    f"step() terminated is {type(terminated).__name__}, "
                    f"expected bool"
                )
            if not isinstance(truncated, bool):
                violations.append(
                    f"step() truncated is {type(truncated).__name__}, "
                    f"expected bool"
                )
            if not isinstance(info, dict):
                violations.append(
                    f"step() info is {type(info).__name__}, expected dict"
                )
    except ProtocolViolationError:
        raise
    except Exception as exc:
        violations.append(f"step() raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # clone() contract
    # ------------------------------------------------------------------
    # Reset to a clean state before testing clone
    try:
        env.reset(seed=1)
        cloned = env.clone()
        if cloned is env:
            violations.append("clone() returned the same object (not a copy)")
    except Exception as exc:
        violations.append(f"clone() raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # get_symmetries() contract
    # ------------------------------------------------------------------
    try:
        env.reset(seed=2)
        sym_obs = env.get_observation()
        dummy_policy = np.ones(env.action_space_size, dtype=np.float32)
        dummy_policy /= dummy_policy.sum()
        symmetries = env.get_symmetries(sym_obs["spatial"], dummy_policy)
        if not isinstance(symmetries, list):
            violations.append(
                f"get_symmetries() returned {type(symmetries).__name__}, "
                f"expected list"
            )
        elif len(symmetries) < 1:
            violations.append(
                "get_symmetries() returned empty list, expected at least "
                "the identity pair"
            )
    except Exception as exc:
        violations.append(
            f"get_symmetries() raised {type(exc).__name__}: {exc}"
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if violations:
        raise ProtocolViolationError(
            f"{type(env).__name__} has {len(violations)} contract "
            f"violation(s): " + "; ".join(violations),
            env_class=type(env),
            missing_methods=violations,
        )

    logger.debug(
        "%s passed all StrategicEnvironment contract checks",
        type(env).__name__,
    )


def check_protocol_compliance(obj: Any, protocol: type) -> list[str]:
    """Check which methods from *protocol* are missing on *obj*.

    This performs a deeper check than ``isinstance(obj, protocol)`` by
    inspecting every public method and property defined in the protocol
    class body (excluding dunder methods inherited from ``object``).

    Args:
        obj: The object to check.
        protocol: A :class:`~typing.Protocol` subclass.

    Returns:
        A sorted list of method/property names that *obj* does not
        provide.  An empty list means full compliance.

    Example::

        missing = check_protocol_compliance(my_env, StrategicEnvironment)
        if missing:
            print(f"Missing: {missing}")
        else:
            print("Fully compliant!")
    """
    missing: list[str] = []

    # Collect all names defined in the protocol's own annotations and
    # namespace, skipping inherited object dunders.
    protocol_members: set[str] = set()
    for name, _ in inspect.getmembers(protocol):
        if name.startswith("_"):
            continue
        # Only include names actually defined in the protocol or its
        # direct bases (not in object/Protocol).
        for klass in protocol.__mro__:
            if klass in (object, Protocol):
                continue
            if name in getattr(klass, "__annotations__", {}) or name in klass.__dict__:
                protocol_members.add(name)
                break

    for name in sorted(protocol_members):
        if not hasattr(obj, name):
            missing.append(name)
        else:
            # Verify that callable members are actually callable
            proto_attr = getattr(protocol, name, None)
            obj_attr = getattr(obj, name, None)
            if callable(proto_attr) and not callable(obj_attr):
                # Properties are not callable on instances but that is
                # expected -- they are accessed as attributes.  Only
                # flag non-property, non-callable mismatches.
                if not isinstance(
                    getattr(type(obj), name, None), property
                ):
                    missing.append(name)

    return sorted(set(missing))
