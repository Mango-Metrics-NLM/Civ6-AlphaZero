"""Deterministic derivation of per-stream seeds from one run seed.

Why this module exists
----------------------
``training_civ6`` derived its RNG seeds by adding hand-picked offsets to
``config.seed``: ``seed + game_idx`` for self-play environments,
``seed + num_games`` for the search, ``seed + EVAL_SEED_OFFSET`` for evaluation.
Two problems followed from that scheme.

1. **No iteration term.** Only the DataLoader's generator varied with the
   training iteration. Self-play and evaluation did not, so a seeded run
   regenerated the *same* games every iteration and scored them on the *same*
   fixed maps -- a 100-iteration run saw as few as ``num_games`` distinct
   starting positions, and the evaluation curve was a fixed-sample probe rather
   than a learning signal.
2. **Collisions were prevented by arithmetic nobody could check.** Whether
   ``seed + game_idx`` overlapped ``seed + num_games`` depended on the game
   count at the call site; the offsets had to be re-reasoned every time one
   moved.

Deriving each stream from a *name* removes both problems. Streams cannot
collide unless they are given the same name, adding an iteration is a parameter
rather than a stride, and the mapping is stable across processes and platforms
because it uses BLAKE2b rather than Python's salted ``hash``.

Backwards compatibility
-----------------------
``derive_seed(None, ...)`` returns ``None``, so an unseeded run stays unseeded
and every caller keeps its "no seed means non-deterministic" behaviour. Seeded
runs draw different streams than they did before this module existed -- the
sequences change, reproducibility does not, and the determinism regression
suite asserts the property (same seed twice is identical, different seeds
differ) rather than any literal value.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

__all__ = ["MAX_SEED", "SeedStream", "derive_seed"]

# numpy's SeedSequence and torch both accept a 32-bit seed; keep well inside it.
MAX_SEED = 2**31 - 1

_DIGEST_BYTES = 8
_SEPARATOR = "|"


class SeedStream:
    """Canonical names for the independent RNG streams in a training run.

    Names, not numbers: a new stream is a new constant here, and it cannot
    collide with an existing one unless it is spelled the same. Keeping them in
    one place also means a reader can enumerate every source of randomness in
    the pipeline without grepping for ``seed +``.
    """

    SELF_PLAY_ENV = "selfplay.env"
    SELF_PLAY_SEARCH = "selfplay.search"
    EVAL_ENV = "eval.env"
    EVAL_SEARCH = "eval.search"
    EVAL_OPPONENT = "eval.opponent"
    REPLAY_SAMPLER = "replay.sampler"
    DATA_LOADER = "data.loader"


def derive_seed(base_seed: int | None, stream: str, *parts: object) -> int | None:
    """Derive a stable sub-seed for one named stream.

    Args:
        base_seed: The run's seed. ``None`` propagates as ``None`` so unseeded
            runs stay unseeded.
        stream: The stream name, normally a :class:`SeedStream` constant.
        *parts: Additional discriminators, typically the iteration index and a
            per-game index. Rendered with ``str`` so ints, strings and enums all
            work.

    Returns:
        A reproducible seed in ``[0, MAX_SEED]``, or ``None`` when ``base_seed``
        is ``None``.

    Examples:
        Two iterations of self-play draw different environments::

            >>> a = derive_seed(42, SeedStream.SELF_PLAY_ENV, 0, 3)
            >>> b = derive_seed(42, SeedStream.SELF_PLAY_ENV, 1, 3)
            >>> a == b
            False

        The same coordinates always give the same seed::

            >>> derive_seed(42, SeedStream.SELF_PLAY_ENV, 0, 3) == a
            True
    """
    if base_seed is None:
        return None
    if not stream:
        raise ValueError("stream must be a non-empty name")

    payload = _SEPARATOR.join([str(base_seed), stream, *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    derived = int.from_bytes(digest, "big") % (MAX_SEED + 1)
    logger.debug("derived seed %d for stream %r from %r", derived, stream, payload)
    return derived
