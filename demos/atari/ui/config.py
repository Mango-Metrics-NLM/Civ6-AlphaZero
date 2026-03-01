"""Configuration for the Atari UI web server."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AtariUIConfig:
    """Configuration for the Atari web UI server.

    Attributes:
        host: Server bind address.
        port: Server port.
        game: Atari game name (e.g. "Pong", "Breakout").
        model_path: Path to a trained model checkpoint (optional).
        device: PyTorch device string.
        mcts_simulations: MCTS simulations per AI move.
        debug: Enable FastAPI debug mode.
        cors_origins: Allowed CORS origins.
        replay_dir: Directory for saved episode replays.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    game: str = "Pong"
    model_path: str | None = None
    device: str = ""
    mcts_simulations: int = 50
    debug: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    replay_dir: str = "replays/atari"

    def __post_init__(self) -> None:
        if not self.device:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        Path(self.replay_dir).mkdir(parents=True, exist_ok=True)
