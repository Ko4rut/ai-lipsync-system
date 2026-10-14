"""Central application settings for GRID dataset inspection and experiments.

Resolution order: process environment > repository .env > defaults below.
The .env file is read without mutating os.environ.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LipSyncSettings:
    """Tunable application settings. No TAR/media is accessed here."""

    data_root: Path | None = None
    fps: float = 25.0
    frame_size: tuple[int, int] = (96, 96)  # width, height
    min_frame_count: int | None = None  # None keeps all valid manifest rows
    duration_gap_threshold_ms: float = 80.0  # reporting only, never a filter

    def __post_init__(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")

        if len(self.frame_size) != 2:
            raise ValueError("frame_size must contain width and height")
        width, height = self.frame_size
        if (type(width) is not int or type(height) is not int
                or width <= 0 or height <= 0):
            raise ValueError("frame_size width and height must be positive integers")

        if self.min_frame_count is not None:
            if type(self.min_frame_count) is not int or self.min_frame_count < 1:
                raise ValueError("min_frame_count must be a positive integer")

        if (not math.isfinite(self.duration_gap_threshold_ms)
                or self.duration_gap_threshold_ms < 0):
            raise ValueError("duration_gap_threshold_ms must be finite and >= 0")

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> LipSyncSettings:
        """Read .env and OS variables, keeping Python defaults when unset.

        OS environment wins over .env. Missing or blank optional values keep
        their documented defaults; an unset/blank min-frame count disables QC
        filtering without affecting the underlying Gold manifest.
        """
        dotenv_path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
        dotenv_data = dotenv_values(dotenv_path) if dotenv_path.is_file() else {}

        def get_value(name: str) -> str | None:
            if name in os.environ:
                return os.environ[name]
            return dotenv_data.get(name)

        def get_float(name: str, default: float) -> float:
            raw = get_value(name)
            if raw is None or not raw.strip():
                return default
            try:
                return float(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be a number: {raw!r}") from error

        def get_int(name: str, default: int) -> int:
            raw = get_value(name)
            if raw is None or not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer: {raw!r}") from error

        root_value = get_value("LIPSYNC_DATA_ROOT")
        data_root = None
        if root_value is not None and root_value.strip():
            data_root = Path(root_value.strip()).expanduser()

        min_frame_count = None
        raw_minimum = get_value("LIPSYNC_MIN_FRAME_COUNT")
        if raw_minimum is not None and raw_minimum.strip():
            min_frame_count = get_int("LIPSYNC_MIN_FRAME_COUNT", 0)

        return cls(
            data_root=data_root,
            fps=get_float("LIPSYNC_FPS", 25.0),
            frame_size=(
                get_int("LIPSYNC_FRAME_WIDTH", 96),
                get_int("LIPSYNC_FRAME_HEIGHT", 96),
            ),
            min_frame_count=min_frame_count,
            duration_gap_threshold_ms=get_float(
                "LIPSYNC_DURATION_GAP_THRESHOLD_MS", 80.0
            ),
        )
