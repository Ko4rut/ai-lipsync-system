"""Data objects returned by the GRID dataset loader.

These models contain data and lightweight derived properties only.
Reading TAR files, decoding media, and filtering stay in grid_dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridSample:
    """One decoded utterance: RGB frames and mono PCM float32 audio."""

    speaker_id: str
    utterance_id: str
    frames: np.ndarray
    frame_indices: tuple[int, ...]
    audio: np.ndarray
    sample_rate: int
    fps: float
    audio_duration_sec: float
    video_duration_sec: float

    @property
    def duration_gap_sec(self) -> float:
        """Signed difference between video frame span and WAV duration."""
        return self.video_duration_sec - self.audio_duration_sec
