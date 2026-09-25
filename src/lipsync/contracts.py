"""Unbatched research data contracts. Times use seconds on a shared timeline.

Image payloads are owned by concrete adapters; numeric features use [time, dim].
This reference representation is for orchestration, not differentiable training.
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


def validate_times(times: tuple[float, ...]) -> None:
    if not times or any(not isfinite(t) or t < 0 for t in times):
        raise ValueError("Timestamps must be non-empty, finite and non-negative")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Timestamps must be strictly increasing")


@dataclass(frozen=True)
class AudioSignal:
    samples: tuple[float, ...]  # mono waveform
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or not self.samples:
            raise ValueError("Audio requires samples and a positive sample rate")
        if any(not isfinite(s) for s in self.samples):
            raise ValueError("Audio samples must be finite")


@dataclass(frozen=True)
class FeatureSequence:
    values: tuple[tuple[float, ...], ...]
    timestamps: tuple[float, ...]

    def __post_init__(self) -> None:
        validate_times(self.timestamps)
        if len(self.values) != len(self.timestamps):
            raise ValueError("Each feature vector needs one timestamp")
        width = len(self.values[0])
        if not width or any(len(row) != width for row in self.values):
            raise ValueError("Feature vectors must have the same non-zero dimension")
        if any(not isfinite(v) for row in self.values for v in row):
            raise ValueError("Features must be finite")


@dataclass(frozen=True)
class VideoFrames:
    frames: tuple[object, ...]
    timestamps: tuple[float, ...]

    def __post_init__(self) -> None:
        validate_times(self.timestamps)
        if len(self.frames) != len(self.timestamps):
            raise ValueError("Each frame needs one timestamp")


@dataclass(frozen=True)
class FaceTrack:
    source: VideoFrames
    crops: tuple[object, ...]
    landmarks: tuple[object, ...]
    transforms: tuple[object, ...]  # crop-to-source transforms for rendering

    def __post_init__(self) -> None:
        length = len(self.source.frames)
        if any(len(items) != length for items in (self.crops, self.landmarks, self.transforms)):
            raise ValueError("Face crops, landmarks and transforms must cover every frame")


@dataclass(frozen=True)
class VisualRepresentation:
    track: FaceTrack
    features: FeatureSequence

    def __post_init__(self) -> None:
        if self.features.timestamps != self.track.source.timestamps:
            raise ValueError("Visual features must follow the source frame timeline")


@dataclass(frozen=True)
class AlignedFeatures:
    audio: FeatureSequence
    visual: FeatureSequence

    def __post_init__(self) -> None:
        if self.audio.timestamps != self.visual.timestamps:
            raise ValueError("Aligned audio and visual features must share timestamps")


@dataclass(frozen=True)
class PipelineResult:
    video: Path
    metrics: dict[str, float]
