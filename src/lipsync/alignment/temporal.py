"""Deterministic feature interpolation; not learned phoneme/viseme alignment."""

from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal

from lipsync.contracts import AlignedFeatures, FeatureSequence


@dataclass(frozen=True)
class LinearTemporalAligner:
    boundary: Literal["error", "hold"] = "error"

    def __post_init__(self) -> None:
        if self.boundary not in ("error", "hold"):
            raise ValueError("Boundary policy must be error or hold")

    def __call__(self, audio: FeatureSequence, visual: FeatureSequence) -> AlignedFeatures:
        times = audio.timestamps
        rows = []
        for time in visual.timestamps:
            if time < times[0] or time > times[-1]:
                if self.boundary == "error":
                    raise ValueError(f"Video timestamp {time} is outside audio feature coverage")
                rows.append(audio.values[0] if time < times[0] else audio.values[-1])
                continue
            right = bisect_left(times, time)
            if times[right] == time:
                rows.append(audio.values[right])
                continue
            left = right - 1
            weight = (time - times[left]) / (times[right] - times[left])
            rows.append(tuple(a + weight * (b - a)
                              for a, b in zip(audio.values[left], audio.values[right])))
        return AlignedFeatures(FeatureSequence(tuple(rows), visual.timestamps), visual)
