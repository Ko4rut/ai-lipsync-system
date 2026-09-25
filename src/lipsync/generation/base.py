from pathlib import Path
from typing import Protocol

from lipsync.contracts import FaceTrack, FeatureSequence, VideoFrames


class LipGenerator(Protocol):
    def __call__(self, fused: FeatureSequence, reference: FaceTrack) -> VideoFrames:
        """Generate face crops on the reference timeline using trained weights."""
        ...


class VideoRenderer(Protocol):
    def __call__(self, generated: VideoFrames, reference: FaceTrack,
                 audio: Path, output: Path) -> Path:
        """Refine/blend crops using reference transforms, encode frames and mux audio."""
        ...
