from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lipsync.contracts import FaceTrack, FeatureSequence, VideoFrames, VisualRepresentation


class VideoPreprocessor(Protocol):
    def __call__(self, path: Path) -> VideoFrames:
        """Decode frames with timestamps and an explicit duration policy."""
        ...


class FaceProcessor(Protocol):
    def __call__(self, frames: VideoFrames) -> FaceTrack:
        """Detect/track a face, estimate landmarks, align crops and retain transforms.

        Implementations must define missing-face and multiple-face policies.
        """
        ...


class VisualFeatureExtractor(Protocol):
    def __call__(self, track: FaceTrack) -> FeatureSequence:
        """Encode mouth, facial geometry and identity on the frame timeline."""
        ...


@dataclass
class VideoPipeline:
    preprocess: VideoPreprocessor
    process_faces: FaceProcessor
    extract: VisualFeatureExtractor

    def run(self, path: Path) -> VisualRepresentation:
        frames = self.preprocess(path)
        track = self.process_faces(frames)
        return VisualRepresentation(track, self.extract(track))
