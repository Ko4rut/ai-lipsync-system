from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lipsync.contracts import AudioSignal, FeatureSequence


class AudioPreprocessor(Protocol):
    def __call__(self, path: Path) -> AudioSignal:
        """Decode, convert to mono and resample at the encoder's sample rate."""
        ...


class AudioFeatureExtractor(Protocol):
    def __call__(self, signal: AudioSignal) -> FeatureSequence:
        """Extract mel/MFCC or pretrained phonetic features with timestamps."""
        ...


class AudioTemporalModel(Protocol):
    def __call__(self, features: FeatureSequence) -> FeatureSequence:
        """Model neighbouring speech units using CNN/RNN/Transformer context."""
        ...


@dataclass
class AudioPipeline:
    preprocess: AudioPreprocessor
    extract: AudioFeatureExtractor
    temporal: AudioTemporalModel

    def run(self, path: Path) -> FeatureSequence:
        signal = self.preprocess(path)
        acoustic_features = self.extract(signal)
        return self.temporal(acoustic_features)