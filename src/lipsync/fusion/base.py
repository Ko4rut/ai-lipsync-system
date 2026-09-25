from typing import Protocol

from lipsync.contracts import AlignedFeatures, FeatureSequence


class FusionModule(Protocol):
    def __call__(self, features: AlignedFeatures) -> FeatureSequence:
        """Combine aligned modalities, preserving the output frame timeline."""
        ...
