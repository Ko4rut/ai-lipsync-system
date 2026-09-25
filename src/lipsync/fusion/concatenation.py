from lipsync.contracts import AlignedFeatures, FeatureSequence


class ConcatenationFusion:
    """Early fusion reference: [T, Da] + [T, Dv] -> [T, Da + Dv]."""

    def __call__(self, features: AlignedFeatures) -> FeatureSequence:
        return FeatureSequence(
            tuple(a + v for a, v in zip(features.audio.values, features.visual.values)),
            features.visual.timestamps,
        )
