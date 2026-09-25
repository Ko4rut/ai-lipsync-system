"""Composable research pipeline; concrete learned components are injected explicitly."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lipsync.audio import AudioPipeline
from lipsync.contracts import AlignedFeatures, FeatureSequence, PipelineResult
from lipsync.evaluation.base import Evaluator
from lipsync.fusion import FusionModule
from lipsync.generation import LipGenerator, VideoRenderer
from lipsync.video import VideoPipeline


class TemporalAligner(Protocol):
    def __call__(self, audio: FeatureSequence, visual: FeatureSequence) -> AlignedFeatures:
        ...


@dataclass
class LipSyncPipeline:
    audio: AudioPipeline
    video: VideoPipeline
    align: TemporalAligner
    fusion: FusionModule
    generate: LipGenerator
    render: VideoRenderer
    evaluate: Evaluator | None = None

    def run(self, audio_path: Path, face_path: Path, output_path: Path) -> PipelineResult:
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("Output must have an .mp4 extension")
        if output_path.exists():
            raise ValueError(f"Refusing to overwrite an existing output: {output_path}")
        for path in (audio_path, face_path):
            if not path.is_file():
                raise ValueError(f"Input file not found: {path}")
        audio_features = self.audio.run(audio_path)
        visual = self.video.run(face_path)
        aligned = self.align(audio_features, visual.features)
        if aligned.visual.timestamps != visual.track.source.timestamps:
            raise ValueError("Alignment changed the reference frame timeline")
        fused = self.fusion(aligned)
        if fused.timestamps != visual.track.source.timestamps:
            raise ValueError("Fusion changed the reference frame timeline")
        generated = self.generate(fused, visual.track)
        if generated.timestamps != visual.track.source.timestamps:
            raise ValueError("Generated frames do not match the reference timeline")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = self.render(generated, visual.track, audio_path, output_path)
        if result.resolve() != output_path.resolve() or not result.is_file() or not result.stat().st_size:
            raise RuntimeError("Renderer must produce a non-empty file at the requested output path")
        metrics = self.evaluate(result, audio_path) if self.evaluate is not None else {}
        return PipelineResult(result, metrics)
