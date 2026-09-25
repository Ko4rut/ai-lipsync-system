import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from lipsync.alignment import LinearTemporalAligner
from lipsync.audio import AudioPipeline
from lipsync.contracts import AudioSignal, FaceTrack, FeatureSequence, VideoFrames
from lipsync.fusion import ConcatenationFusion
from lipsync.pipeline import LipSyncPipeline
from lipsync.video import VideoPipeline


class FeatureTests(unittest.TestCase):
    def test_alignment_interpolates_by_time_and_fusion_preserves_order(self):
        audio = FeatureSequence(((0.0, 2.0), (8.0, 6.0)), (0.0, 0.08))
        visual = FeatureSequence(((10.0,), (20.0,), (30.0,)), (0.0, 0.02, 0.08))
        aligned = LinearTemporalAligner()(audio, visual)
        fused = ConcatenationFusion()(aligned)
        self.assertEqual(fused.values, ((0.0, 2.0, 10.0), (2.0, 3.0, 20.0), (8.0, 6.0, 30.0)))
        self.assertEqual(fused.timestamps, visual.timestamps)

    def test_outside_audio_coverage_requires_explicit_hold_policy(self):
        audio = FeatureSequence(((3.0,),), (0.04,))
        visual = FeatureSequence(((1.0,), (2.0,)), (0.0, 0.08))
        with self.assertRaisesRegex(ValueError, "coverage"):
            LinearTemporalAligner()(audio, visual)
        self.assertEqual(LinearTemporalAligner("hold")(audio, visual).audio.values,
                         ((3.0,), (3.0,)))

    def test_invalid_features_rejected(self):
        for values, times in (
            ((), ()), (((1.0,),), (0.0, 0.1)),
            (((1.0,), (2.0,)), (0.0, 0.0)),
            (((1.0,), (2.0, 3.0)), (0.0, 0.1)),
            (((float("nan"),),), (0.0,)),
        ):
            with self.subTest(values=values, times=times), self.assertRaises(ValueError):
                FeatureSequence(values, times)


class ResearchPipelineTests(unittest.TestCase):
    def test_components_receive_aligned_features_and_source_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_path, face_path = root / "audio.wav", root / "face.mp4"
            audio_path.write_bytes(b"fixture")
            face_path.write_bytes(b"fixture")
            signal = AudioSignal((0.0, 1.0), 16000)
            acoustic = FeatureSequence(((1.0,), (3.0,)), (0.0, 0.08))
            temporal = FeatureSequence(((2.0,), (6.0,)), (0.0, 0.08))
            frames = VideoFrames(("frame",), (0.04,))
            track = FaceTrack(frames, ("crop",), ("landmarks",), ("transform",))
            visual = FeatureSequence(((9.0,),), (0.04,))
            generated = VideoFrames(("generated crop",), (0.04,))
            preprocess_audio = Mock(return_value=signal)
            extract_audio = Mock(return_value=acoustic)
            temporal_model = Mock(return_value=temporal)
            process_faces = Mock(return_value=track)
            generate = Mock(return_value=generated)

            def render(crops, reference, audio, output):
                self.assertIs(crops, generated)
                self.assertIs(reference, track)
                self.assertEqual(audio, audio_path)
                output.write_bytes(b"renderer fixture, not a real video")
                return output

            pipeline = LipSyncPipeline(
                AudioPipeline(preprocess_audio, extract_audio, temporal_model),
                VideoPipeline(Mock(return_value=frames), process_faces, Mock(return_value=visual)),
                LinearTemporalAligner(), ConcatenationFusion(), generate, render,
            )
            result = pipeline.run(audio_path, face_path, root / "result.mp4")
            extract_audio.assert_called_once_with(signal)
            temporal_model.assert_called_once_with(acoustic)
            process_faces.assert_called_once_with(frames)
            fused, reference = generate.call_args.args
            self.assertEqual(fused.values, ((4.0, 9.0),))
            self.assertIs(reference, track)
            self.assertEqual(result.metrics, {})
            with self.assertRaisesRegex(ValueError, "overwrite"):
                pipeline.run(audio_path, face_path, result.video)

            generate.return_value = VideoFrames(("wrong timestamp",), (0.0,))
            with self.assertRaisesRegex(ValueError, "reference timeline"):
                pipeline.run(audio_path, face_path, root / "invalid.mp4")
            self.assertFalse((root / "invalid.mp4").exists())
