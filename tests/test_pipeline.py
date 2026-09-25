import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lipsync.config import ExperimentConfig
from lipsync.baseline import run


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "experiment.json"
        self.config.write_text(json.dumps({
            "name": "test", "backend": "wav2lip", "repository": "repo",
            "python": "runtime/python", "checkpoint": "weights/model.pth",
        }), encoding="utf-8")
        self.audio, self.face = self.root / "voice.wav", self.root / "face.mp4"
        self.audio.write_bytes(b"audio fixture")
        self.face.write_bytes(b"face fixture")

    def test_config_paths_are_relative_to_config(self):
        config = ExperimentConfig.load(self.config)
        self.assertEqual(config.repository, self.root / "repo")

    @patch("lipsync.baseline.subprocess.run")
    @patch("lipsync.baseline.probe", return_value={"streams": []})
    def test_dry_run_does_not_launch_backend_or_invent_metrics(self, probe, process):
        manifest_path = run(self.config, self.audio, self.face, self.root / "outputs", True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "planned")
        self.assertEqual(manifest["metrics"], {})
        self.assertFalse(Path(manifest["output"]).exists())
        process.assert_not_called()

    @patch("lipsync.baseline.probe", return_value={"streams": []})
    @patch("lipsync.baseline.shutil.which", return_value="ffmpeg")
    @patch("lipsync.baseline.Wav2LipBackend.validate")
    @patch("lipsync.baseline.sha256", return_value="test-hash")
    @patch("lipsync.baseline.subprocess.run")
    def test_failed_inference_persists_failure(self, process, *mocks):
        process.return_value.returncode = 2
        with self.assertRaisesRegex(RuntimeError, "Inference failed"):
            run(self.config, self.audio, self.face, self.root / "outputs")
        paths = list((self.root / "outputs").glob("*/manifest.json"))
        manifest = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["returncode"], 2)


if __name__ == "__main__":
    unittest.main()
