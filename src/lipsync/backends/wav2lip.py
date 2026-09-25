from pathlib import Path

from lipsync.config import ExperimentConfig


class Wav2LipBackend:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def validate(self) -> None:
        for label, path in (
            ("Wav2Lip inference script", self.config.repository / "inference.py"),
            ("Backend Python interpreter", self.config.python),
            ("Model checkpoint", self.config.checkpoint),
        ):
            if not path.is_file():
                raise ValueError(f"{label} not found: {path}")

    def command(self, audio: Path, face: Path, output: Path) -> list[str]:
        return [
            str(self.config.python), str(self.config.repository / "inference.py"),
            "--checkpoint_path", str(self.config.checkpoint),
            "--face", str(face), "--audio", str(audio), "--outfile", str(output),
        ]
