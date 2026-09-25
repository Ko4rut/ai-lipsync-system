from pathlib import Path
from typing import Protocol


class InferenceBackend(Protocol):
    def validate(self) -> None:
        """Check model artifacts and runtime availability."""
        ...

    def command(self, audio: Path, face: Path, output: Path) -> list[str]:
        """Build a subprocess argument list without a shell."""
        ...
