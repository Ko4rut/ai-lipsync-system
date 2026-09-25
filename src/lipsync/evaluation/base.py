from pathlib import Path
from typing import Protocol


class Evaluator(Protocol):
    def __call__(self, video: Path, audio: Path) -> dict[str, float]:
        """Return measured metrics; paired image metrics need separate ground truth."""
        ...
