"""Experiment configuration. Relative paths resolve against the config file."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    repository: Path
    python: Path
    checkpoint: Path

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        path = path.resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"name", "backend", "repository", "python", "checkpoint"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError(f"Config must contain exactly: {', '.join(sorted(required))}")
        if any(not isinstance(value, str) or not value.strip() for value in data.values()):
            raise ValueError("All config values must be non-empty strings")
        if data["backend"] != "wav2lip":
            raise ValueError("Only the wav2lip backend is currently implemented")

        def resolve(value: str) -> Path:
            return (path.parent / Path(value).expanduser()).resolve()

        return cls(data["name"], resolve(data["repository"]),
                   resolve(data["python"]), resolve(data["checkpoint"]))
