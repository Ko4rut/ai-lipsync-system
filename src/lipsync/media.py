"""Read media metadata through ffprobe before model inference."""

import json
import shutil
import subprocess
from pathlib import Path


def probe(path: Path, expected_stream: str) -> dict:
    if not path.is_file():
        raise ValueError(f"Input file not found: {path}")
    executable = shutil.which("ffprobe")
    if not executable:
        raise ValueError("ffprobe is required on PATH (install FFmpeg)")
    result = subprocess.run(
        [executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    if result.returncode:
        raise ValueError(f"Cannot inspect {path}: {result.stderr.strip()}")
    metadata = json.loads(result.stdout)
    if not any(s.get("codec_type") == expected_stream for s in metadata.get("streams", [])):
        raise ValueError(f"Input has no {expected_stream} stream: {path}")
    return metadata
