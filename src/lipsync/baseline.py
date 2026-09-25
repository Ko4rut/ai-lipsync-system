"""Run an external baseline as a whole, without exposing its internal stages."""

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from lipsync.backends.wav2lip import Wav2LipBackend
from lipsync.config import ExperimentConfig
from lipsync.media import probe


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config_path: Path, audio: Path, face: Path, output_root: Path,
        dry_run: bool = False) -> Path:
    config = ExperimentConfig.load(config_path)
    audio, face = audio.resolve(), face.resolve()
    audio_info = probe(audio, "audio")
    face_info = probe(face, "video")
    backend = Wav2LipBackend(config)
    # Dry runs validate media and record the plan without requiring model artifacts.
    if not dry_run:
        if not shutil.which("ffmpeg"):
            raise ValueError("ffmpeg is required on PATH")
        backend.validate()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / "result.mp4"
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if dry_run else "running",
        "config_path": str(config_path.resolve()),
        "config": {k: str(v) for k, v in asdict(config).items()},
        "backend": "wav2lip",
        "inputs": {
            "audio": {"path": str(audio), "sha256": sha256(audio), "metadata": audio_info},
            "face": {"path": str(face), "sha256": sha256(face), "metadata": face_info},
        },
        "command": backend.command(audio, face, output),
        "working_directory": str(config.repository),
        "output": str(output),
        "metrics": {},
    }

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    if dry_run:
        return manifest_path
    try:
        manifest["checkpoint_sha256"] = sha256(config.checkpoint)
        manifest["inference_script_sha256"] = sha256(config.repository / "inference.py")
        save()
        with (run_dir / "inference.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(manifest["command"], cwd=config.repository,
                                    stdout=log, stderr=subprocess.STDOUT)
        manifest["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Inference failed; see {run_dir / 'inference.log'}")
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError("Backend finished without producing a non-empty video")
        manifest["output_metadata"] = probe(output, "video")
        probe(output, "audio")
        manifest["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error) or type(error).__name__
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    return manifest_path
