import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from lipsync.config import ExperimentConfig
from lipsync.backends.wav2lip import Wav2LipBackend
from lipsync.baseline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LipSync research experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check tools and optional baseline artifacts")
    doctor.add_argument("--config", type=Path)
    infer = commands.add_parser("infer", help="Run Wav2Lip or validate a planned experiment")
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--audio", type=Path, required=True)
    infer.add_argument("--face", type=Path, required=True)
    infer.add_argument("--output-root", type=Path, default=Path("outputs"))
    infer.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            checks = {tool: shutil.which(tool) for tool in ("ffmpeg", "ffprobe")}
            print(json.dumps(checks, indent=2))
            if args.config:
                Wav2LipBackend(ExperimentConfig.load(args.config)).validate()
                print("Baseline paths: OK (model loading has not been tested)")
            return 0 if all(checks.values()) else 1
        manifest = run(args.config, args.audio, args.face, args.output_root, args.dry_run)
        print(f"{'Plan saved' if args.dry_run else 'Inference completed'}: {manifest}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
