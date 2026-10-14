from pathlib import Path

from src.lipsync.setting.config import LipSyncSettings
from src.lipsync.evaluation.duration_report import DurationGapReporter


def main() -> None:
    settings = LipSyncSettings.from_env()

    if settings.data_root is None:
        raise RuntimeError("LIPSYNC_DATA_ROOT is not configured")

    project_root = Path(__file__).resolve().parents[4]

    reporter = DurationGapReporter(
        root=settings.data_root,
        output_dir=project_root / "outputs" / "duration_gap",
        fps=settings.fps,
        threshold_ms=settings.duration_gap_threshold_ms,
    )

    report_paths = reporter.generate()

    for name, path in report_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()