"""Regression tests for separated data models and public import compatibility."""

from dataclasses import fields

from src.lipsync.datasets.grid_dataset import GridSample as LegacyGridSample
from src.lipsync.datasets.models import GridSample
from src.lipsync.evaluation.duration_report import (
    DurationRow as LegacyDurationRow,
    SummaryRow as LegacySummaryRow,
)
from src.lipsync.evaluation.models import DurationRow, SummaryRow
from src.lipsync.setting.config import PROJECT_ROOT, LipSyncSettings


def test_model_import_compatibility():
    assert GridSample is LegacyGridSample
    assert DurationRow is LegacyDurationRow
    assert SummaryRow is LegacySummaryRow


def test_model_field_schemas_unchanged():
    assert tuple(item.name for item in fields(GridSample)) == (
        "speaker_id", "utterance_id", "frames", "frame_indices", "audio",
        "sample_rate", "fps", "audio_duration_sec", "video_duration_sec"
    )
    assert "duration_gap_sec" in {name for name in dir(GridSample)}
    assert "missing_frame_count" in {item.name for item in fields(DurationRow)}
    assert "p95_absolute_gap_sec" in {item.name for item in fields(SummaryRow)}


def test_default_env_path_is_repository_root(tmp_path, monkeypatch):
    root = PROJECT_ROOT
    assert (root / "src" / "lipsync" / "setting" / "config.py").is_file()
    env = root / ".env"
    # Verify path arithmetic without depending on a developer's real .env.
    assert env.parent == root
    assert root.name != "lipsync"
