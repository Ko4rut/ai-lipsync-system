"""Configuration and quality-selection tests; no real GRID files needed."""
from __future__ import annotations

from dataclasses import replace

import pytest

from lipsync.setting.config import LipSyncSettings
from src.lipsync.datasets import grid_dataset
from src.lipsync.datasets.manifest import ManifestRecord


def make_record(uid: str, frames: int) -> ManifestRecord:
    return ManifestRecord(
        split="train",
        speaker_id="s1",
        utterance_id=uid,
        shard_member_prefix=f"s1/{uid}/",
        frame_count=frames,
        first_frame_index=0,
        last_frame_index=frames - 1,
        face_detection_ratio=1.0,
        audio_duration_sec=1.2,
        audio_sample_rate=16000,
        source_shard_path="s1.tar",
    )


@pytest.fixture()
def stub_manifest(monkeypatch):
    records = [make_record("short", 5), make_record("normal", 75)]

    class FakeManifest:
        def __init__(self, root, split):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index]

    class FakeReader:
        def __init__(self, root, **kwargs):
            self.calls = 0

        def read_utterance(self, record):
            self.calls += 1
            raise RuntimeError("read reached")

        def close(self):
            pass

    monkeypatch.setattr(grid_dataset, "GoldManifestIndex", FakeManifest)
    monkeypatch.setattr(grid_dataset, "GridShardReader", FakeReader)
    return records


def test_no_filter_by_default(tmp_path, stub_manifest):
    dataset = grid_dataset.GridUtteranceDataset(
        tmp_path, "train", settings=LipSyncSettings()
    )
    assert len(dataset) == 2
    assert dataset._eligible_indices == [0, 1]
    dataset.close()


def test_filter_skips_short_record_before_tar_read(tmp_path, stub_manifest):
    dataset = grid_dataset.GridUtteranceDataset(
        tmp_path, "train", min_frame_count=50, settings=LipSyncSettings()
    )
    assert len(dataset.index) == 2
    assert len(dataset) == 1
    assert dataset._eligible_indices == [1]
    assert dataset.reader.calls == 0
    with pytest.raises(RuntimeError, match="read reached"):
        dataset[0]
    assert dataset.reader.calls == 1
    dataset.close()


def test_settings_threshold_applied(tmp_path, stub_manifest):
    settings = LipSyncSettings(min_frame_count=6, fps=30.0)
    dataset = grid_dataset.GridUtteranceDataset(
        tmp_path, "train", settings=settings
    )
    assert len(dataset) == 1
    assert dataset.fps == 30.0
    dataset.close()


def test_explicit_args_override_settings(tmp_path, stub_manifest):
    settings = LipSyncSettings(min_frame_count=50, fps=30.0, frame_size=(32, 32))
    dataset = grid_dataset.GridUtteranceDataset(
        tmp_path, "train", settings=settings, min_frame_count=1,
        fps=25.0, frame_size=(96, 96)
    )
    assert len(dataset) == 2
    assert dataset.fps == 25.0
    assert dataset.frame_size == (96, 96)
    dataset.close()


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True])
def test_invalid_minimum(tmp_path, minimum):
    with pytest.raises(ValueError, match="min_frame_count"):
        grid_dataset.GridUtteranceDataset(
            tmp_path, "train", min_frame_count=minimum,
            settings=LipSyncSettings()
        )


def test_env_file_and_os_precedence(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "LIPSYNC_DATA_ROOT=placeholder_dataset\n"
        "LIPSYNC_FPS=24\n"
        "LIPSYNC_FRAME_WIDTH=64\n"
        "LIPSYNC_FRAME_HEIGHT=72\n"
        "LIPSYNC_MIN_FRAME_COUNT=50\n"
        "LIPSYNC_DURATION_GAP_THRESHOLD_MS=90\n",
        encoding="utf-8",
    )
    for key in ("LIPSYNC_DATA_ROOT", "LIPSYNC_FPS", "LIPSYNC_FRAME_WIDTH",
                "LIPSYNC_FRAME_HEIGHT", "LIPSYNC_MIN_FRAME_COUNT",
                "LIPSYNC_DURATION_GAP_THRESHOLD_MS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LIPSYNC_MIN_FRAME_COUNT", "60")
    settings = LipSyncSettings.from_env(path)
    assert settings.data_root is not None
    assert settings.fps == 24.0
    assert settings.frame_size == (64, 72)
    assert settings.min_frame_count == 60
    assert settings.duration_gap_threshold_ms == 90.0


def test_blank_minimum_disables_filter(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("LIPSYNC_MIN_FRAME_COUNT=\n", encoding="utf-8")
    monkeypatch.delenv("LIPSYNC_MIN_FRAME_COUNT", raising=False)
    settings = LipSyncSettings.from_env(path)
    assert settings.min_frame_count is None


@pytest.mark.parametrize("value", ["oops", "-4", "nan", "inf"])
def test_invalid_fps_env(tmp_path, monkeypatch, value):
    monkeypatch.setenv("LIPSYNC_FPS", value)
    with pytest.raises(ValueError):
        LipSyncSettings.from_env(tmp_path / "missing.env")
