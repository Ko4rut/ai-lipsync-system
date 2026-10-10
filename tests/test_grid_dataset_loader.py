"""Small synthetic TAR + manifest tests: no real GRID dataset or Colab needed."""
from __future__ import annotations

import csv
import io
import tarfile
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.lipsync.datasets import (
    DatasetReadError, GoldManifestIndex, GridUtteranceDataset, ManifestError,
)


FIELDS = [
    "split", "speaker_id", "utterance_id", "shard_path", "shard_member_prefix",
    "frame_count", "first_frame_index", "last_frame_index", "face_detection_ratio",
    "audio_duration_sec", "audio_sample_rate", "valid",
]


def fixture_dataset(tmp_path: Path, *, corrupt_audio: bool = False, gap: bool = False) -> Path:
    gold = tmp_path / "gold"
    shards = tmp_path / "silver" / "shards"
    gold.mkdir(parents=True)
    shards.mkdir(parents=True)
    indices = (0, 1, 3) if gap else (0, 1, 2)
    frames = []
    for i in indices:
        image = np.full((12, 12, 3), i * 35, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        frames.append((f"s1/bbaf2n/{i}.jpg", encoded.tobytes()))
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    frames.append(("s1/bbaf2n/audio.wav", b"bad" if corrupt_audio else output.getvalue()))
    with tarfile.open(shards / "s1.tar", "w") as tar:
        for name, data in frames:
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    row = {
        "split": "train", "speaker_id": "s1", "utterance_id": "bbaf2n",
        "shard_path": r"G:\old\dataset\silver\shards\s1.tar",
        "shard_member_prefix": "s1/bbaf2n/", "frame_count": 3,
        "first_frame_index": 0, "last_frame_index": indices[-1],
        "face_detection_ratio": 1, "audio_duration_sec": 1,
        "audio_sample_rate": 16000, "valid": "True",
    }
    with (gold / "train_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return tmp_path


def test_read_from_relocated_shard(tmp_path: Path) -> None:
    root = fixture_dataset(tmp_path)
    dataset = GridUtteranceDataset(root, "train")
    sample = dataset[0]
    assert sample.frames.shape == (3, 12, 12, 3)
    assert sample.frames.dtype == np.uint8
    assert sample.frame_indices == (0, 1, 2)
    assert sample.audio.shape == (16000,)
    assert sample.sample_rate == 16000
    assert sample.audio_duration_sec == pytest.approx(1.0)
    dataset.close()


def test_preserves_missing_source_frame_indices(tmp_path: Path) -> None:
    sample = GridUtteranceDataset(fixture_dataset(tmp_path, gap=True), "train")[0]
    assert sample.frame_indices == (0, 1, 3)
    assert sample.video_duration_sec == pytest.approx(4 / 25)


def test_cache_copy(tmp_path: Path) -> None:
    root = fixture_dataset(tmp_path / "data")
    dataset = GridUtteranceDataset(root, "train", cache_dir=tmp_path / "cache")
    assert dataset[0].speaker_id == "s1"
    assert (tmp_path / "cache" / "s1.tar").exists()
    dataset.close()


def test_invalid_pcm_audio_rejected(tmp_path: Path) -> None:
    dataset = GridUtteranceDataset(fixture_dataset(tmp_path, corrupt_audio=True), "train")
    with pytest.raises((DatasetReadError, EOFError, wave.Error)):
        dataset[0]


def test_manifest_rejects_unsafe_prefix(tmp_path: Path) -> None:
    root = fixture_dataset(tmp_path)
    path = root / "gold" / "train_manifest.csv"
    content = path.read_text().replace("s1/bbaf2n/", "../escape/")
    path.write_text(content)
    with pytest.raises(ManifestError):
        GoldManifestIndex(root, "train")


def test_manifest_rejects_wrong_shard(tmp_path: Path) -> None:
    root = fixture_dataset(tmp_path)
    path = root / "gold" / "train_manifest.csv"
    path.write_text(path.read_text().replace("s1.tar", "s2.tar"))
    with pytest.raises(ManifestError):
        GoldManifestIndex(root, "train")


def test_missing_shard_fails_with_context(tmp_path: Path) -> None:
    root = fixture_dataset(tmp_path)
    (root / "silver" / "shards" / "s1.tar").unlink()
    with pytest.raises(DatasetReadError, match="Missing Silver shard"):
        GridUtteranceDataset(root, "train")[0]
