"""Tests for decoded GRID samples, using fake TAR bytes and no real media."""
from __future__ import annotations

import io
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.lipsync.datasets import grid_dataset
from src.lipsync.datasets.error_exception import DatasetReadError
from src.lipsync.datasets.manifest import ManifestRecord


def make_jpeg(value: int, width: int = 12, height: int | None = None) -> bytes:
    if height is None:
        height = width
    image = np.full((height, width, 3), value, dtype=np.uint8)
    success, data = cv2.imencode(".jpg", image)
    assert success
    return data.tobytes()


def make_wav(rate: int = 16000, channels: int = 1, count: int = 1600) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(b"\x00" * count * channels * 2)
    return output.getvalue()


@pytest.fixture
def sample_record() -> ManifestRecord:
    return ManifestRecord(
        split="train", speaker_id="s1", utterance_id="bbaf2n",
        shard_member_prefix="s1/bbaf2n/", frame_count=2,
        first_frame_index=0, last_frame_index=2,
        face_detection_ratio=1.0, audio_duration_sec=0.1,
        audio_sample_rate=16000, source_shard_path="old/s1.tar",
    )


def make_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record: ManifestRecord,
    raw_frames: dict[int, bytes] | None = None,
    wav_bytes: bytes | None = None,
    frame_size: tuple[int, int] = (96, 96),
) -> grid_dataset.GridUtteranceDataset:
    if raw_frames is None:
        raw_frames = {0: make_jpeg(10), 2: make_jpeg(80)}
    if wav_bytes is None:
        wav_bytes = make_wav()

    class FakeManifest:
        def __init__(self, root: Path, split: str) -> None:
            self.record = record

        def __getitem__(self, index: int) -> ManifestRecord:
            return self.record

        def __len__(self) -> int:
            return 1

    class FakeReader:
        def __init__(self, root: Path, **kwargs) -> None:
            pass

        def read_utterance(self, given_record: ManifestRecord):
            return raw_frames, wav_bytes

        def close(self) -> None:
            pass

    monkeypatch.setattr(grid_dataset, "GoldManifestIndex", FakeManifest)
    monkeypatch.setattr(grid_dataset, "GridShardReader", FakeReader)
    return grid_dataset.GridUtteranceDataset(
        tmp_path,
        "train",
        frame_size=frame_size,
        min_frame_count=1,  # Synthetic 2-frame fixtures must ignore user training filters.
    )


def test_decoded_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_record: ManifestRecord):
    dataset = make_dataset(monkeypatch, tmp_path, sample_record)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample.frames.shape == (2, 96, 96, 3)
    assert sample.frames.dtype == np.uint8
    assert sample.frame_indices == (0, 2)
    assert sample.audio.shape == (1600,)
    assert sample.audio.dtype == np.float32
    assert sample.sample_rate == 16000
    assert sample.audio_duration_sec == pytest.approx(0.1)
    assert sample.video_duration_sec == pytest.approx(3 / 25)
    dataset.close()


def test_corrupt_jpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_record: ManifestRecord):
    dataset = make_dataset(monkeypatch, tmp_path, sample_record, raw_frames={0: b"bad", 2: make_jpeg(80)})
    with pytest.raises(DatasetReadError, match="Cannot decode frame 0"):
        dataset[0]
    dataset.close()


def test_variable_image_shapes_are_resized_and_padded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_record: ManifestRecord,
):
    frames = {
        0: make_jpeg(10, width=12, height=24),
        2: make_jpeg(80, width=24, height=12),
    }
    dataset = make_dataset(
        monkeypatch,
        tmp_path,
        sample_record,
        raw_frames=frames,
        frame_size=(20, 16),
    )

    sample = dataset[0]

    assert sample.frames.shape == (2, 16, 20, 3)
    assert np.all(sample.frames[0, :, :6] == 0)
    assert np.all(sample.frames[1, :3, :] == 0)
    dataset.close()


def test_bad_frame_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_record: ManifestRecord):
    sample_record = ManifestRecord(
        **{**vars(sample_record), "frame_count": 3}
    )
    dataset = make_dataset(monkeypatch, tmp_path, sample_record)
    with pytest.raises(DatasetReadError, match="expected 3 frames"):
        dataset[0]
    dataset.close()


def test_corrupt_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_record: ManifestRecord):
    dataset = make_dataset(monkeypatch, tmp_path, sample_record, wav_bytes=b"invalid WAV")
    with pytest.raises(DatasetReadError, match="Cannot decode WAV"):
        dataset[0]
    dataset.close()


def test_stereo_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_record: ManifestRecord):
    dataset = make_dataset(monkeypatch, tmp_path, sample_record, wav_bytes=make_wav(channels=2))
    with pytest.raises(DatasetReadError, match="Expected mono PCM16"):
        dataset[0]
    dataset.close()


def test_sample_rate_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_record: ManifestRecord):
    dataset = make_dataset(monkeypatch, tmp_path, sample_record, wav_bytes=make_wav(rate=8000))
    with pytest.raises(DatasetReadError, match="Audio rate differs"):
        dataset[0]
    dataset.close()


def test_empty_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_record: ManifestRecord):
    dataset = make_dataset(monkeypatch, tmp_path, sample_record, wav_bytes=make_wav(count=0))
    with pytest.raises(DatasetReadError, match="Empty audio"):
        dataset[0]
    dataset.close()


def test_invalid_fps(tmp_path: Path):
    with pytest.raises(ValueError, match="fps must be positive"):
        grid_dataset.GridUtteranceDataset(tmp_path, "train", fps=0)


@pytest.mark.parametrize("frame_size", [(0, 96), (96, 0), (-1, 96)])
def test_invalid_frame_size(tmp_path: Path, frame_size: tuple[int, int]):
    with pytest.raises(ValueError, match="frame_size width and height must be positive"):
        grid_dataset.GridUtteranceDataset(
            tmp_path,
            "train",
            frame_size=frame_size,
        )
