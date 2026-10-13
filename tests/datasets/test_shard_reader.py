"""Focused GridShardReader tests using tiny synthetic TARs (no GRID files needed)."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from src.lipsync.datasets.error_exception import DatasetReadError
from src.lipsync.datasets.manifest import ManifestRecord
from src.lipsync.datasets.shard_reader import GridShardReader


def make_record(speaker: str = "s1", utterance: str = "bbaf2n") -> ManifestRecord:
    return ManifestRecord(
        split="train",
        speaker_id=speaker,
        utterance_id=utterance,
        shard_member_prefix=f"{speaker}/{utterance}/",
        frame_count=2,
        first_frame_index=0,
        last_frame_index=2,
        face_detection_ratio=1.0,
        audio_duration_sec=1.0,
        audio_sample_rate=16000,
        source_shard_path=f"old/location/{speaker}.tar",
    )


def make_shard(
    root: Path,
    speaker: str = "s1",
    members: dict[str, bytes] | None = None,
) -> Path:
    """Write small media-like bytes into a TAR archive."""
    if members is None:
        members = {
            f"{speaker}/bbaf2n/0.jpg": b"jpeg-frame-0",
            f"{speaker}/bbaf2n/2.jpg": b"jpeg-frame-2",
            f"{speaker}/bbaf2n/audio.wav": b"wav-audio",
            f"{speaker}/bbaf2n/notes.txt": b"ignored",
        }

    directory = root / "silver" / "shards"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{speaker}.tar"

    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    return path


def test_list_frame_indices_does_not_read_large_media(tmp_path: Path) -> None:
    make_shard(tmp_path)
    reader = GridShardReader(tmp_path, max_member_bytes=1)

    try:
        assert reader.list_frame_indices(make_record()) == [0, 2]
    finally:
        reader.close()


def test_read_frame_fetches_requested_jpeg(tmp_path: Path) -> None:
    make_shard(tmp_path)
    reader = GridShardReader(tmp_path)

    try:
        assert reader.read_frame(make_record(), 2) == b"jpeg-frame-2"
    finally:
        reader.close()


def test_read_audio_fetches_wav(tmp_path: Path) -> None:
    make_shard(tmp_path)
    reader = GridShardReader(tmp_path)

    try:
        assert reader.read_audio(make_record()) == b"wav-audio"
    finally:
        reader.close()


def test_read_utterance_keeps_return_contract(tmp_path: Path) -> None:
    make_shard(tmp_path)
    reader = GridShardReader(tmp_path)

    try:
        frames, audio = reader.read_utterance(make_record())
        assert frames == {0: b"jpeg-frame-0", 2: b"jpeg-frame-2"}
        assert audio == b"wav-audio"
    finally:
        reader.close()


def test_missing_frame_reports_context(tmp_path: Path) -> None:
    make_shard(tmp_path)
    reader = GridShardReader(tmp_path)

    try:
        with pytest.raises(DatasetReadError, match="Frame 1 not found"):
            reader.read_frame(make_record(), 1)
    finally:
        reader.close()


def test_negative_frame_index_is_rejected(tmp_path: Path) -> None:
    make_shard(tmp_path)
    reader = GridShardReader(tmp_path)

    try:
        with pytest.raises(ValueError, match="non-negative"):
            reader.read_frame(make_record(), -1)
    finally:
        reader.close()


def test_missing_audio_still_allows_frame_access(tmp_path: Path) -> None:
    make_shard(tmp_path, members={"s1/bbaf2n/0.jpg": b"frame"})
    reader = GridShardReader(tmp_path)

    try:
        assert reader.list_frame_indices(make_record()) == [0]
        assert reader.read_frame(make_record(), 0) == b"frame"

        with pytest.raises(DatasetReadError, match="Audio missing"):
            reader.read_audio(make_record())

        with pytest.raises(DatasetReadError, match="Audio missing"):
            reader.read_utterance(make_record())
    finally:
        reader.close()


def test_missing_frames_still_allows_audio_access(tmp_path: Path) -> None:
    make_shard(tmp_path, members={"s1/bbaf2n/audio.wav": b"wav"})
    reader = GridShardReader(tmp_path)

    try:
        assert reader.read_audio(make_record()) == b"wav"

        with pytest.raises(DatasetReadError, match="No JPEG frames"):
            reader.list_frame_indices(make_record())
    finally:
        reader.close()


def test_missing_utterance_is_reported(tmp_path: Path) -> None:
    make_shard(tmp_path)
    reader = GridShardReader(tmp_path)

    try:
        with pytest.raises(DatasetReadError, match="Utterance missing"):
            reader.read_audio(make_record(utterance="unknown"))
    finally:
        reader.close()


def test_missing_tar_is_reported(tmp_path: Path) -> None:
    reader = GridShardReader(tmp_path)

    try:
        with pytest.raises(DatasetReadError, match="Missing Silver shard"):
            reader.read_audio(make_record())
    finally:
        reader.close()


def test_member_size_limit(tmp_path: Path) -> None:
    make_shard(tmp_path)
    reader = GridShardReader(tmp_path, max_member_bytes=4)

    try:
        assert reader.list_frame_indices(make_record()) == [0, 2]

        with pytest.raises(DatasetReadError, match="Oversized TAR member"):
            reader.read_frame(make_record(), 0)

        with pytest.raises(DatasetReadError, match="Oversized TAR member"):
            reader.read_audio(make_record())
    finally:
        reader.close()


def test_duplicate_frame_numbers(tmp_path: Path) -> None:
    make_shard(
        tmp_path,
        members={
            "s1/bbaf2n/1.jpg": b"a",
            "s1/bbaf2n/01.jpg": b"b",
            "s1/bbaf2n/audio.wav": b"c",
        },
    )
    reader = GridShardReader(tmp_path)

    try:
        with pytest.raises(DatasetReadError, match="Duplicate frame index 1"):
            reader.list_frame_indices(make_record())
    finally:
        reader.close()


def test_local_cache(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    cache = tmp_path / "cache"
    make_shard(root)
    reader = GridShardReader(root, cache_dir=cache)

    try:
        assert reader.read_audio(make_record()) == b"wav-audio"
        assert (cache / "s1.tar").is_file()
    finally:
        reader.close()


def test_lru_evicts_old_shard_and_close_clears_handles(tmp_path: Path) -> None:
    make_shard(tmp_path, speaker="s1")
    make_shard(tmp_path, speaker="s2")
    reader = GridShardReader(tmp_path, max_open_shards=1)

    try:
        assert reader.read_audio(make_record("s1")) == b"wav-audio"
        assert len(reader._opened) == 1

        assert reader.read_audio(make_record("s2")) == b"wav-audio"
        assert len(reader._opened) == 1
        assert "s2" in reader._opened
    finally:
        reader.close()

    assert len(reader._opened) == 0
