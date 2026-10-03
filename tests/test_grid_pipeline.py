"""Tests for the GRID preprocessing pipeline.

These tests run WITHOUT downloading the GRID dataset or requiring
Google Colab / Google Drive.  They validate:

  1.  Speaker list contains exactly 33 speakers and excludes s21.
  2.  Fixed seed=42 produces the expected deterministic 27/3/3 split.
  3.  Speaker split sets are mutually disjoint.
  4.  Audio/video pairing correctly classifies utterances.
  5.  StateManager skips a valid DONE speaker.
  6.  FAILED/interrupted speaker can be retried.
  7.  TAR packaging produces the required internal member structure.
  8.  Gold manifests contain only valid samples.
  9.  Download fallback selection uses mocks (no real HTTP calls).
 10.  Pipeline modules can be imported outside Google Colab.
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Make src/ importable without installation
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Test 10: import outside Colab (must not crash even if google.colab absent)
# ---------------------------------------------------------------------------

def test_modules_importable_outside_colab() -> None:
    """All pipeline modules must be importable without google.colab installed."""
    # google.colab should NOT be importable in a normal dev environment
    try:
        import google.colab  # type: ignore[import-not-found]
        pytest.skip("google.colab is available; this test targets non-Colab environments.")
    except ImportError:
        pass

    from data.config import PipelineConfig, USABLE_SPEAKERS, EXCLUDED_SPEAKERS
    from data.exceptions import GRIDPipelineError
    from data.utils import sha256_file, atomic_json_write, max_consecutive_run
    from data.state import StateManager
    from data.split import compute_speaker_split
    from data import __all__ as pkg_all

    assert "PipelineConfig" in pkg_all
    assert "compute_speaker_split" in pkg_all


def test_audio_conversion_temp_file_keeps_wav_suffix(tmp_path: Path) -> None:
    """ffmpeg must receive a .wav output path so it can select the WAV muxer."""
    from data.config import PipelineConfig
    from data.preprocess import process_audio_utterance

    source = tmp_path / "input.wav"
    output = tmp_path / "audio.wav"
    source.write_bytes(b"fake input")
    commands: List[List[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"RIFF" + b"\x00" * 40)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout="16000\n1\n1.0\n",
            stderr="",
        )

    config = PipelineConfig(workspace_root=tmp_path)
    with patch("data.preprocess.subprocess.run", side_effect=fake_run):
        result = process_audio_utterance(
            "test-utterance",
            audio_path=source,
            output_path=output,
            config=config,
        )

    ffmpeg_command = next(command for command in commands if command[0] == "ffmpeg")
    assert ffmpeg_command[-1].endswith(".wav")
    assert ffmpeg_command[-1].endswith("audio.tmp.wav")
    assert result.success
    assert output.exists()


def test_sfd_adapter_converts_numpy_nhwc_to_torch_nchw() -> None:
    """The face-alignment SFD API requires a PyTorch NCHW batch."""
    import numpy as np
    import torch

    from data.preprocess import _SFDDetectorAdapter

    class FakeSFDDetector:
        received = None

        def detect_from_batch(self, images):
            self.received = images
            return [np.array([[1.0, 2.0, 30.0, 40.0, 0.99]])]

    fake = FakeSFDDetector()
    adapter = _SFDDetectorAdapter(fake)
    result = adapter.get_detections_for_batch(
        np.zeros((1, 48, 64, 3), dtype=np.uint8)
    )

    assert isinstance(fake.received, torch.Tensor)
    assert tuple(fake.received.shape) == (1, 3, 48, 64)
    assert result[0].tolist() == [1.0, 2.0, 30.0, 40.0]


# ---------------------------------------------------------------------------
# Test 1: speaker list
# ---------------------------------------------------------------------------

def test_usable_speaker_count() -> None:
    """Exactly 33 usable speakers; s21 excluded."""
    from data.config import USABLE_SPEAKERS, EXCLUDED_SPEAKERS

    assert len(USABLE_SPEAKERS) == 33
    assert "s21" not in USABLE_SPEAKERS
    assert "s21" in EXCLUDED_SPEAKERS


def test_usable_speakers_range() -> None:
    """Usable speakers are drawn from s1–s34 only."""
    from data.config import USABLE_SPEAKERS

    for spk in USABLE_SPEAKERS:
        assert spk.startswith("s")
        num = int(spk[1:])
        assert 1 <= num <= 34


# ---------------------------------------------------------------------------
# Test 2: deterministic split with seed=42
# ---------------------------------------------------------------------------

EXPECTED_TRAIN = [
    "s29", "s6", "s12", "s17", "s28", "s13", "s22", "s16", "s11",
    "s15", "s20", "s24", "s10", "s31", "s26", "s7", "s27", "s1",
    "s33", "s14", "s19", "s3", "s18", "s23", "s4", "s32", "s5",
]
EXPECTED_VAL = ["s30", "s34", "s9"]
EXPECTED_TEST = ["s25", "s2", "s8"]


def test_deterministic_split_seed42() -> None:
    """Seed=42 must produce the exact expected 27/3/3 split."""
    from data.split import compute_speaker_split

    train, val, test = compute_speaker_split(seed=42)

    assert train == EXPECTED_TRAIN, f"Train mismatch: {train}"
    assert val == EXPECTED_VAL, f"Val mismatch: {val}"
    assert test == EXPECTED_TEST, f"Test mismatch: {test}"


def test_split_counts() -> None:
    from data.split import compute_speaker_split

    train, val, test = compute_speaker_split(seed=42)
    assert len(train) == 27
    assert len(val) == 3
    assert len(test) == 3


# ---------------------------------------------------------------------------
# Test 3: disjoint splits
# ---------------------------------------------------------------------------

def test_split_disjoint() -> None:
    """Train, val, test sets must be mutually disjoint and cover all 33 speakers."""
    from data.split import compute_speaker_split
    from data.config import USABLE_SPEAKERS

    train, val, test = compute_speaker_split(seed=42)
    train_s, val_s, test_s = set(train), set(val), set(test)

    assert train_s & val_s == set(), f"Train ∩ Val non-empty: {train_s & val_s}"
    assert train_s & test_s == set(), f"Train ∩ Test non-empty: {train_s & test_s}"
    assert val_s & test_s == set(), f"Val ∩ Test non-empty: {val_s & test_s}"
    assert train_s | val_s | test_s == set(USABLE_SPEAKERS), "Union != all usable speakers"
    assert "s21" not in (train_s | val_s | test_s)


# ---------------------------------------------------------------------------
# Test 4: utterance pairing
# ---------------------------------------------------------------------------

def test_utterance_pairing_identifies_correctly() -> None:
    """Pairing logic classifies PAIRED / MISSING_AUDIO / MISSING_VIDEO."""
    from data.preprocess import UtterancePairingStatus

    video_map: Dict[str, Path] = {
        "bbaf2n": Path("/fake/video/bbaf2n.mpg"),
        "bbwi2a": Path("/fake/video/bbwi2a.mpg"),  # no audio
    }
    audio_map: Dict[str, Path] = {
        "bbaf2n": Path("/fake/audio/bbaf2n.wav"),
        "lgwl7p": Path("/fake/audio/lgwl7p.wav"),  # no video
    }

    all_uids = sorted(set(video_map) | set(audio_map))
    from data.preprocess import UtteranceCandidate

    results = []
    for uid in all_uids:
        has_v = uid in video_map
        has_a = uid in audio_map
        if has_v and has_a:
            status = UtterancePairingStatus.PAIRED
        elif has_v:
            status = UtterancePairingStatus.MISSING_AUDIO
        else:
            status = UtterancePairingStatus.MISSING_VIDEO
        results.append(UtteranceCandidate(
            utterance_id=uid,
            video_path=video_map.get(uid),
            audio_path=audio_map.get(uid),
            pairing_status=status,
        ))

    statuses = {r.utterance_id: r.pairing_status for r in results}
    assert statuses["bbaf2n"] == UtterancePairingStatus.PAIRED
    assert statuses["bbwi2a"] == UtterancePairingStatus.MISSING_AUDIO
    assert statuses["lgwl7p"] == UtterancePairingStatus.MISSING_VIDEO


# ---------------------------------------------------------------------------
# Tests 5 & 6: StateManager resume logic
# ---------------------------------------------------------------------------

def _make_state_manager_with_fake_drive(tmp_path: Path, speaker_id: str = "s1"):
    from data.state import StateManager
    from data.config import USABLE_SPEAKERS

    state_file = tmp_path / "state" / "processing_state.json"
    sm = StateManager(state_file=state_file, usable_speakers=USABLE_SPEAKERS)
    return sm, state_file


def test_state_manager_skips_done_speaker(tmp_path: Path) -> None:
    """A speaker marked DONE with artefacts present should be skipped."""
    from data.state import StateManager
    from data.config import USABLE_SPEAKERS, SpeakerStatus

    state_file = tmp_path / "state" / "processing_state.json"
    sm = StateManager(state_file=state_file, usable_speakers=USABLE_SPEAKERS)

    # Simulate completed speaker s1
    shard_path = tmp_path / "shards" / "s1.tar"
    meta_path = tmp_path / "metadata" / "s1.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_bytes(b"fake tar content")
    meta_path.write_text("{}", encoding="utf-8")

    # Compute real sha256 for the fake file
    import hashlib
    sha = hashlib.sha256(b"fake tar content").hexdigest()

    sm.record_done(
        "s1",
        shard_path=str(shard_path),
        metadata_path=str(meta_path),
        shard_size_bytes=shard_path.stat().st_size,
        shard_sha256=sha,
        video_source_used="primary",
        audio_source_used="primary",
    )

    assert sm.is_done("s1")
    assert sm.get_status("s1") == SpeakerStatus.DONE

    # verify_done_artefacts should return True (file exists, size & sha match)
    assert sm.verify_done_artefacts(
        "s1",
        silver_shards_dir=shard_path.parent,
        silver_metadata_dir=meta_path.parent,
    )


def test_state_manager_retries_failed_speaker(tmp_path: Path) -> None:
    """A speaker marked FAILED should be scheduled for retry."""
    from data.state import StateManager
    from data.config import USABLE_SPEAKERS, SpeakerStatus

    state_file = tmp_path / "state" / "processing_state.json"
    sm = StateManager(state_file=state_file, usable_speakers=USABLE_SPEAKERS)

    sm.transition("s2", SpeakerStatus.DOWNLOADING)
    sm.record_failure("s2", "Connection reset")

    assert sm.get_status("s2") == SpeakerStatus.FAILED
    assert sm.should_retry("s2")
    assert not sm.is_done("s2")


def test_state_survives_reload(tmp_path: Path) -> None:
    """Persisted state must round-trip through JSON serialisation."""
    from data.state import StateManager
    from data.config import USABLE_SPEAKERS, SpeakerStatus

    state_file = tmp_path / "state" / "processing_state.json"
    sm = StateManager(state_file=state_file, usable_speakers=USABLE_SPEAKERS)
    sm.transition("s3", SpeakerStatus.PREPROCESSING)
    sm.record_failure("s3", "Some error")

    # Reload
    sm2 = StateManager(state_file=state_file, usable_speakers=USABLE_SPEAKERS)
    assert sm2.get_status("s3") == SpeakerStatus.FAILED
    assert sm2.get_speaker_info("s3")["last_error"] == "Some error"


# ---------------------------------------------------------------------------
# Test 7: TAR packaging member structure
# ---------------------------------------------------------------------------

def test_tar_packaging_structure(tmp_path: Path) -> None:
    """TAR must contain sN/utterance_id/filename entries for valid samples only."""
    from data.shard import package_speaker_shard, verify_shard
    from data.config import PipelineConfig

    config = PipelineConfig(workspace_root=tmp_path)

    speaker_id = "s1"
    silver_local = tmp_path / "silver" / speaker_id
    package_dir = tmp_path / "package"

    # Build fake Silver structure
    valid_uids = ["bbaf2n", "abcd1e"]
    invalid_uid = "badf00d"

    for uid in valid_uids:
        utt_dir = silver_local / uid
        utt_dir.mkdir(parents=True, exist_ok=True)
        (utt_dir / "0.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        (utt_dir / "1.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
        (utt_dir / "audio.wav").write_bytes(b"RIFF" + b"\x00" * 36)

    # Invalid utterance directory exists locally but must NOT appear in TAR
    invalid_dir = silver_local / invalid_uid
    invalid_dir.mkdir(parents=True, exist_ok=True)
    (invalid_dir / "0.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)

    tar_path = package_speaker_shard(
        speaker_id,
        silver_local_dir=silver_local,
        package_dir=package_dir,
        valid_utterance_ids=valid_uids,
        config=config,
    )

    assert tar_path.exists()
    assert tar_path.stat().st_size > 0

    with tarfile.open(tar_path) as tf:
        member_names = {m.name for m in tf.getmembers()}

    # Valid utterances present
    for uid in valid_uids:
        assert f"s1/{uid}/audio.wav" in member_names, f"Missing: s1/{uid}/audio.wav"
        assert f"s1/{uid}/0.jpg" in member_names

    # Invalid utterance absent
    for name in member_names:
        assert invalid_uid not in name, f"Invalid uid '{invalid_uid}' found in TAR: {name}"

    # verify_shard should pass
    verify_shard(tar_path, speaker_id, valid_uids)


# ---------------------------------------------------------------------------
# Test 8: Gold manifests contain only valid samples
# ---------------------------------------------------------------------------

def test_gold_manifests_only_valid_samples(tmp_path: Path) -> None:
    """Gold manifests must contain only samples with valid=True."""
    import csv
    from data.split import build_gold_manifests, compute_speaker_split
    from data.config import PipelineConfig, USABLE_SPEAKERS, SCHEMA_VERSION
    from data.utils import atomic_json_write

    config = PipelineConfig(
        drive_root=tmp_path / "drive",
        workspace_root=tmp_path / "workspace",
    )

    # Create fake Silver metadata for ALL usable speakers
    train, val, test = compute_speaker_split(seed=42)
    all_speakers = train + val + test

    for speaker_id in all_speakers:
        meta = {
            "schema_version": SCHEMA_VERSION,
            "dataset": "GRID",
            "speaker_id": speaker_id,
            "status": "DONE",
            "source": {"video_source_used": "primary", "audio_source_used": "primary"},
            "processing": {"expected_fps": 25, "target_audio_sample_rate": 16000},
            "summary": {"total_candidates": 3, "valid_samples": 2, "invalid_samples": 1},
            "invalid_reason_counts": {"NO_FRAMES": 1},
            "samples": [
                {
                    "utterance_id": "utt001",
                    "valid": True,
                    "invalid_reason": None,
                    "source_fps": 25.0,
                    "source_frame_count": 75,
                    "cropped_frame_count": 75,
                    "face_detection_ratio": 1.0,
                    "max_consecutive_face_frames": 75,
                    "detected_frame_indices": list(range(75)),
                    "audio_duration_sec": 3.0,
                    "audio_sample_rate": 16000,
                    "audio_channels": 1,
                    "shard_member_prefix": f"{speaker_id}/utt001/",
                },
                {
                    "utterance_id": "utt002",
                    "valid": True,
                    "invalid_reason": None,
                    "source_fps": 25.0,
                    "source_frame_count": 50,
                    "cropped_frame_count": 50,
                    "face_detection_ratio": 1.0,
                    "max_consecutive_face_frames": 50,
                    "detected_frame_indices": list(range(50)),
                    "audio_duration_sec": 2.0,
                    "audio_sample_rate": 16000,
                    "audio_channels": 1,
                    "shard_member_prefix": f"{speaker_id}/utt002/",
                },
                {
                    "utterance_id": "utt003",
                    "valid": False,
                    "invalid_reason": "NO_FRAMES",
                    "source_fps": 0.0,
                    "source_frame_count": 0,
                    "cropped_frame_count": 0,
                    "face_detection_ratio": 0.0,
                    "max_consecutive_face_frames": 0,
                    "detected_frame_indices": [],
                    "audio_duration_sec": None,
                    "audio_sample_rate": None,
                    "audio_channels": None,
                    "shard_member_prefix": f"{speaker_id}/utt003/",
                },
            ],
            "shard": {
                "path": str(config.silver_shards_dir / f"{speaker_id}.tar"),
                "size_bytes": 1000,
                "sha256": "abc123",
            },
        }
        meta_path = config.silver_metadata_dir / f"{speaker_id}.json"
        atomic_json_write(meta_path, meta)

    ok = build_gold_manifests(config)
    assert ok, "build_gold_manifests returned False"

    for split_name in ("train", "val", "test"):
        csv_path = config.gold_dir / f"{split_name}_manifest.csv"
        assert csv_path.exists(), f"Missing: {csv_path}"
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        # Every row must have valid=True (or the column value "True")
        for row in rows:
            assert str(row["valid"]).lower() in ("true", "1"), (
                f"Invalid sample found in {split_name}: {row}"
            )
        # No 'utt003' (the invalid utterance) should appear
        utt_ids = {row["utterance_id"] for row in rows}
        assert "utt003" not in utt_ids, f"Invalid utterance utt003 found in {split_name} manifest"


# ---------------------------------------------------------------------------
# Test 9: download fallback via mocks
# ---------------------------------------------------------------------------

def test_download_fallback_to_zenodo_on_primary_failure(tmp_path: Path) -> None:
    """When primary download fails, the downloader must fall back to Zenodo."""
    from data.downloader import SpeakerDownloader
    from data.config import PipelineConfig
    from data.exceptions import DownloadError

    config = PipelineConfig(
        workspace_root=tmp_path,
        drive_root=tmp_path / "drive",
    )
    config.download_max_retries = 1  # fail fast in tests

    downloader = SpeakerDownloader(config)

    fake_zip = tmp_path / "s1_fake.zip"
    import zipfile
    with zipfile.ZipFile(fake_zip, "w") as zf:
        zf.writestr("s1/bbaf2n.mpg", b"fake mpg content")

    fake_tar = tmp_path / "s1_fake.tar"
    import tarfile as _tf
    with _tf.open(fake_tar, "w") as tf:
        # Create a member
        import io
        data = b"fake wav"
        info = _tf.TarInfo(name="s1/bbaf2n.wav")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    # Mock: primary video fails, zenodo video succeeds; primary audio succeeds
    with (
        patch.object(downloader, "_download_primary_video", side_effect=DownloadError("primary down")),
        patch.object(downloader, "_download_zenodo_video", return_value=fake_zip),
        patch.object(downloader, "_download_primary_audio", return_value=fake_tar),
    ):
        video_arch, audio_arch, video_src, audio_src = downloader.download("s1")

    assert video_src == "zenodo_fallback"
    assert audio_src == "primary"
    assert video_arch == fake_zip
    assert audio_arch == fake_tar


# ---------------------------------------------------------------------------
# Utility: consecutive run helper
# ---------------------------------------------------------------------------

def test_max_consecutive_run() -> None:
    from data.utils import max_consecutive_run

    assert max_consecutive_run([]) == 0
    assert max_consecutive_run([5]) == 1
    assert max_consecutive_run([0, 1, 2, 3]) == 4
    assert max_consecutive_run([0, 1, 3, 4, 5, 6]) == 4
    assert max_consecutive_run([0, 2, 4, 6]) == 1
    # Unsorted input
    assert max_consecutive_run([4, 1, 2, 3, 0]) == 5


# ---------------------------------------------------------------------------
# Frame index preservation (unit test — no actual face detection)
# ---------------------------------------------------------------------------

def test_frame_index_not_renumbered(tmp_path: Path) -> None:
    """Frame filenames must preserve source frame index, not densely renumber."""
    # Simulate: frames 0,1,2,17,18,19 were detected (13–16 skipped)
    detected = [0, 1, 2, 17, 18, 19]
    out_dir = tmp_path / "frames"
    out_dir.mkdir()

    for idx in detected:
        (out_dir / f"{idx}.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 8)

    saved = sorted(int(p.stem) for p in out_dir.glob("*.jpg"))
    assert saved == sorted(detected), (
        "Frame files must use original source indices, not renumbered."
    )
    # Ensure that gap frames (3–16) are NOT present
    for missing in range(3, 17):
        assert not (out_dir / f"{missing}.jpg").exists()
