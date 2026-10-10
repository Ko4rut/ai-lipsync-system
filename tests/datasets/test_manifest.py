
import csv
from pathlib import Path

import pytest

from src.lipsync.datasets.manifest import (
    FIELDS,
    GoldManifestIndex,
    ManifestError,
    ManifestRecord,
    _record,
)


# ============================================================
# TEST FIXTURES
# ============================================================

@pytest.fixture
def valid_row() -> dict[str, str]:
    """Tạo một dòng Gold Manifest hợp lệ."""

    return {
        "split": "train",
        "speaker_id": "s1",
        "utterance_id": "bbaf2n",
        "shard_path": r"G:\dataset\silver\shards\s1.tar",
        "shard_member_prefix": "s1/bbaf2n/",
        "frame_count": "75",
        "first_frame_index": "0",
        "last_frame_index": "74",
        "face_detection_ratio": "1.0",
        "audio_duration_sec": "1.19",
        "audio_sample_rate": "16000",
        "valid": "True",
    }


def create_manifest_csv(
    root: Path,
    split: str,
    rows: list[dict[str, str]],
    fieldnames: list[str] | None = None,
) -> Path:
    """Tạo file Gold Manifest giả lập để phục vụ unit test."""

    gold_directory = root / "gold"
    gold_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = gold_directory / f"{split}_manifest.csv"

    if fieldnames is None:
        fieldnames = sorted(FIELDS)

    with manifest_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


# ============================================================
# 1. PARSER TESTS
# ============================================================

def test_parse_valid_record(valid_row):
    """Parser phải chuyển đổi một dòng CSV hợp lệ."""

    record = _record(
        row=valid_row,
        line=2,
        expected_split="train",
    )

    assert isinstance(record, ManifestRecord)

    assert record.split == "train"
    assert record.speaker_id == "s1"
    assert record.utterance_id == "bbaf2n"

    assert record.frame_count == 75
    assert record.first_frame_index == 0
    assert record.last_frame_index == 74

    assert record.face_detection_ratio == 1.0
    assert record.audio_duration_sec == 1.19
    assert record.audio_sample_rate == 16000

    assert record.shard_member_prefix == "s1/bbaf2n/"


def test_parse_trims_whitespace(valid_row):
    """Parser phải loại bỏ khoảng trắng thừa."""

    valid_row["speaker_id"] = "  s1  "
    valid_row["frame_count"] = " 75 "

    record = _record(valid_row, 2, "train")

    assert record.speaker_id == "s1"
    assert record.frame_count == 75


def test_parse_missing_field(valid_row):
    """Thiếu field bắt buộc phải phát sinh ManifestError."""

    del valid_row["frame_count"]

    with pytest.raises(ManifestError):
        _record(valid_row, 2, "train")


# ============================================================
# 2. VALIDATION TESTS
# ============================================================

@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("split", "test"),
        ("speaker_id", "invalid"),
        ("speaker_id", "s21"),
        ("speaker_id", "s01"),
        ("utterance_id", "abc/123"),
        ("shard_member_prefix", "s1/wrong/"),
        ("shard_path", "s2.tar"),
        ("valid", "False"),
        ("frame_count", "0"),
        ("frame_count", "abc"),
        ("frame_count", "76"),
        ("first_frame_index", "-1"),
        ("last_frame_index", "73"),
        ("face_detection_ratio", "1.5"),
        ("face_detection_ratio", "-0.5"),
        ("audio_duration_sec", "0"),
        ("audio_sample_rate", "0"),
    ],
)
def test_invalid_record(
    valid_row,
    field,
    invalid_value,
):
    """Các trường hợp dữ liệu không hợp lệ phải bị từ chối."""

    valid_row[field] = invalid_value

    with pytest.raises(ManifestError):
        _record(
            row=valid_row,
            line=2,
            expected_split="train",
        )


# ============================================================
# 3. GOLD MANIFEST INDEX TESTS
# ============================================================

def test_load_manifest(tmp_path, valid_row):
    """GoldManifestIndex phải đọc được CSV hợp lệ."""

    second_row = valid_row.copy()

    second_row["utterance_id"] = "bbaf3s"
    second_row["shard_member_prefix"] = "s1/bbaf3s/"

    create_manifest_csv(
        root=tmp_path,
        split="train",
        rows=[valid_row, second_row],
    )

    manifest = GoldManifestIndex(
        root=tmp_path,
        split="train",
    )

    # Test __len__
    assert len(manifest) == 2

    # Test __getitem__
    first_record = manifest[0]
    second_record = manifest[1]

    assert first_record.utterance_id == "bbaf2n"
    assert second_record.utterance_id == "bbaf3s"

    # Test __iter__
    records = list(manifest)

    assert len(records) == 2
    assert all(
        isinstance(record, ManifestRecord)
        for record in records
    )


def test_invalid_split(tmp_path):
    """Không được chấp nhận split ngoài train/val/test."""

    with pytest.raises(ValueError):
        GoldManifestIndex(
            root=tmp_path,
            split="invalid",
        )


def test_missing_manifest_file(tmp_path):
    """File manifest không tồn tại phải phát sinh lỗi."""

    with pytest.raises(ManifestError):
        GoldManifestIndex(
            root=tmp_path,
            split="train",
        )


def test_missing_required_columns(tmp_path, valid_row):
    """CSV thiếu cột bắt buộc phải bị từ chối."""

    fieldnames = [
        field
        for field in sorted(FIELDS)
        if field != "audio_sample_rate"
    ]

    create_manifest_csv(
        root=tmp_path,
        split="train",
        rows=[valid_row],
        fieldnames=fieldnames,
    )

    with pytest.raises(
    ManifestError,
    match="audio_sample_rate",
    ):
        GoldManifestIndex(
            root=tmp_path,
            split="train",
        )


def test_duplicate_utterance(tmp_path, valid_row):
    """Không chấp nhận trùng speaker_id và utterance_id."""

    duplicate_row = valid_row.copy()

    create_manifest_csv(
        root=tmp_path,
        split="train",
        rows=[valid_row, duplicate_row],
    )

    with pytest.raises(
        ManifestError,
        match="Duplicate utterance",
    ):
        GoldManifestIndex(
            root=tmp_path,
            split="train",
        )


def test_empty_manifest_with_header(tmp_path):
    """Manifest chỉ có header phải trả về collection rỗng."""

    create_manifest_csv(
        root=tmp_path,
        split="train",
        rows=[],
    )

    manifest = GoldManifestIndex(
        root=tmp_path,
        split="train",
    )

    assert len(manifest) == 0
