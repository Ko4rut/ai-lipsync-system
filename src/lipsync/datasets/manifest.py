"""Validated Gold manifest index for the preprocessed GRID corpus."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterator
from .error_exception import ManifestError

SPLITS = frozenset({"train", "val", "test"})
FIELDS = frozenset({
    "split", "speaker_id", "utterance_id", "shard_path",
    "shard_member_prefix", "frame_count", "first_frame_index",
    "last_frame_index", "face_detection_ratio", "audio_duration_sec",
    "audio_sample_rate", "valid",
})
_SPEAKER = re.compile(r"s[1-9][0-9]*\Z")
_UTTERANCE = re.compile(r"[A-Za-z0-9_-]+\Z")

@dataclass(frozen=True)
class ManifestRecord:
    split: str
    speaker_id: str
    utterance_id: str
    shard_member_prefix: str
    frame_count: int
    first_frame_index: int
    last_frame_index: int
    face_detection_ratio: float
    audio_duration_sec: float
    audio_sample_rate: int
    source_shard_path: str


def _record(row: dict[str, str], line: int, expected_split: str) -> ManifestRecord:
    try:
        split = row["split"].strip()
        speaker = row["speaker_id"].strip()
        utterance = row["utterance_id"].strip()
        prefix = row["shard_member_prefix"].strip()
        shard = row["shard_path"].strip()
        valid = row["valid"].strip().lower()
        count = int(row["frame_count"])
        first = int(row["first_frame_index"])
        last = int(row["last_frame_index"])
        ratio = float(row["face_detection_ratio"])
        duration = float(row["audio_duration_sec"])
        rate = int(row["audio_sample_rate"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ManifestError(f"Manifest row {line}: invalid or missing value: {exc}") from exc

    if split != expected_split:
        raise ManifestError(f"Manifest row {line}: expected split {expected_split!r}, got {split!r}")
    if not _SPEAKER.fullmatch(speaker) or speaker == "s21":
        raise ManifestError(f"Manifest row {line}: invalid GRID speaker: {speaker!r}")
    if not _UTTERANCE.fullmatch(utterance):
        raise ManifestError(f"Manifest row {line}: invalid utterance ID: {utterance!r}")
    if prefix != f"{speaker}/{utterance}/":
        raise ManifestError(f"Manifest row {line}: unsafe/incorrect TAR member prefix: {prefix!r}")
    # This field can be an obsolete Windows or Colab Drive absolute path.
    # Validate its filename, but NEVER use it to choose the file to read.
    filename = PureWindowsPath(shard.replace("/", "\\")).name
    if filename != f"{speaker}.tar":
        raise ManifestError(f"Manifest row {line}: wrong shard filename: {shard!r}")
    if valid not in {"true", "1"}:
        raise ManifestError(f"Manifest row {line}: sample is not marked valid")
    if count < 1 or first < 0 or last < first or last - first + 1 < count:
        raise ManifestError(f"Manifest row {line}: inconsistent frame indices/count")
    if not (0.0 <= ratio <= 1.0) or duration <= 0 or rate <= 0:
        raise ManifestError(f"Manifest row {line}: invalid media metadata")

    return ManifestRecord(
        split, speaker, utterance, prefix, count, first, last,
        ratio, duration, rate, shard,
    )


class GoldManifestIndex:
    """
    Đọc và quản lý danh sách utterance từ Gold Manifest.

    Trách nhiệm:
    1. Xác định đường dẫn đến manifest CSV.
    2. Đọc dữ liệu từ CSV.
    3. Kiểm tra cấu trúc và tính hợp lệ.
    4. Chuyển từng dòng thành ManifestRecord.
    5. Cho phép truy cập các record theo index.
    """

    def __init__(self, root: str | Path, split: str) -> None:

        # STEP 1: Kiểm tra split có hợp lệ hay không
        if split not in SPLITS:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Expected one of: {sorted(SPLITS)}"
            )

        # STEP 2: Xác định thư mục gốc của dataset
        self.root = Path(root)
        self.root = self.root.expanduser()
        self.root = self.root.resolve()

        # Lưu tên split
        self.split = split

        # STEP 3: Xây dựng đường dẫn đến manifest
        manifest_filename = f"{split}_manifest.csv"
        gold_directory = self.root / "gold"

        self.path = gold_directory / manifest_filename

        # STEP 4: Khởi tạo danh sách chứa record
        records: list[ManifestRecord] = []

        # STEP 5: Đọc file CSV
        try:
            with self.path.open(
                mode="r",
                newline="",
                encoding="utf-8-sig"
            ) as file:

                reader = csv.DictReader(file)

                # STEP 6: Lấy danh sách tên cột
                column_names = reader.fieldnames

                if column_names is None:
                    raise ManifestError(
                        "Manifest CSV is empty or has no header"
                    )

                # STEP 7: Kiểm tra các cột bắt buộc
                available_fields = set(column_names)

                missing_fields = FIELDS.difference(
                    available_fields
                )

                if len(missing_fields) > 0:
                    raise ManifestError(
                        f"Missing required columns: "
                        f"{sorted(missing_fields)}"
                    )

                # STEP 8: Đọc từng dòng dữ liệu
                line_number = 2

                for row in reader:

                    # Chuyển dòng CSV thành ManifestRecord
                    record = _record(
                        row=row,
                        line=line_number,
                        expected_split=split
                    )

                    # Thêm record vào danh sách
                    records.append(record)

                    line_number += 1

        except OSError as error:
            raise ManifestError(
                f"Cannot read manifest: {self.path}. "
                f"Reason: {error}"
            ) from error

        # STEP 9: Kiểm tra utterance bị trùng lặp
        seen: set[tuple[str, str]] = set()

        for record in records:

            speaker_id = record.speaker_id
            utterance_id = record.utterance_id

            record_key = (speaker_id, utterance_id)

            if record_key in seen:
                raise ManifestError(
                    f"Duplicate utterance detected: "
                    f"{record_key}"
                )

            seen.add(record_key)

        # STEP 10: Lưu dữ liệu vào thuộc tính của object
        self._records = tuple(records)

    def __len__(self) -> int:
        """Trả về tổng số utterance trong manifest."""

        total_records = len(self._records)

        return total_records

    def __getitem__(self, index: int) -> ManifestRecord:
        """Lấy ManifestRecord tại vị trí index."""

        record = self._records[index]

        return record

    def __iter__(self) -> Iterator[ManifestRecord]:
        """Cho phép duyệt các record bằng vòng lặp for."""

        iterator = iter(self._records)

        return iterator