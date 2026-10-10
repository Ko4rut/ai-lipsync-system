"""Validated Gold manifest index for the preprocessed GRID corpus."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterator

SPLITS = frozenset({"train", "val", "test"})
FIELDS = frozenset({
    "split", "speaker_id", "utterance_id", "shard_path",
    "shard_member_prefix", "frame_count", "first_frame_index",
    "last_frame_index", "face_detection_ratio", "audio_duration_sec",
    "audio_sample_rate", "valid",
})
_SPEAKER = re.compile(r"s[1-9][0-9]*\Z")
_UTTERANCE = re.compile(r"[A-Za-z0-9_-]+\Z")


class ManifestError(ValueError):
    """Invalid or inconsistent Gold manifest."""


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
    """Load the Gold manifest once; each item identifies one utterance.

    Original shard_path is kept for provenance but never used as a filesystem
    location: shards are resolved relative to the supplied dataset root.
    """

    def __init__(self, root: str | Path, split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.path = self.root / "gold" / f"{split}_manifest.csv"
        try:
            with self.path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not FIELDS.issubset(reader.fieldnames):
                    missing = sorted(FIELDS - set(reader.fieldnames or []))
                    raise ManifestError(f"{self.path}: missing columns: {missing}")
                records = [_record(row, line, split) for line, row in enumerate(reader, 2)]
        except OSError as exc:
            raise ManifestError(f"Cannot read manifest {self.path}: {exc}") from exc

        seen: set[tuple[str, str]] = set()
        for entry in records:
            key = (entry.speaker_id, entry.utterance_id)
            if key in seen:
                raise ManifestError(f"Duplicate utterance in {self.path}: {key}")
            seen.add(key)
        self._records = tuple(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> ManifestRecord:
        return self._records[index]

    def __iter__(self) -> Iterator[ManifestRecord]:
        return iter(self._records)
