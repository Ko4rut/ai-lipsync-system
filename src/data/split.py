"""Deterministic speaker-disjoint Gold split and manifest generation.

Gold layer:
- Never duplicates Silver data (no TAR copies).
- Contains train/val/test manifests at utterance level.
- split_metadata.json records split provenance.

Split is fully deterministic given seed=42 and the canonical speaker list.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    EXCLUDED_SPEAKERS,
    SCHEMA_VERSION,
    SPLIT_SEED,
    SPLIT_TEST_COUNT,
    SPLIT_TRAIN_COUNT,
    SPLIT_VAL_COUNT,
    USABLE_SPEAKERS,
    PipelineConfig,
)
from .utils import atomic_json_write

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------

def compute_speaker_split(
    *,
    seed: int = SPLIT_SEED,
    train_count: int = SPLIT_TRAIN_COUNT,
    val_count: int = SPLIT_VAL_COUNT,
    test_count: int = SPLIT_TEST_COUNT,
    usable_speakers: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (train_speakers, val_speakers, test_speakers).

    Uses Python's standard ``random.Random`` with *seed* for reproducibility.
    The canonical usable speaker list is constructed deterministically
    (sorted by speaker number) before shuffling.
    """
    speakers = list(usable_speakers if usable_speakers is not None else USABLE_SPEAKERS)
    rng = random.Random(seed)
    rng.shuffle(speakers)

    train = speakers[:train_count]
    val = speakers[train_count: train_count + val_count]
    test = speakers[train_count + val_count: train_count + val_count + test_count]
    return train, val, test


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

_MANIFEST_FIELDNAMES = [
    "split",
    "speaker_id",
    "utterance_id",
    "shard_path",
    "shard_member_prefix",
    "frame_count",
    "first_frame_index",
    "last_frame_index",
    "face_detection_ratio",
    "audio_duration_sec",
    "audio_sample_rate",
    "valid",
]


def _load_speaker_metadata(metadata_path: Path) -> Optional[dict]:
    """Load and return a speaker's Silver JSON metadata, or None on failure."""
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cannot read speaker metadata %s: %s", metadata_path, exc)
        return None


def build_gold_manifests(
    config: PipelineConfig,
    *,
    allow_partial: bool = False,
    partial_label: str = "partial",
) -> bool:
    """Build Gold manifests from all Silver speaker metadata.

    Parameters
    ----------
    allow_partial:
        If True, create manifests even when some speakers are missing/failed.
        The resulting files will be labelled as partial in split_metadata.json.
    partial_label:
        Added to manifest filenames when *allow_partial* is True.

    Returns True on success, False if preconditions are not satisfied and
    *allow_partial* is False.
    """
    train_speakers, val_speakers, test_speakers = compute_speaker_split(
        seed=config.split_seed,
        train_count=config.split_train_count,
        val_count=config.split_val_count,
        test_count=config.split_test_count,
        usable_speakers=config.usable_speakers(),
    )

    split_map: Dict[str, str] = {}
    for s in train_speakers:
        split_map[s] = "train"
    for s in val_speakers:
        split_map[s] = "val"
    for s in test_speakers:
        split_map[s] = "test"

    # Check all speakers are DONE with valid metadata
    missing_speakers: List[str] = []
    for speaker_id in config.usable_speakers():
        meta_path = config.silver_metadata_dir / f"{speaker_id}.json"
        meta = _load_speaker_metadata(meta_path)
        if meta is None or meta.get("status") != "DONE":
            missing_speakers.append(speaker_id)

    if missing_speakers:
        logger.warning(
            "Gold generation: %d speakers missing or not DONE: %s",
            len(missing_speakers),
            missing_speakers[:10],
        )
        if not allow_partial:
            logger.error(
                "Cannot create Gold manifests: %d speakers incomplete. "
                "Use --allow-partial-gold to create partial manifests.",
                len(missing_speakers),
            )
            return False

    # Gather rows per split
    rows: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    for speaker_id in config.usable_speakers():
        if speaker_id not in split_map:
            continue
        split_name = split_map[speaker_id]

        meta_path = config.silver_metadata_dir / f"{speaker_id}.json"
        meta = _load_speaker_metadata(meta_path)
        if meta is None:
            logger.warning("[%s] No metadata available; skipping from Gold.", speaker_id)
            continue

        shard_path = (meta.get("shard") or {}).get("path") or str(
            config.silver_shards_dir / f"{speaker_id}.tar"
        )

        for sample in meta.get("samples", []):
            if not sample.get("valid", False):
                continue

            indices = sample.get("detected_frame_indices") or []
            frame_count = sample.get("cropped_frame_count", 0)
            first_idx = min(indices) if indices else None
            last_idx = max(indices) if indices else None

            rows[split_name].append({
                "split": split_name,
                "speaker_id": speaker_id,
                "utterance_id": sample.get("utterance_id", ""),
                "shard_path": shard_path,
                "shard_member_prefix": sample.get("shard_member_prefix", ""),
                "frame_count": frame_count,
                "first_frame_index": first_idx,
                "last_frame_index": last_idx,
                "face_detection_ratio": sample.get("face_detection_ratio", ""),
                "audio_duration_sec": sample.get("audio_duration_sec", ""),
                "audio_sample_rate": sample.get("audio_sample_rate", ""),
                "valid": True,
            })

    gold_dir = config.gold_dir
    gold_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        if allow_partial and missing_speakers:
            fname = f"{split_name}_manifest_{partial_label}.csv"
        else:
            fname = f"{split_name}_manifest.csv"
        csv_path = gold_dir / fname
        _write_manifest_csv(csv_path, rows[split_name])
        logger.info(
            "Gold [%s] manifest: %d utterances -> %s", split_name, len(rows[split_name]), csv_path
        )

    # Split metadata
    split_meta = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "speaker_disjoint",
        "seed": config.split_seed,
        "train_count": config.split_train_count,
        "val_count": config.split_val_count,
        "test_count": config.split_test_count,
        "train_speakers": train_speakers,
        "val_speakers": val_speakers,
        "test_speakers": test_speakers,
        "excluded_speakers": list(config.excluded_speakers),
        "partial": bool(missing_speakers) and allow_partial,
        "missing_speakers": missing_speakers if missing_speakers else [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "utterance_counts": {k: len(v) for k, v in rows.items()},
    }
    meta_fname = "split_metadata.json"
    atomic_json_write(gold_dir / meta_fname, split_meta)
    logger.info("Gold split metadata: %s", gold_dir / meta_fname)
    return True


def _write_manifest_csv(path: Path, rows: List[dict]) -> None:
    """Write *rows* to a CSV at *path* atomically."""
    tmp = path.with_suffix(".csv.tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        tmp.rename(path)
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise
