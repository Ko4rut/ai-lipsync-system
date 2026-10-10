"""Integration checks against real GRID Gold CSVs when configured.

Set LIPSYNC_DATA_ROOT as an environment variable or put it in the repo .env.
No heavy Silver TARs are opened by this test.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.lipsync.datasets import GoldManifestIndex


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dataset_root() -> Path:
    """Explicit environment variables take precedence over the optional .env."""
    configured = os.getenv("LIPSYNC_DATA_ROOT")

    if not configured:
        try:
            from dotenv import dotenv_values
        except ImportError:
            pytest.skip(
                "Set LIPSYNC_DATA_ROOT or install python-dotenv to read the project .env"
            )
        configured = dotenv_values(PROJECT_ROOT / ".env").get("LIPSYNC_DATA_ROOT")

    if not configured:
        pytest.skip("LIPSYNC_DATA_ROOT is not configured (environment or .env)")

    root = Path(configured).expanduser()
    if not root.is_dir():
        pytest.fail(f"Configured LIPSYNC_DATA_ROOT does not exist: {root}")
    return root


@pytest.mark.integration
def test_load_real_gold_manifests() -> None:
    """Read actual Gold manifests and check speaker-disjoint splits."""
    root = _dataset_root()
    speakers_by_split: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        manifest = GoldManifestIndex(root=root, split=split)
        assert len(manifest) > 0, f"{split} manifest has no utterances"

        print(f"\\n===== {split.upper()} =====")
        print(f"Total utterances: {len(manifest)}")

        for index in range(min(3, len(manifest))):
            record = manifest[index]
            print(
                f"Speaker: {record.speaker_id} | "
                f"Utterance: {record.utterance_id} | "
                f"Frames: {record.frame_count}"
            )

        speakers: set[str] = set()
        for record in manifest:
            assert record.split == split
            speakers.add(record.speaker_id)
        speakers_by_split[split] = speakers

    assert speakers_by_split["train"].isdisjoint(speakers_by_split["val"])
    assert speakers_by_split["train"].isdisjoint(speakers_by_split["test"])
    assert speakers_by_split["val"].isdisjoint(speakers_by_split["test"])
