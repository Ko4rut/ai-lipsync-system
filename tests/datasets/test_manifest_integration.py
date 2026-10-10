
import os
from pathlib import Path

import pytest

from src.lipsync.datasets.manifest import GoldManifestIndex


def test_load_real_gold_manifests():
    """Kiểm tra GoldManifestIndex với dataset thực tế."""

    # STEP 1: Lấy đường dẫn dataset từ environment
    dataset_path = os.getenv("LIPSYNC_DATA_ROOT")

    if not dataset_path:
        pytest.skip("GRID_DATASET_ROOT is not configured")

    dataset_root = Path(dataset_path)

    # STEP 2: Khai báo các split cần kiểm tra
    splits = ["train", "val", "test"]

    speakers_by_split = {}

    # STEP 3: Đọc từng Gold Manifest
    for split in splits:

        manifest = GoldManifestIndex(
            root=dataset_root,
            split=split
        )

        # Kiểm tra có dữ liệu
        assert len(manifest) > 0

        print(f"\n===== {split.upper()} =====")
        print(f"Total utterances: {len(manifest)}")

        # In thử 3 record đầu tiên
        for index in range(min(3, len(manifest))):
            record = manifest[index]

            print(
                f"Speaker: {record.speaker_id} | "
                f"Utterance: {record.utterance_id} | "
                f"Frames: {record.frame_count}"
            )

        # Kiểm tra speaker trong mỗi split
        speakers = set()

        for record in manifest:
            assert record.split == split
            speakers.add(record.speaker_id)

        speakers_by_split[split] = speakers

    # STEP 4: Kiểm tra speaker không bị trùng giữa các split
    assert speakers_by_split["train"].isdisjoint(
        speakers_by_split["val"]
    )

    assert speakers_by_split["train"].isdisjoint(
        speakers_by_split["test"]
    )

    assert speakers_by_split["val"].isdisjoint(
        speakers_by_split["test"]
    )
