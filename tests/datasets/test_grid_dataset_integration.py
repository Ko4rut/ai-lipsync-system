
"""Integration tests for GridUtteranceDataset using real GRID data."""

import os
from pathlib import Path

import numpy as np
import pytest
from dotenv import load_dotenv

from src.lipsync.datasets.grid_dataset import GridUtteranceDataset


@pytest.fixture(scope="module")
def dataset_root() -> Path:
    """Get the real GRID dataset root from the project .env file."""

    # STEP 1: Load environment configuration.
    project_root = Path(__file__).resolve().parents[2]
    env_file = project_root / ".env"

    load_dotenv(env_file)

    # STEP 2: Read the dataset path.
    root_value = os.getenv("LIPSYNC_DATA_ROOT")

    if not root_value:
        pytest.skip("LIPSYNC_DATA_ROOT is not configured")

    root = Path(root_value).expanduser()

    # STEP 3: Verify the dataset directories exist.
    if not root.is_dir():
        pytest.fail(f"Dataset root does not exist: {root}")

    gold_dir = root / "gold"
    silver_dir = root / "silver" / "shards"

    if not gold_dir.is_dir():
        pytest.fail(f"Gold directory not found: {gold_dir}")

    if not silver_dir.is_dir():
        pytest.fail(f"Silver shards directory not found: {silver_dir}")

    return root


@pytest.mark.parametrize(
    "split, index",
    [
        ("train", 0),
        ("val", 5),
        ("test", 0),
    ],
)
def test_real_grid_dataset(
    dataset_root: Path,
    split: str,
    index: int,
) -> None:
    """Validate decoded frames and audio from a real GRID utterance."""

    # STEP 1: Initialize the dataset.
    dataset = GridUtteranceDataset(
        root=dataset_root,
        split=split,
        frame_size=(96, 96),
    )

    try:
        # STEP 2: Verify the requested sample exists.
        assert len(dataset) > 0
        assert index < len(dataset)

        record = dataset.index[index]

        # STEP 3: Read and decode a real utterance.
        sample = dataset[index]

        # STEP 4: Check speaker and utterance metadata.
        assert sample.speaker_id == record.speaker_id
        assert sample.utterance_id == record.utterance_id

        # STEP 5: Validate the decoded video frames.
        assert sample.frames.ndim == 4
        assert sample.frames.shape[1:] == (96, 96, 3)
        assert sample.frames.dtype == np.uint8

        frame_count = len(sample.frames)
        assert frame_count == record.frame_count
        assert frame_count == len(sample.frame_indices)

        assert sample.frame_indices[0] == record.first_frame_index
        assert sample.frame_indices[-1] == record.last_frame_index

        # STEP 6: Validate decoded audio.
        assert sample.audio.ndim == 1
        assert sample.audio.dtype == np.float32
        assert sample.sample_rate == record.audio_sample_rate

        assert len(sample.audio) > 0
        assert np.all(np.isfinite(sample.audio))
        assert np.all(sample.audio >= -1.0)
        assert np.all(sample.audio <= 1.0)

        # STEP 7: Validate the calculated durations.
        expected_audio_duration = len(sample.audio) / sample.sample_rate

        assert sample.audio_duration_sec == pytest.approx(
            expected_audio_duration
        )

        assert sample.video_duration_sec > 0

        # STEP 8: Print a compact report.
        print(f"\n===== REAL GRID TEST: {split.upper()} =====")
        print(f"Dataset length: {len(dataset)}")
        print(f"Speaker: {sample.speaker_id}")
        print(f"Utterance: {sample.utterance_id}")
        print(f"Frames: {sample.frames.shape}")
        print(f"Frames dtype: {sample.frames.dtype}")
        print(f"Audio: {sample.audio.shape}")
        print(f"Audio dtype: {sample.audio.dtype}")
        print(f"Sample rate: {sample.sample_rate}")
        print(f"Video duration: {sample.video_duration_sec:.3f}s")
        print(f"Audio duration: {sample.audio_duration_sec:.3f}s")
        print("All assertions passed!")

    finally:
        # STEP 9: Always release open TAR handles.
        dataset.close()
