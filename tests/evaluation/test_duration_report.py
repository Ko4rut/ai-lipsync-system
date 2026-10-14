import pytest

from src.lipsync.datasets.manifest import ManifestRecord
from src.lipsync.evaluation.duration_report import (
    DurationGapReporter,
)


def test_duration_gap(tmp_path):

    record = ManifestRecord(
        split="train",
        speaker_id="s1",
        utterance_id="bbaf2n",
        shard_member_prefix="s1/bbaf2n/",
        frame_count=75,
        first_frame_index=0,
        last_frame_index=74,
        face_detection_ratio=1.0,
        audio_duration_sec=1.19,
        audio_sample_rate=16000,
        source_shard_path="silver/shards/s1.tar",
    )

    reporter = DurationGapReporter(root=tmp_path)

    result = reporter.analyze_record(record)

    assert result.video_duration_sec == pytest.approx(3.0)
    assert result.audio_duration_sec == pytest.approx(1.19)
    assert result.duration_gap_sec == pytest.approx(1.81)
    assert result.missing_frame_count == 0
    assert result.exceeds_threshold is True