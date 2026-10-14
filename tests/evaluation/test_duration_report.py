import csv
import pytest
from src.lipsync.datasets.manifest import ManifestRecord
from src.lipsync.evaluation.duration_report import (
    DurationGapReporter,
)    
from src.lipsync.evaluation.models import DurationRow


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

def test_write_csv(tmp_path):

    reporter = DurationGapReporter(root=tmp_path)

    row = DurationRow(
        split="train",
        speaker_id="s1",
        utterance_id="bbaf2n",
        frame_count=75,
        first_frame_index=0,
        last_frame_index=74,
        missing_frame_count=0,
        video_duration_sec=3.0,
        audio_duration_sec=1.19,
        duration_gap_sec=1.81,
        absolute_gap_sec=1.81,
        exceeds_threshold=True,
    )

    output_path = tmp_path / "reports" / "test.csv"

    result = reporter._write_csv(
        path=output_path,
        rows=[row],
        row_type=DurationRow,
    )

    assert result.exists()

    with result.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["speaker_id"] == "s1"
    assert rows[0]["utterance_id"] == "bbaf2n"
    assert float(rows[0]["duration_gap_sec"]) == 1.81