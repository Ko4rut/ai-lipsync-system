from __future__ import annotations

import csv
import math

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from statistics import mean, median

from src.lipsync.datasets.manifest import (
    GoldManifestIndex,
    ManifestRecord,
)
@dataclass(frozen=True)
class SummaryRow:
    """Kết quả thống kê của một nhóm utterance."""

    group: str
    utterances: int

    mean_gap_sec: float
    median_gap_sec: float

    min_gap_sec: float
    max_gap_sec: float

    p95_absolute_gap_sec: float

    exceeds_threshold_count: int
    exceeds_threshold_pct: float

    has_missing_frames_count: int

@dataclass(frozen=True)
class DurationRow:
    split: str
    speaker_id: str
    utterance_id: str

    frame_count: int
    first_frame_index: int
    last_frame_index: int
    missing_frame_count: int

    video_duration_sec: float
    audio_duration_sec: float

    duration_gap_sec: float
    absolute_gap_sec: float

    exceeds_threshold: bool
    
    
class DurationGapReporter:

    def __init__(
        self,
        root: str | Path,
        output_dir: str | Path = "outputs/duration_gap",
        fps: float = 25.0,
        threshold_ms: float = 80.0,
    ) -> None:

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be positive and finite")

        if not math.isfinite(threshold_ms) or threshold_ms < 0:
            raise ValueError("Invalid threshold")

        self.root = Path(root).expanduser()
        self.output_dir = Path(output_dir).expanduser()

        self.fps = fps
        self.threshold_ms = threshold_ms

    def analyze_record(
        self,
        record: ManifestRecord,
    ) -> DurationRow:

        # 1. Tính khoảng frame
        frame_span = (
            record.last_frame_index
            - record.first_frame_index
            + 1
        )

        # 2. Kiểm tra metadata
        if record.frame_count <= 0:
            raise ValueError("Invalid frame count")

        if frame_span < record.frame_count:
            raise ValueError("Frame count exceeds frame span")

        audio_duration = record.audio_duration_sec

        if not math.isfinite(audio_duration) or audio_duration <= 0:
            raise ValueError("Invalid audio duration")

        # 3. Số frame còn thiếu trong khoảng chỉ số
        missing_frames = frame_span - record.frame_count

        # 4. Tính thời lượng video
        video_duration = frame_span / self.fps

        # 5. Tính duration gap
        duration_gap = video_duration - audio_duration
        absolute_gap = abs(duration_gap)

        # 6. Kiểm tra vượt ngưỡng
        absolute_gap_ms = absolute_gap * 1000

        exceeds_threshold = (
            absolute_gap_ms > self.threshold_ms
            and not math.isclose(
                absolute_gap_ms,
                self.threshold_ms,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )

        # 7. Trả về kết quả
        return DurationRow(
            split=record.split,
            speaker_id=record.speaker_id,
            utterance_id=record.utterance_id,
            frame_count=record.frame_count,
            first_frame_index=record.first_frame_index,
            last_frame_index=record.last_frame_index,
            missing_frame_count=missing_frames,
            video_duration_sec=video_duration,
            audio_duration_sec=audio_duration,
            duration_gap_sec=duration_gap,
            absolute_gap_sec=absolute_gap,
            exceeds_threshold=exceeds_threshold,
        )

    @staticmethod
    def summarize(
        group: str,
        rows: list[DurationRow],
    ) -> SummaryRow:
        """Tổng hợp thống kê duration gap của một nhóm."""

        # STEP 1: Kiểm tra danh sách đầu vào
        if len(rows) == 0:
            raise ValueError(
                f"Cannot summarize empty group: {group}"
            )

        # STEP 2: Khởi tạo các biến
        gaps: list[float] = []
        absolute_gaps: list[float] = []

        exceeded_count = 0
        missing_frames_count = 0

        # STEP 3: Duyệt từng utterance
        for row in rows:

            gaps.append(row.duration_gap_sec)
            absolute_gaps.append(row.absolute_gap_sec)

            if row.exceeds_threshold:
                exceeded_count += 1

            if row.missing_frame_count > 0:
                missing_frames_count += 1

        # STEP 4: Tính thống kê cơ bản
        total_utterances = len(rows)

        mean_gap = mean(gaps)
        median_gap = median(gaps)

        min_gap = min(gaps)
        max_gap = max(gaps)

        # STEP 5: Tính P95 của độ lệch tuyệt đối
        absolute_gaps.sort()

        p95_index = math.ceil(
            0.95 * total_utterances
        ) - 1

        p95_gap = absolute_gaps[p95_index]

        # STEP 6: Tính tỷ lệ vượt ngưỡng
        exceeded_percentage = (
            exceeded_count / total_utterances
        ) * 100.0

        # STEP 7: Trả về kết quả
        return SummaryRow(
            group=group,
            utterances=total_utterances,
            mean_gap_sec=mean_gap,
            median_gap_sec=median_gap,
            min_gap_sec=min_gap,
            max_gap_sec=max_gap,
            p95_absolute_gap_sec=p95_gap,
            exceeds_threshold_count=exceeded_count,
            exceeds_threshold_pct=exceeded_percentage,
            has_missing_frames_count=missing_frames_count,
        )

    # def _write_csv(self, path, rows, row_type):
    #     # Ghi dữ liệu ra CSV

    # def generate(self):
    #     # Điều phối toàn bộ quá trình tạo báo cáo