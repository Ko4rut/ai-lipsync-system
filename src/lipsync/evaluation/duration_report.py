from __future__ import annotations

import csv
import math

from pathlib import Path
from statistics import mean, median
from dataclasses import asdict, fields
from src.lipsync.datasets.manifest import (
    GoldManifestIndex,
    ManifestRecord,
)
from .models import DurationRow, SummaryRow

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


    def _write_csv(
        self,
        path: str | Path,
        rows: list[DurationRow] | list[SummaryRow],
        row_type: type[DurationRow] | type[SummaryRow],
    ) -> Path:
        """Export dataclass rows to a UTF-8 CSV file."""

        # STEP 1: Xác định vị trí file đầu ra
        output_path = Path(path)

        # STEP 2: Kiểm tra loại dữ liệu
        for row in rows:
            if not isinstance(row, row_type):
                raise TypeError(
                    f"Expected {row_type.__name__}, "
                    f"got {type(row).__name__}"
                )

        # STEP 3: Tạo thư mục nếu chưa tồn tại
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # STEP 4: Lấy tên các cột từ dataclass
        column_names = []

        for field in fields(row_type):
            column_names.append(field.name)

        # STEP 5: Mở file CSV để ghi
        with output_path.open(
            mode="w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=column_names,
            )

            # STEP 6: Ghi header
            writer.writeheader()

            # STEP 7: Ghi từng dòng dữ liệu
            for row in rows:
                row_data = asdict(row)
                writer.writerow(row_data)

        # STEP 8: Trả về đường dẫn file
        return output_path

    def generate(self) -> dict[str, Path]:
        """Analyze train/val/test and export duration-gap reports."""

        all_rows: list[DurationRow] = []
        split_summaries: list[SummaryRow] = []
        speaker_summaries: list[SummaryRow] = []

        # STEP 1: Analyze each split
        for split in ("train", "val", "test"):

            manifest = GoldManifestIndex(
                root=self.root,
                split=split,
            )

            rows: list[DurationRow] = []
            speaker_groups: dict[str, list[DurationRow]] = {}

            for record in manifest:
                row = self.analyze_record(record)
                rows.append(row)

                speaker_id = row.speaker_id

                if speaker_id not in speaker_groups:
                    speaker_groups[speaker_id] = []

                speaker_groups[speaker_id].append(row)

            if not rows:
                continue

            # STEP 2: Aggregate split-level statistics
            split_summary = self.summarize(
                group=split,
                rows=rows,
            )

            split_summaries.append(split_summary)
            all_rows.extend(rows)

            # STEP 3: Aggregate speaker-level statistics
            for speaker_id, speaker_rows in speaker_groups.items():

                speaker_summary = self.summarize(
                    group=f"{split}/{speaker_id}",
                    rows=speaker_rows,
                )

                speaker_summaries.append(speaker_summary)

        if not all_rows:
            raise ValueError("No utterances found in Gold manifests")

        # STEP 4: Aggregate overall statistics
        overall_summary = self.summarize(
            group="ALL",
            rows=all_rows,
        )

        split_summaries.append(overall_summary)

        # STEP 5: Export CSV files
        utterance_path = self._write_csv(
            path=self.output_dir / "utterance_gaps.csv",
            rows=all_rows,
            row_type=DurationRow,
        )

        split_path = self._write_csv(
            path=self.output_dir / "split_summary.csv",
            rows=split_summaries,
            row_type=SummaryRow,
        )

        speaker_path = self._write_csv(
            path=self.output_dir / "speaker_summary.csv",
            rows=speaker_summaries,
            row_type=SummaryRow,
        )

        # STEP 6: Return generated report paths
        return {
            "utterance_gaps": utterance_path,
            "split_summary": split_path,
            "speaker_summary": speaker_path,
        }
