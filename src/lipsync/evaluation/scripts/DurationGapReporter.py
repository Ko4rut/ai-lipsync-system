
from pathlib import Path

from lipsync.setting.config import LipSyncSettings

from src.lipsync.datasets.manifest import GoldManifestIndex
from src.lipsync.evaluation.duration_report import DurationGapReporter


def main() -> None:
    # STEP 1: Xác định thư mục gốc của project
    project_root = Path(__file__).resolve().parents[4]

    # STEP 2: Đọc biến môi trường từ .env
    settings = LipSyncSettings.from_env(project_root / ".env")
    data_root = settings.data_root

    if not data_root:
        raise RuntimeError(
            "LIPSYNC_DATA_ROOT is not configured in .env"
        )

    # STEP 3: Khởi tạo class DurationGapReporter
    reporter = DurationGapReporter(
        root=data_root,
        output_dir=project_root / "outputs" / "duration_gap",
        fps=settings.fps,
        threshold_ms=settings.duration_gap_threshold_ms,
    )

    # STEP 4: Phân tích từng split
    for split in ("train", "val", "test"):
        print(f"\n========== {split.upper()} ==========")

        manifest = GoldManifestIndex(
            root=data_root,
            split=split,
        )

        rows = []

        # Phân tích tất cả utterance trong split
        for record in manifest:
            duration_row = reporter.analyze_record(record)
            rows.append(duration_row)

        if not rows:
            print("No utterances found.")
            continue

        # STEP 5: In kết quả của utterance đầu tiên
        first_row = rows[0]

        print(f"Speaker: {first_row.speaker_id}")
        print(f"Utterance: {first_row.utterance_id}")
        print(f"Video duration: {first_row.video_duration_sec:.3f}s")
        print(f"Audio duration: {first_row.audio_duration_sec:.3f}s")
        print(f"Duration gap: {first_row.duration_gap_sec:.3f}s")

        # STEP 6: Tổng hợp thống kê toàn split
        summary = reporter.summarize(
            group=split,
            rows=rows,
        )

        print("\n----- SUMMARY -----")
        print(f"Total utterances: {summary.utterances}")
        print(f"Mean gap: {summary.mean_gap_sec:.3f}s")
        print(f"Median gap: {summary.median_gap_sec:.3f}s")
        print(f"Min gap: {summary.min_gap_sec:.3f}s")
        print(f"Max gap: {summary.max_gap_sec:.3f}s")
        print(f"P95 absolute gap: {summary.p95_absolute_gap_sec:.3f}s")

        print(
            f"Exceeds threshold: "
            f"{summary.exceeds_threshold_count} "
            f"({summary.exceeds_threshold_pct:.2f}%)"
        )

        print(
            f"Utterances with missing frames: "
            f"{summary.has_missing_frames_count}"
        )
    print("\n===== TOP 5 NEGATIVE GAPS =====")

    negative_gaps = sorted(
        rows,
        key=lambda row: row.duration_gap_sec,
    )

    for row in negative_gaps[:5]:
        print(
            f"{row.speaker_id}/{row.utterance_id} | "
            f"video={row.video_duration_sec:.3f}s | "
            f"audio={row.audio_duration_sec:.3f}s | "
            f"gap={row.duration_gap_sec:+.3f}s"
        )

    print("\n===== UTTERANCES WITH MISSING FRAMES =====")

    for row in rows:
        if row.missing_frame_count > 0:
            print(
                f"{row.speaker_id}/{row.utterance_id} | "
                f"missing={row.missing_frame_count}"
            )
    if split == "test":
        anomalous_ids = {
            "lbbx6s",
            "pwav7n",
            "swihzs",
            "sbaaza",
            "pbau5n",
        }

        print("\n===== S8 ANOMALY DETAILS =====")

        for record in manifest:
            if record.speaker_id != "s8":
                continue

            if record.utterance_id not in anomalous_ids:
                continue

            print(
                f"\nUtterance: {record.utterance_id}"
                f"\nFrame count: {record.frame_count}"
                f"\nFirst frame: {record.first_frame_index}"
                f"\nLast frame: {record.last_frame_index}"
                f"\nFace detection ratio: {record.face_detection_ratio:.3f}"
                f"\nAudio duration: {record.audio_duration_sec:.3f}s"
            )


if __name__ == "__main__":
    main()
