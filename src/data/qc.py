"""QC (Quality Control) module for the GRID preprocessing pipeline.

Validates each utterance's Silver output at sample level.
A single bad utterance must not crash the speaker run.
All QC failures are captured in structured metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from .config import PipelineConfig
from .utils import max_consecutive_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QC reason constants (machine-readable strings)
# ---------------------------------------------------------------------------

class InvalidReason:
    MISSING_AUDIO = "MISSING_AUDIO"
    MISSING_VIDEO = "MISSING_VIDEO"
    DUPLICATE_AUDIO = "DUPLICATE_AUDIO"
    DUPLICATE_VIDEO = "DUPLICATE_VIDEO"
    VIDEO_DECODE_FAILED = "VIDEO_DECODE_FAILED"
    AUDIO_DECODE_FAILED = "AUDIO_DECODE_FAILED"
    INVALID_FPS = "INVALID_FPS"
    NO_FRAMES = "NO_FRAMES"
    FACE_NOT_DETECTED = "FACE_NOT_DETECTED"
    INSUFFICIENT_CONSECUTIVE_FACE_FRAMES = "INSUFFICIENT_CONSECUTIVE_FACE_FRAMES"
    AUDIO_CONVERSION_FAILED = "AUDIO_CONVERSION_FAILED"
    EMPTY_AUDIO = "EMPTY_AUDIO"
    INVALID_OUTPUT_AUDIO_RATE = "INVALID_OUTPUT_AUDIO_RATE"
    CORRUPT_OUTPUT = "CORRUPT_OUTPUT"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


# ---------------------------------------------------------------------------
# Per-utterance QC result
# ---------------------------------------------------------------------------

@dataclass
class UtteranceQCResult:
    utterance_id: str
    valid: bool = False
    invalid_reason: Optional[str] = None

    # Video metrics
    source_fps: float = 0.0
    source_frame_count: int = 0
    cropped_frame_count: int = 0
    face_detection_ratio: float = 0.0
    max_consecutive_face_frames: int = 0

    # Audio metrics
    audio_duration_sec: Optional[float] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None

    # Shard member location
    shard_member_prefix: str = ""

    # Frame index details
    detected_frame_indices: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# QC validation function
# ---------------------------------------------------------------------------

def validate_utterance(
    utterance_id: str,
    silver_utterance_dir: Path,
    *,
    source_fps: float,
    source_frame_count: int,
    cropped_frame_count: int,
    detected_frame_indices: List[int],
    audio_duration_sec: Optional[float],
    audio_sample_rate: Optional[int],
    audio_channels: Optional[int],
    speaker_id: str,
    config: PipelineConfig,
) -> UtteranceQCResult:
    """Run all QC checks for one utterance and return a result.

    This function never raises; all failures become an invalid QCResult.
    """
    res = UtteranceQCResult(
        utterance_id=utterance_id,
        source_fps=source_fps,
        source_frame_count=source_frame_count,
        cropped_frame_count=cropped_frame_count,
        audio_duration_sec=audio_duration_sec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        detected_frame_indices=sorted(detected_frame_indices),
        shard_member_prefix=f"{speaker_id}/{utterance_id}/",
    )

    if source_frame_count == 0:
        res.invalid_reason = InvalidReason.NO_FRAMES
        return res

    # Detection ratio
    res.face_detection_ratio = (
        cropped_frame_count / source_frame_count if source_frame_count > 0 else 0.0
    )
    res.max_consecutive_face_frames = max_consecutive_run(detected_frame_indices)

    # --- Video QC ---
    if cropped_frame_count == 0:
        res.invalid_reason = InvalidReason.FACE_NOT_DETECTED
        return res

    if res.face_detection_ratio < config.min_face_detection_ratio:
        res.invalid_reason = InvalidReason.FACE_NOT_DETECTED
        return res

    if res.max_consecutive_face_frames < config.min_consecutive_face_frames:
        res.invalid_reason = InvalidReason.INSUFFICIENT_CONSECUTIVE_FACE_FRAMES
        return res

    # Check that cropped JPEGs are readable
    for idx in detected_frame_indices[:3]:  # sample spot-check first 3
        jpeg_path = silver_utterance_dir / f"{idx}.jpg"
        if not jpeg_path.exists():
            res.invalid_reason = InvalidReason.CORRUPT_OUTPUT
            return res
        img = cv2.imread(str(jpeg_path))
        if img is None:
            res.invalid_reason = InvalidReason.CORRUPT_OUTPUT
            return res

    # --- Audio QC ---
    audio_path = silver_utterance_dir / "audio.wav"
    if not audio_path.exists():
        res.invalid_reason = InvalidReason.AUDIO_DECODE_FAILED
        return res

    if audio_sample_rate != config.target_audio_sample_rate:
        res.invalid_reason = InvalidReason.INVALID_OUTPUT_AUDIO_RATE
        return res

    if audio_duration_sec is None or audio_duration_sec < config.min_audio_duration_sec:
        res.invalid_reason = InvalidReason.EMPTY_AUDIO
        return res

    res.valid = True
    return res


# ---------------------------------------------------------------------------
# Speaker QC summary
# ---------------------------------------------------------------------------

@dataclass
class SpeakerQCSummary:
    speaker_id: str
    total_candidates: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    invalid_reason_counts: Dict[str, int] = field(default_factory=dict)
    results: List[UtteranceQCResult] = field(default_factory=list)

    def add(self, result: UtteranceQCResult) -> None:
        self.results.append(result)
        self.total_candidates += 1
        if result.valid:
            self.valid_samples += 1
        else:
            self.invalid_samples += 1
            reason = result.invalid_reason or InvalidReason.UNKNOWN_ERROR
            self.invalid_reason_counts[reason] = (
                self.invalid_reason_counts.get(reason, 0) + 1
            )

    def log_summary(self) -> None:
        logger.info(
            "[%s] QC summary: %d total, %d valid, %d invalid",
            self.speaker_id,
            self.total_candidates,
            self.valid_samples,
            self.invalid_samples,
        )
        if self.invalid_reason_counts:
            for reason, count in sorted(self.invalid_reason_counts.items()):
                logger.info("  - %s: %d", reason, count)
