"""Data-transfer models for audio/video duration-gap analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DurationRow:
    """Duration analysis for one Gold manifest utterance."""

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


@dataclass(frozen=True)
class SummaryRow:
    """Aggregated signed-gap and absolute-gap statistics for a group."""

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
