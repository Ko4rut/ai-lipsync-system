"""Centralized configuration for the GRID Audio-Visual Speech Corpus pipeline.

All magic constants, paths, and tuneable parameters live here.
Nothing in other modules should embed bare numeric or string literals
that belong to project-level configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Speaker catalogue
# ---------------------------------------------------------------------------

#: All speaker IDs present on the Sheffield GRID server.
ALL_GRID_SPEAKERS: List[str] = [f"s{i}" for i in range(1, 35)]

#: Speakers that cannot be used for this audio-visual pipeline.
EXCLUDED_SPEAKERS: List[str] = ["s21"]

#: Usable speakers in deterministic order (no filesystem dependency).
USABLE_SPEAKERS: List[str] = [s for s in ALL_GRID_SPEAKERS if s not in EXCLUDED_SPEAKERS]

assert len(USABLE_SPEAKERS) == 33, "Expected exactly 33 usable GRID speakers."


# ---------------------------------------------------------------------------
# Remote Bronze sources
# ---------------------------------------------------------------------------

#: Base URL of the official Sheffield GRID corpus server.
SHEFFIELD_BASE_URL: str = "https://spandh.dcs.shef.ac.uk"

#: URL template for normal-quality video archive (per-speaker).
#: Format: SHEFFIELD_VIDEO_URL_TEMPLATE.format(speaker_id="s1")
SHEFFIELD_VIDEO_URL_TEMPLATE: str = (
    SHEFFIELD_BASE_URL + "/gridcorpus/{speaker_id}/video/{speaker_id}.mpg_vcd.zip"
)

#: URL template for audio archive (per-speaker).
SHEFFIELD_AUDIO_URL_TEMPLATE: str = (
    SHEFFIELD_BASE_URL + "/gridcorpus/{speaker_id}/audio/{speaker_id}.tar"
)

#: Zenodo record ID containing the full GRID dataset as fallback.
ZENODO_RECORD_ID: str = "3625687"

#: Zenodo API record URL used to resolve file download links.
ZENODO_RECORD_API_URL: str = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

#: Zenodo per-speaker archive filename template (video/full bundle).
ZENODO_SPEAKER_FILENAME_TEMPLATE: str = "{speaker_id}.zip"

#: Zenodo combined 25 kHz audio archive filename.
ZENODO_AUDIO_ARCHIVE_FILENAME: str = "audio_25k.zip"


# ---------------------------------------------------------------------------
# Audio / Video processing parameters
# ---------------------------------------------------------------------------

#: Expected video frame rate for GRID corpus utterances.
EXPECTED_VIDEO_FPS: float = 25.0

#: Tolerance ±% before an observed FPS is considered materially different.
#: Example: 0.05 means FPS within [23.75, 26.25] is accepted.
FPS_TOLERANCE_FRACTION: float = 0.05

#: Original GRID audio sample rate.
GRID_AUDIO_SAMPLE_RATE: int = 25_000

#: Target audio sample rate for Wav2Lip training.
TARGET_AUDIO_SAMPLE_RATE: int = 16_000

#: Target number of audio channels (mono).
TARGET_AUDIO_CHANNELS: int = 1

#: PCM encoding for output WAV files (signed 16-bit).
TARGET_AUDIO_PCM_FORMAT: str = "pcm_s16le"

#: Minimum face detection ratio for an utterance to be considered valid.
#: 0.5 means at least half of all decoded frames must have a detected face.
MIN_FACE_DETECTION_RATIO: float = 0.5

#: Minimum number of consecutive source-frame indices with a detected face
#: for an utterance to be valid (Wav2Lip needs a window of ≥5).
MIN_CONSECUTIVE_FACE_FRAMES: int = 5

#: JPEG quality for saved face crops (1–95; 95 is near-lossless).
JPEG_QUALITY: int = 95

#: Face detection batch size (frames fed to the detector at once).
FACE_DETECTION_BATCH_SIZE: int = 16

#: Minimum audio duration (seconds) for an utterance to be considered valid.
MIN_AUDIO_DURATION_SEC: float = 0.1


# ---------------------------------------------------------------------------
# Silver state machine
# ---------------------------------------------------------------------------

class SpeakerStatus:
    """Allowed values for the speaker processing state machine."""
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    EXTRACTING = "EXTRACTING"
    PREPROCESSING = "PREPROCESSING"
    QC = "QC"
    PACKAGING = "PACKAGING"
    UPLOADING = "UPLOADING"
    DONE = "DONE"
    FAILED = "FAILED"


#: All statuses that represent a terminal success state.
TERMINAL_SUCCESS_STATUSES = frozenset({SpeakerStatus.DONE})

#: Statuses that should be retried on the next run.
RETRYABLE_STATUSES = frozenset({
    SpeakerStatus.FAILED,
    SpeakerStatus.PENDING,
    SpeakerStatus.DOWNLOADING,
    SpeakerStatus.EXTRACTING,
    SpeakerStatus.PREPROCESSING,
    SpeakerStatus.QC,
    SpeakerStatus.PACKAGING,
    SpeakerStatus.UPLOADING,
})


# ---------------------------------------------------------------------------
# Gold split parameters
# ---------------------------------------------------------------------------

#: Random seed for deterministic speaker-disjoint Gold split.
SPLIT_SEED: int = 42

#: Number of speakers in the training split.
SPLIT_TRAIN_COUNT: int = 27

#: Number of speakers in the validation split.
SPLIT_VAL_COUNT: int = 3

#: Number of speakers in the test split.
SPLIT_TEST_COUNT: int = 3


# ---------------------------------------------------------------------------
# Paths (runtime paths are resolved at runtime; these are defaults)
# ---------------------------------------------------------------------------

#: Default Google Drive root for this project (Colab).
DEFAULT_DRIVE_ROOT: str = (
    # "/content/drive/MyDrive/2026-2027/coding/pattern_recognition/20261002_grid_dataset" # Đường dẫn drive của Tùng Thiện
    "/content/drive/MyDrive/20261002_grid_dataset" # Đường dẫn drive của các bạn khác
)

#: Default transient Colab local workspace root.
DEFAULT_WORKSPACE_ROOT: str = "/content/grid_pipeline"

#: Schema version embedded in all persisted JSON artefacts.
SCHEMA_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Download parameters
# ---------------------------------------------------------------------------

#: Maximum number of attempts per download (primary source).
DOWNLOAD_MAX_RETRIES: int = 5

#: Initial delay (seconds) before the first retry.
DOWNLOAD_RETRY_INITIAL_DELAY_SEC: float = 2.0

#: Exponential backoff multiplier between retries.
DOWNLOAD_RETRY_BACKOFF_FACTOR: float = 2.0

#: Maximum delay (seconds) between retries.
DOWNLOAD_RETRY_MAX_DELAY_SEC: float = 60.0

#: HTTP connection timeout (seconds).
DOWNLOAD_CONNECT_TIMEOUT_SEC: float = 30.0

#: HTTP read timeout per chunk (seconds).
DOWNLOAD_READ_TIMEOUT_SEC: float = 60.0

#: Download chunk size (bytes) for streaming.
DOWNLOAD_CHUNK_SIZE: int = 1024 * 1024  # 1 MiB

#: Chunk size for SHA-256 hashing.
HASH_CHUNK_SIZE: int = 1024 * 1024  # 1 MiB


# ---------------------------------------------------------------------------
# Dataclass view (optional — used by PipelineConfig)
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Runtime-resolved configuration assembled from defaults + CLI overrides.

    Paths in this object are always ``pathlib.Path`` instances resolved at
    construction time.  Modules should accept a ``PipelineConfig`` rather than
    reading global constants directly so that tests can override any field.
    """

    # Paths
    drive_root: Path = field(default_factory=lambda: Path(DEFAULT_DRIVE_ROOT))
    workspace_root: Path = field(default_factory=lambda: Path(DEFAULT_WORKSPACE_ROOT))

    # Speaker selection
    excluded_speakers: List[str] = field(default_factory=lambda: list(EXCLUDED_SPEAKERS))

    # Audio / Video
    expected_fps: float = EXPECTED_VIDEO_FPS
    fps_tolerance_fraction: float = FPS_TOLERANCE_FRACTION
    target_audio_sample_rate: int = TARGET_AUDIO_SAMPLE_RATE
    target_audio_channels: int = TARGET_AUDIO_CHANNELS
    min_face_detection_ratio: float = MIN_FACE_DETECTION_RATIO
    min_consecutive_face_frames: int = MIN_CONSECUTIVE_FACE_FRAMES
    jpeg_quality: int = JPEG_QUALITY
    face_detection_batch_size: int = FACE_DETECTION_BATCH_SIZE
    min_audio_duration_sec: float = MIN_AUDIO_DURATION_SEC

    # Gold split
    split_seed: int = SPLIT_SEED
    split_train_count: int = SPLIT_TRAIN_COUNT
    split_val_count: int = SPLIT_VAL_COUNT
    split_test_count: int = SPLIT_TEST_COUNT

    # Download
    download_max_retries: int = DOWNLOAD_MAX_RETRIES
    download_retry_initial_delay_sec: float = DOWNLOAD_RETRY_INITIAL_DELAY_SEC
    download_retry_backoff_factor: float = DOWNLOAD_RETRY_BACKOFF_FACTOR
    download_retry_max_delay_sec: float = DOWNLOAD_RETRY_MAX_DELAY_SEC
    download_connect_timeout_sec: float = DOWNLOAD_CONNECT_TIMEOUT_SEC
    download_read_timeout_sec: float = DOWNLOAD_READ_TIMEOUT_SEC

    # Runtime behaviour
    limit_utterances: int = 0          # 0 = no limit
    keep_workspace: bool = False
    dry_run: bool = False
    force: bool = False

    # -----------------------------------------------------------------
    # Derived path helpers
    # -----------------------------------------------------------------

    @property
    def silver_shards_dir(self) -> Path:
        return self.drive_root / "silver" / "shards"

    @property
    def silver_metadata_dir(self) -> Path:
        return self.drive_root / "silver" / "metadata"

    @property
    def gold_dir(self) -> Path:
        return self.drive_root / "gold"

    @property
    def state_dir(self) -> Path:
        return self.drive_root / "state"

    @property
    def logs_dir(self) -> Path:
        return self.drive_root / "logs"

    @property
    def state_file(self) -> Path:
        return self.state_dir / "processing_state.json"

    # Local workspace
    @property
    def downloads_dir(self) -> Path:
        return self.workspace_root / "downloads"

    @property
    def raw_dir(self) -> Path:
        return self.workspace_root / "raw"

    @property
    def silver_local_dir(self) -> Path:
        return self.workspace_root / "silver"

    @property
    def package_dir(self) -> Path:
        return self.workspace_root / "package"

    @property
    def cache_dir(self) -> Path:
        return self.workspace_root / "cache"

    @property
    def temp_dir(self) -> Path:
        return self.workspace_root / "temp"

    def speaker_raw_dir(self, speaker_id: str) -> Path:
        return self.raw_dir / speaker_id

    def speaker_silver_local_dir(self, speaker_id: str) -> Path:
        return self.silver_local_dir / speaker_id

    def speaker_downloads_dir(self, speaker_id: str) -> Path:
        return self.downloads_dir / speaker_id

    def usable_speakers(self) -> List[str]:
        """Return usable speakers respecting the excluded list."""
        return [s for s in ALL_GRID_SPEAKERS if s not in self.excluded_speakers]
