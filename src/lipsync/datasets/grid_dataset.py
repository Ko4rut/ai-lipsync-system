"""Build decoded NumPy samples from GRID Gold metadata and Silver TAR bytes."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

from src.lipsync.config import LipSyncSettings

import cv2
import numpy as np

from .error_exception import DatasetReadError
from .manifest import GoldManifestIndex, ManifestRecord
from .shard_reader import GridShardReader


@dataclass
class GridSample:
    """A decoded GRID utterance ready for audio/video preprocessing."""

    speaker_id: str
    utterance_id: str
    frames: np.ndarray
    frame_indices: tuple[int, ...]
    audio: np.ndarray
    sample_rate: int
    fps: float
    audio_duration_sec: float
    video_duration_sec: float

    @property
    def duration_gap_sec(self) -> float:
        """Difference between frame-index video span and WAV audio duration."""
        return self.video_duration_sec - self.audio_duration_sec


class GridUtteranceDataset:
    """Load and decode a complete utterance, not a Wav2Lip training window.

    Frames have RGB uint8 shape (T, H, W, 3). Each decoded frame is resized
    proportionally and padded to ``frame_size`` without changing the TAR data.
    Audio is mono float32 with values approximately between -1 and 1.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        cache_dir: str | Path | None = None,
        fps: float | None = None,
        max_open_shards: int = 2,
        frame_size: tuple[int, int] | None = None,
        min_frame_count: int | None = None,
        settings: LipSyncSettings | None = None,
    ) -> None:
        # Explicit arguments override application settings.
        if settings is None:
            settings = LipSyncSettings.from_env()
        if fps is None:
            fps = settings.fps
        if frame_size is None:
            frame_size = settings.frame_size
        if min_frame_count is None:
            min_frame_count = settings.min_frame_count

        if min_frame_count is not None and (type(min_frame_count) is not int or min_frame_count < 1):
            raise ValueError("min_frame_count must be a positive integer")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if len(frame_size) != 2:
            raise ValueError("frame_size must contain width and height")

        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame_size width and height must be positive")

        self.index = GoldManifestIndex(root=root, split=split)
        self.min_frame_count = min_frame_count
        self._eligible_indices: list[int] = []
        for manifest_index, record in enumerate(self.index):
            if not self._is_eligible_record(record):
                continue
            self._eligible_indices.append(manifest_index)
        self.reader = GridShardReader(
            root=root,
            cache_dir=cache_dir,
            max_open_shards=max_open_shards,
        )
        self.fps = fps
        self.frame_size = frame_size

    def _is_eligible_record(self, record: ManifestRecord) -> bool:
        """Quality filter on Gold metadata, before opening Silver TARs."""
        if self.min_frame_count is not None:
            if record.frame_count < self.min_frame_count:
                return False
        return True

    def __len__(self) -> int:
        """Return the number of eligible utterances."""
        return len(self._eligible_indices)

    def _validate_frames(
        self,
        record: ManifestRecord,
        raw_frames: dict[int, bytes],
    ) -> tuple[int, ...]:
        """Check that JPEG frame indices match the Gold manifest."""
        frame_indices = list(raw_frames.keys())
        frame_indices.sort()

        actual_count = len(frame_indices)
        if actual_count != record.frame_count:
            raise DatasetReadError(
                f"{record.speaker_id}/{record.utterance_id}: "
                f"expected {record.frame_count} frames, found {actual_count}"
            )

        first_index = frame_indices[0]
        last_index = frame_indices[-1]

        if first_index != record.first_frame_index:
            raise DatasetReadError(
                f"Manifest first frame differs from TAR: {record.utterance_id}"
            )

        if last_index != record.last_frame_index:
            raise DatasetReadError(
                f"Manifest last frame differs from TAR: {record.utterance_id}"
            )

        return tuple(frame_indices)

    def _decode_frames(
        self,
        record: ManifestRecord,
        raw_frames: dict[int, bytes],
        frame_indices: tuple[int, ...],
    ) -> np.ndarray:
        """Decode JPEG bytes into a fixed-size RGB uint8 NumPy array."""
        decoded_frames: list[np.ndarray] = []
        expected_shape: tuple[int, ...] | None = None

        for frame_index in frame_indices:
            jpeg_bytes = raw_frames[frame_index]
            encoded_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)

            try:
                bgr_frame = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
            except cv2.error as error:
                raise DatasetReadError(
                    f"Cannot decode frame {frame_index} of {record.utterance_id}: {error}"
                ) from error

            if bgr_frame is None:
                raise DatasetReadError(
                    f"Cannot decode frame {frame_index} of {record.utterance_id}"
                )

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            normalized_frame = self._resize_with_padding(rgb_frame)

            if expected_shape is None:
                expected_shape = normalized_frame.shape
            elif normalized_frame.shape != expected_shape:
                raise DatasetReadError(
                    f"Cannot normalize frame dimensions in {record.utterance_id}"
                )

            decoded_frames.append(normalized_frame)

        return np.stack(decoded_frames, axis=0)

    def _resize_with_padding(self, frame: np.ndarray) -> np.ndarray:
        """Resize one frame proportionally and center it on a black canvas."""
        target_width, target_height = self.frame_size
        source_height, source_width = frame.shape[:2]

        width_scale = target_width / source_width
        height_scale = target_height / source_height
        resize_scale = min(width_scale, height_scale)

        resized_width = max(1, round(source_width * resize_scale))
        resized_height = max(1, round(source_height * resize_scale))

        interpolation = cv2.INTER_LINEAR
        if resize_scale < 1.0:
            interpolation = cv2.INTER_AREA

        resized_frame = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=interpolation,
        )

        output_frame = np.zeros(
            (target_height, target_width, 3),
            dtype=np.uint8,
        )
        offset_x = (target_width - resized_width) // 2
        offset_y = (target_height - resized_height) // 2
        output_frame[
            offset_y:offset_y + resized_height,
            offset_x:offset_x + resized_width,
        ] = resized_frame

        return output_frame

    def _decode_audio(
        self,
        record: ManifestRecord,
        wav_bytes: bytes,
    ) -> tuple[np.ndarray, int]:
        """Decode mono PCM16 WAV bytes into float32 samples and sample rate."""
        try:
            with wave.open(io.BytesIO(wav_bytes), mode="rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channel_count = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                compression_type = wav_file.getcomptype()

                if compression_type != "NONE" or channel_count != 1 or sample_width != 2:
                    raise DatasetReadError(
                        f"Expected mono PCM16 WAV: {record.utterance_id}"
                    )

                total_frames = wav_file.getnframes()
                pcm_bytes = wav_file.readframes(total_frames)
        except (wave.Error, EOFError, OSError) as error:
            raise DatasetReadError(
                f"Cannot decode WAV of {record.utterance_id}: {error}"
            ) from error

        if sample_rate != record.audio_sample_rate:
            raise DatasetReadError(
                f"Audio rate differs from Gold manifest: {record.utterance_id}"
            )

        if len(pcm_bytes) == 0:
            raise DatasetReadError(f"Empty audio: {record.utterance_id}")

        if len(pcm_bytes) % 2 != 0:
            raise DatasetReadError(f"Invalid PCM16 byte count: {record.utterance_id}")

        pcm_samples = np.frombuffer(pcm_bytes, dtype="<i2")
        float_samples = pcm_samples.astype(np.float32)
        audio = float_samples / 32768.0

        return audio, sample_rate

    def __getitem__(self, index: int) -> GridSample:
        """Use one manifest row to load, validate, and decode an utterance."""
        # STEP 1: Get metadata from the Gold manifest.
        manifest_index = self._eligible_indices[index]
        record = self.index[manifest_index]

        # STEP 2: Read JPEG and WAV bytes from the Silver TAR.
        raw_frames, wav_bytes = self.reader.read_utterance(record)

        # STEP 3: Validate frame indices before decoding the images.
        frame_indices = self._validate_frames(record, raw_frames)

        # STEP 4: Decode JPEG and WAV data into NumPy arrays.
        frames = self._decode_frames(record, raw_frames, frame_indices)
        audio, sample_rate = self._decode_audio(record, wav_bytes)

        # STEP 5: Compute actual audio duration and frame-index video span.
        audio_duration = len(audio) / sample_rate
        first_index = frame_indices[0]
        last_index = frame_indices[-1]
        video_duration = (last_index - first_index + 1) / self.fps

        # STEP 6: Return a simple object containing one decoded sample.
        return GridSample(
            speaker_id=record.speaker_id,
            utterance_id=record.utterance_id,
            frames=frames,
            frame_indices=frame_indices,
            audio=audio,
            sample_rate=sample_rate,
            fps=self.fps,
            audio_duration_sec=audio_duration,
            video_duration_sec=video_duration,
        )

    def close(self) -> None:
        """Release open TAR files held by the reader."""
        self.reader.close()
