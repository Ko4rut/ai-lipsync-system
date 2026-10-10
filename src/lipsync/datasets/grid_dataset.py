"""Build decoded NumPy samples from GRID Gold metadata and Silver TAR bytes."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

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

    Frames have RGB uint8 shape (T, H, W, 3).
    Audio is mono float32 with values approximately between -1 and 1.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        cache_dir: str | Path | None = None,
        fps: float = 25.0,
        max_open_shards: int = 2,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")

        self.index = GoldManifestIndex(root=root, split=split)
        self.reader = GridShardReader(
            root=root,
            cache_dir=cache_dir,
            max_open_shards=max_open_shards,
        )
        self.fps = fps

    def __len__(self) -> int:
        """Return the number of utterances in the selected Gold manifest."""
        return len(self.index)

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
        """Decode JPEG bytes into an RGB uint8 NumPy array."""
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

            if expected_shape is None:
                expected_shape = rgb_frame.shape
            elif rgb_frame.shape != expected_shape:
                raise DatasetReadError(
                    f"Variable crop dimensions in {record.utterance_id}"
                )

            decoded_frames.append(rgb_frame)

        return np.stack(decoded_frames, axis=0)

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
        record = self.index[index]

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
