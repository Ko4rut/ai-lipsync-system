"""Decode processed GRID samples without coupling the reader to any model."""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .manifest import GoldManifestIndex, ManifestRecord
from .shard_reader import DatasetReadError, GridShardReader


@dataclass
class GridSample:
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
        return self.video_duration_sec - self.audio_duration_sec


class GridUtteranceDataset:
    """Map-style reader; __getitem__ returns one utterance, not a train window.

    RGB images: [T,H,W,3] uint8 (all crops in an utterance must match size).
    Mono audio: float32 in approximately [-1,1].
    """

    def __init__(
        self, root: str | Path, split: str,
        *, cache_dir: str | Path | None = None, fps: float = 25.0,
        max_open_shards: int = 2,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.index = GoldManifestIndex(root, split)
        self.reader = GridShardReader(root, cache_dir=cache_dir, max_open_shards=max_open_shards)
        self.fps = fps

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> GridSample:
        record: ManifestRecord = self.index[index]
        raw_frames, wav = self.reader.read_utterance(record)
        indices = tuple(sorted(raw_frames))
        if len(indices) != record.frame_count:
            raise DatasetReadError(
                f"{record.speaker_id}/{record.utterance_id}: "
                f"expected {record.frame_count} frames, found {len(indices)}"
            )
        if indices[0] != record.first_frame_index or indices[-1] != record.last_frame_index:
            raise DatasetReadError(f"Manifest frame bounds differ from TAR: {record.utterance_id}")

        frames = []
        for frame_index in indices:
            array = np.frombuffer(raw_frames[frame_index], np.uint8)
            bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if bgr is None:
                raise DatasetReadError(f"Cannot decode frame {frame_index} of {record.utterance_id}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        sizes = {f.shape for f in frames}
        if len(sizes) != 1:
            raise DatasetReadError(f"Variable crop dimensions in {record.utterance_id}")
        with wave.open(io.BytesIO(wav), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            if handle.getcomptype() != "NONE" or channels != 1 or width != 2:
                raise DatasetReadError(f"Expected mono PCM16 WAV: {record.utterance_id}")
            signal = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        if rate != record.audio_sample_rate:
            raise DatasetReadError(f"Audio rate differs from Gold manifest: {record.utterance_id}")
        if signal.size == 0:
            raise DatasetReadError(f"Empty audio: {record.utterance_id}")
        audio = signal.astype(np.float32) / 32768.0
        return GridSample(
            record.speaker_id, record.utterance_id, np.stack(frames),
            indices, audio, rate, self.fps,
            len(audio) / rate, (indices[-1] - indices[0] + 1) / self.fps,
        )

    def close(self) -> None:
        self.reader.close()
