"""Input extraction, normalisation, audio/video pairing and preprocessing.

Responsibilities:
- Extract speaker video (ZIP) and audio (TAR or ZIP) archives
- Normalise arbitrary archive layouts into a canonical interface
- Build utterance index (paired / missing / duplicate)
- Decode video frames → face detection → JPEG crops (original frame indices)
- Convert audio to 16 kHz mono WAV via ffmpeg
- Bootstrap Wav2Lip-compatible S3FD face detector when needed
"""

from __future__ import annotations

import enum
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    FACE_DETECTION_BATCH_SIZE,
    JPEG_QUALITY,
    ZENODO_AUDIO_ARCHIVE_FILENAME,
    PipelineConfig,
)
from .exceptions import (
    ArchiveError,
    FaceDetectionSetupError,
    PreprocessingError,
    SourceLayoutError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utterance pairing
# ---------------------------------------------------------------------------

class UtterancePairingStatus(str, enum.Enum):
    PAIRED = "PAIRED"
    MISSING_AUDIO = "MISSING_AUDIO"
    MISSING_VIDEO = "MISSING_VIDEO"
    DUPLICATE_AUDIO = "DUPLICATE_AUDIO"
    DUPLICATE_VIDEO = "DUPLICATE_VIDEO"


@dataclass
class UtteranceCandidate:
    """Represents one potential utterance for a speaker."""
    utterance_id: str
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    pairing_status: UtterancePairingStatus = UtterancePairingStatus.MISSING_AUDIO


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------

def _extract_zip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"Cannot extract ZIP {archive}: {exc}") from exc
    logger.debug("Extracted %s -> %s", archive, dest)


def _extract_tar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    except (tarfile.TarError, EOFError) as exc:
        raise ArchiveError(f"Cannot extract TAR {archive}: {exc}") from exc
    logger.debug("Extracted %s -> %s", archive, dest)


def _extract_archive(archive: Path, dest: Path) -> None:
    suf = archive.suffix.lower()
    if suf == ".zip":
        _extract_zip(archive, dest)
    elif suf in (".tar", ".tgz", ".gz", ".bz2"):
        _extract_tar(archive, dest)
    else:
        # Try magic bytes
        with archive.open("rb") as fh:
            magic = fh.read(4)
        if magic[:2] == b"PK":
            _extract_zip(archive, dest)
        else:
            _extract_tar(archive, dest)


# ---------------------------------------------------------------------------
# Layout normalisation: find MPG video files
# ---------------------------------------------------------------------------

def _find_mpg_files(root: Path) -> Dict[str, Path]:
    """Walk *root* and return ``{utterance_id: path}`` for all .mpg files."""
    result: Dict[str, Path] = {}
    for p in root.rglob("*.mpg"):
        uid = p.stem.lower()
        if uid in result:
            logger.warning("Duplicate video for utterance '%s': %s vs %s", uid, result[uid], p)
        result[uid] = p
    return result


def _find_wav_files(root: Path) -> Dict[str, Path]:
    """Walk *root* and return ``{utterance_id: path}`` for all .wav files."""
    result: Dict[str, Path] = {}
    for p in root.rglob("*.wav"):
        uid = p.stem.lower()
        if uid in result:
            logger.warning("Duplicate audio for utterance '%s': %s vs %s", uid, result[uid], p)
        result[uid] = p
    return result


# ---------------------------------------------------------------------------
# Public: extract and build utterance index
# ---------------------------------------------------------------------------

def extract_and_index(
    speaker_id: str,
    video_archive: Path,
    audio_archive: Path,
    raw_dir: Path,
    *,
    is_zenodo_audio: bool = False,
    config: PipelineConfig,
) -> List[UtteranceCandidate]:
    """Extract archives and return a list of :class:`UtteranceCandidate`.

    Parameters
    ----------
    is_zenodo_audio:
        True when *audio_archive* is the combined Zenodo ``audio_25k.zip``.
        In that case only the target speaker's members are extracted.
    """
    video_root = raw_dir / "video"
    audio_root = raw_dir / "audio"

    # --- Extract video ---
    logger.info("[%s] Extracting video archive: %s", speaker_id, video_archive)
    _extract_archive(video_archive, video_root)

    # --- Extract audio ---
    if is_zenodo_audio:
        logger.info(
            "[%s] Extracting speaker audio from combined Zenodo archive.", speaker_id
        )
        _extract_zenodo_audio_for_speaker(audio_archive, speaker_id, audio_root)
    else:
        logger.info("[%s] Extracting audio archive: %s", speaker_id, audio_archive)
        _extract_archive(audio_archive, audio_root)

    # --- Find files ---
    video_map = _find_mpg_files(video_root)
    audio_map = _find_wav_files(audio_root)

    if not video_map:
        raise SourceLayoutError(
            f"[{speaker_id}] No .mpg files found after extracting {video_archive}. "
            "Check archive structure."
        )
    if not audio_map:
        raise SourceLayoutError(
            f"[{speaker_id}] No .wav files found after extracting audio archive. "
            "Check archive structure."
        )

    all_uids = sorted(set(video_map) | set(audio_map))
    candidates: List[UtteranceCandidate] = []

    for uid in all_uids:
        has_v = uid in video_map
        has_a = uid in audio_map
        if has_v and has_a:
            status = UtterancePairingStatus.PAIRED
        elif has_v:
            status = UtterancePairingStatus.MISSING_AUDIO
        else:
            status = UtterancePairingStatus.MISSING_VIDEO

        candidates.append(
            UtteranceCandidate(
                utterance_id=uid,
                video_path=video_map.get(uid),
                audio_path=audio_map.get(uid),
                pairing_status=status,
            )
        )

    paired = sum(1 for c in candidates if c.pairing_status == UtterancePairingStatus.PAIRED)
    logger.info(
        "[%s] Utterance index: %d total, %d paired, %d missing audio, %d missing video",
        speaker_id, len(candidates), paired,
        sum(1 for c in candidates if c.pairing_status == UtterancePairingStatus.MISSING_AUDIO),
        sum(1 for c in candidates if c.pairing_status == UtterancePairingStatus.MISSING_VIDEO),
    )
    return candidates


def _extract_zenodo_audio_for_speaker(
    audio_25k_zip: Path, speaker_id: str, dest: Path
) -> None:
    """Extract only the *speaker_id* members from ``audio_25k.zip``."""
    dest.mkdir(parents=True, exist_ok=True)
    prefix = speaker_id + "/"  # e.g. "s1/"
    try:
        with zipfile.ZipFile(audio_25k_zip) as zf:
            members = [m for m in zf.namelist() if m.startswith(prefix)]
            if not members:
                # Some Zenodo archives may not use a subdirectory prefix.
                # Try extracting all and filtering.
                logger.warning(
                    "No members with prefix '%s' in Zenodo audio archive; "
                    "extracting all and filtering.", prefix
                )
                zf.extractall(dest)
            else:
                for member in members:
                    zf.extract(member, dest)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"Cannot open Zenodo audio archive {audio_25k_zip}: {exc}") from exc


# ---------------------------------------------------------------------------
# Face detection bootstrap
# ---------------------------------------------------------------------------

def ensure_face_detection_available() -> None:
    """Check that the supported ``face_alignment`` S3FD API is available."""
    try:
        from face_alignment.detection.sfd import FaceDetector  # noqa: F401
    except ImportError as exc:
        raise FaceDetectionSetupError(
            "The supported S3FD detector is not installed. "
            "Run: pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc


class _SFDDetectorAdapter:
    """Expose the batch API expected by the preprocessing loop."""

    def __init__(self, detector) -> None:
        self._detector = detector

    def get_detections_for_batch(self, images: np.ndarray):
        import torch

        # ``process_video_utterance`` supplies an RGB uint8 NumPy batch in
        # NHWC layout.  face-alignment 1.4.1's SFD detector expects a PyTorch
        # tensor in NCHW layout; passing NumPy through directly makes every
        # batch fail at ``img_batch.size(0)``.
        tensor = torch.from_numpy(images.transpose(0, 3, 1, 2).copy())
        detections = self._detector.detect_from_batch(tensor)

        results = []
        for faces in detections:
            if len(faces) == 0:
                results.append(None)
                continue
            results.append(np.clip(faces[0], 0, None)[:4])
        return results


def _build_face_detector(config: PipelineConfig):  # type: ignore[return]
    """Instantiate and return the face detector.

    Returns ``(detector, device_str)``.
    Raises :class:`.FaceDetectionSetupError` if setup fails.
    """
    from face_alignment.detection.sfd import FaceDetector

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    try:
        detector = _SFDDetectorAdapter(
            FaceDetector(device=device, verbose=False)
        )
    except Exception as exc:
        raise FaceDetectionSetupError(
            f"Could not initialise face detector: {exc}"
        ) from exc

    logger.info("Face detector ready (device=%s)", device)
    return detector, device


# ---------------------------------------------------------------------------
# Video preprocessing: decode → detect → crop → save JPEGs
# ---------------------------------------------------------------------------

@dataclass
class VideoProcessingResult:
    total_frames: int = 0
    detected_frames: int = 0
    failed_frames: int = 0
    source_fps: float = 0.0
    source_width: int = 0
    source_height: int = 0
    decodable: bool = False
    invalid_reason: Optional[str] = None
    #: Maps source_frame_index -> saved JPEG path
    frame_paths: Dict[int, Path] = field(default_factory=dict)

    @property
    def detection_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.detected_frames / self.total_frames


def process_video_utterance(
    utterance_id: str,
    video_path: Path,
    output_dir: Path,
    detector,
    *,
    config: PipelineConfig,
) -> VideoProcessingResult:
    """Decode *video_path*, detect faces, and save per-frame JPEG crops.

    Frame filenames use the ORIGINAL source frame index (0-based),
    NOT a dense renumbering after face-detection skips.
    """
    result = VideoProcessingResult()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.invalid_reason = "VIDEO_DECODE_FAILED"
        return result

    try:
        result.source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        result.source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result.source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate FPS
        expected = config.expected_fps
        tol = expected * config.fps_tolerance_fraction
        if not (expected - tol <= result.source_fps <= expected + tol):
            result.invalid_reason = "INVALID_FPS"
            return result

        # Read all frames (per-utterance — typically ~75 frames for GRID)
        frames: List[Tuple[int, np.ndarray]] = []  # (frame_index, bgr)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((frame_idx, frame))
            frame_idx += 1

        result.total_frames = len(frames)
        result.decodable = True

        if result.total_frames == 0:
            result.invalid_reason = "NO_FRAMES"
            return result

        # Batch face detection
        batch_size = config.face_detection_batch_size
        for batch_start in range(0, len(frames), batch_size):
            batch = frames[batch_start: batch_start + batch_size]
            batch_indices = [f[0] for f in batch]
            batch_frames_rgb = [cv2.cvtColor(f[1], cv2.COLOR_BGR2RGB) for f in batch]

            try:
                predictions = detector.get_detections_for_batch(
                    np.array(batch_frames_rgb)
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Face detection batch failed: %s", utterance_id, exc
                )
                result.failed_frames += len(batch)
                continue

            for src_idx, frame_bgr, pred in zip(batch_indices, [f[1] for f in batch], predictions):
                if pred is None:
                    result.failed_frames += 1
                    continue

                # pred: [[x1, y1, x2, y2, confidence], ...]
                # Take the first (highest confidence) detection.
                try:
                    x1, y1, x2, y2 = [int(v) for v in pred[:4]]
                except (TypeError, ValueError, IndexError):
                    result.failed_frames += 1
                    continue

                # Clamp to frame bounds
                h, w = frame_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    result.failed_frames += 1
                    continue

                crop = frame_bgr[y1:y2, x1:x2]
                jpeg_path = output_dir / f"{src_idx}.jpg"
                ok_write = cv2.imwrite(
                    str(jpeg_path),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
                )
                if ok_write:
                    result.detected_frames += 1
                    result.frame_paths[src_idx] = jpeg_path
                else:
                    logger.warning("[%s] Failed to write JPEG for frame %d", utterance_id, src_idx)
                    result.failed_frames += 1

    finally:
        cap.release()

    return result


# ---------------------------------------------------------------------------
# Audio preprocessing: ffmpeg conversion to 16 kHz mono WAV
# ---------------------------------------------------------------------------

@dataclass
class AudioProcessingResult:
    success: bool = False
    invalid_reason: Optional[str] = None
    original_sample_rate: Optional[int] = None
    output_sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_sec: Optional[float] = None
    output_path: Optional[Path] = None


def process_audio_utterance(
    utterance_id: str,
    audio_path: Path,
    output_path: Path,
    *,
    config: PipelineConfig,
) -> AudioProcessingResult:
    """Convert *audio_path* to 16 kHz mono WAV at *output_path* via ffmpeg."""
    result = AudioProcessingResult()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe original
    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        lines = probe.stdout.strip().splitlines()
        if len(lines) >= 1:
            result.original_sample_rate = int(lines[0]) if lines[0].isdigit() else None
        if len(lines) >= 2:
            result.channels = int(lines[1]) if lines[1].isdigit() else None
        if len(lines) >= 3:
            try:
                result.duration_sec = float(lines[2])
            except ValueError:
                pass
    except Exception as exc:
        logger.debug("[%s] ffprobe failed: %s", utterance_id, exc)

    # Convert
    # Keep ``.wav`` as the final suffix so ffmpeg can infer the output
    # container.  ``audio.wav.tmp`` ends in ``.tmp`` and makes ffmpeg fail
    # with "Unable to find a suitable output format".
    tmp_out = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ac", str(config.target_audio_channels),
            "-ar", str(config.target_audio_sample_rate),
            "-acodec", "pcm_s16le",
            str(tmp_out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            result.invalid_reason = "AUDIO_CONVERSION_FAILED"
            logger.warning("[%s] ffmpeg conversion failed: %s", utterance_id, proc.stderr[-500:])
            return result

        if not tmp_out.exists() or tmp_out.stat().st_size == 0:
            result.invalid_reason = "EMPTY_AUDIO"
            return result

        tmp_out.rename(output_path)

        # Validate output
        probe_cmd2 = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(output_path),
        ]
        probe2 = subprocess.run(probe_cmd2, capture_output=True, text=True, timeout=30)
        lines2 = probe2.stdout.strip().splitlines()
        out_sr = int(lines2[0]) if lines2 and lines2[0].isdigit() else None
        out_ch = int(lines2[1]) if len(lines2) > 1 and lines2[1].isdigit() else None
        out_dur: Optional[float] = None
        if len(lines2) > 2:
            try:
                out_dur = float(lines2[2])
            except ValueError:
                pass

        result.output_sample_rate = out_sr
        result.channels = out_ch
        result.duration_sec = out_dur
        result.output_path = output_path

        if out_sr != config.target_audio_sample_rate:
            result.invalid_reason = "INVALID_OUTPUT_AUDIO_RATE"
            return result

        if out_dur is not None and out_dur < config.min_audio_duration_sec:
            result.invalid_reason = "EMPTY_AUDIO"
            return result

        result.success = True
        return result

    except subprocess.TimeoutExpired:
        result.invalid_reason = "AUDIO_CONVERSION_FAILED"
        return result
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
