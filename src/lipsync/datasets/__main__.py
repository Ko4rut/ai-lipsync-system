"""Simple demos for GoldManifestIndex, GridShardReader and GridUtteranceDataset."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .grid_dataset import GridSample, GridUtteranceDataset
from .manifest import GoldManifestIndex
from .shard_reader import GridShardReader


def demo_manifest(root: str | Path, split: str = "train", index: int = 0) -> None:
    """Show how one manifest row becomes a ManifestRecord."""
    manifest = GoldManifestIndex(root=root, split=split)
    record = manifest[index]

    print("\n===== GoldManifestIndex =====")
    print(f"Manifest: {manifest.path}")
    print(f"Number of utterances: {len(manifest)}")
    print(f"Returned object type: {type(record).__name__}")
    print(f"Record: {record}")
    print(f"Speaker: {record.speaker_id}")
    print(f"Utterance: {record.utterance_id}")
    print(f"Shard member prefix: {record.shard_member_prefix}")
    print(f"Frames (metadata): {record.frame_count}")
    print(f"Audio sample rate (metadata): {record.audio_sample_rate}")


def demo_shard_reader(
    root: str | Path,
    split: str = "train",
    index: int = 0,
    cache_dir: str | Path | None = None,
) -> None:
    """Show how a ManifestRecord locates bytes in a speaker TAR."""
    manifest = GoldManifestIndex(root=root, split=split)
    record = manifest[index]
    reader = GridShardReader(root=root, cache_dir=cache_dir)

    try:
        indices = reader.list_frame_indices(record)
        first_index = indices[0]
        jpeg = reader.read_frame(record, first_index)
        wav = reader.read_audio(record)
        all_frames, all_audio = reader.read_utterance(record)

        print("\n===== GridShardReader =====")
        print(f"Speaker: {record.speaker_id}")
        print(f"Utterance: {record.utterance_id}")
        print(f"Frame indices (first 10): {indices[:10]}")
        print(f"Frame {first_index}: {len(jpeg)} JPEG bytes")
        print(f"First 12 JPEG bytes: {jpeg[:12]!r}")
        print(f"Audio: {len(wav)} WAV bytes")
        print(f"First 12 WAV bytes: {wav[:12]!r}")
        print(f"read_utterance() returned {len(all_frames)} frames")
        print(f"read_utterance() returned {len(all_audio)} audio bytes")
    finally:
        reader.close()


def show_frame(sample: GridSample, position: int = 0) -> None:
    """Open one decoded frame in a desktop OpenCV window."""
    import cv2

    if position < 0 or position >= len(sample.frame_indices):
        raise ValueError(f"Frame position must be from 0 to {len(sample.frame_indices)-1}")

    if os.name != "nt":
        if not os.getenv("DISPLAY") and not os.getenv("WAYLAND_DISPLAY"):
            print("No desktop display found; cannot open image window.")
            return

    rgb = sample.frames[position]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    height, width = bgr.shape[:2]
    enlarged = cv2.resize(
        bgr,
        (width * 4, height * 4),
        interpolation=cv2.INTER_NEAREST,
    )
    frame_index = sample.frame_indices[position]
    title = f"GRID {sample.speaker_id}/{sample.utterance_id} - frame {frame_index}"

    print(f"Showing original frame index {frame_index}; press a key to close.")
    try:
        cv2.imshow(title, enlarged)
        cv2.waitKey(0)
    except cv2.error as error:
        print(f"Cannot open OpenCV image window: {error}")
    finally:
        cv2.destroyAllWindows()


def play_audio(sample: GridSample) -> None:
    """Play the already-decoded PCM audio (optional sounddevice dependency)."""
    try:
        import sounddevice
    except ImportError:
        print("Install sounddevice for playback: python -m pip install sounddevice")
        return

    print(f"Playing audio: {sample.audio_duration_sec:.3f} seconds")
    try:
        sounddevice.play(sample.audio, samplerate=sample.sample_rate)
        sounddevice.wait()
    except Exception as error:
        print(f"Audio device cannot play sample: {error}")


def demo_grid_dataset(
    root: str | Path,
    split: str = "train",
    index: int = 0,
    preview: str = "frame",
    frame_position: int = 0,
    cache_dir: str | Path | None = None,
) -> None:
    """Show one decoded GridSample; optionally show image and play audio."""
    dataset = GridUtteranceDataset(root=root, split=split, cache_dir=cache_dir)

    try:
        sample = dataset[index]

        print("\n===== GridUtteranceDataset =====")
        print(f"Dataset length: {len(dataset)}")
        print(f"Result type: {type(sample).__name__}")
        print(f"Speaker: {sample.speaker_id}")
        print(f"Utterance: {sample.utterance_id}")
        print(f"Frames shape (T,H,W,3): {sample.frames.shape}")
        print(f"Frames dtype: {sample.frames.dtype}")
        print(f"Frame indices (first 10): {sample.frame_indices[:10]}")
        print(f"First RGB pixel: {sample.frames[0, 0, 0].tolist()}")
        print(f"Audio shape: {sample.audio.shape}")
        print(f"Audio dtype: {sample.audio.dtype}")
        print(f"First 10 audio samples: {sample.audio[:10]}")
        print(f"Sample rate: {sample.sample_rate} Hz")
        print(f"FPS: {sample.fps}")
        print(f"Video span: {sample.video_duration_sec:.3f} s")
        print(f"Audio duration: {sample.audio_duration_sec:.3f} s")
        print(f"Duration gap: {sample.duration_gap_sec:.3f} s")

        if abs(sample.duration_gap_sec) > 0.08:
            print("WARNING: A/V duration gap >80ms; verify real synchronization.")

        if preview == "frame" or preview == "both":
            show_frame(sample, frame_position)

        if preview == "audio" or preview == "both":
            play_audio(sample)
    finally:
        dataset.close()


def main() -> None:
    """Select a single demo to run."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    project_root = Path(__file__).resolve().parents[3]
    if load_dotenv is not None:
        load_dotenv(project_root / ".env")

    parser = argparse.ArgumentParser(description="Explore three GRID dataset classes")
    parser.add_argument("--root", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--demo", choices=("manifest", "shard", "dataset"), default="dataset")
    parser.add_argument("--preview", choices=("none", "frame", "audio", "both"), default="frame")
    parser.add_argument("--frame-position", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    dataset_root = args.root
    if not dataset_root:
        dataset_root = os.getenv("LIPSYNC_DATA_ROOT")

    if not dataset_root:
        parser.error("Set LIPSYNC_DATA_ROOT in .env or pass --root")

    if args.demo == "manifest":
        demo_manifest(dataset_root, args.split, args.index)
    elif args.demo == "shard":
        demo_shard_reader(dataset_root, args.split, args.index, args.cache_dir)
    else:
        demo_grid_dataset(
            root=dataset_root,
            split=args.split,
            index=args.index,
            preview=args.preview,
            frame_position=args.frame_position,
            cache_dir=args.cache_dir,
        )


if __name__ == "__main__":
    main()
