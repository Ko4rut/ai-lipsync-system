"""Inspect one real Gold utterance without training or writing media."""
from __future__ import annotations

import argparse

from . import GridUtteranceDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Read one processed GRID utterance")
    parser.add_argument("--root", required=True, help="Dataset root containing gold/ and silver/")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--cache-dir", default=None, help="Optional local SSD TAR cache")
    args = parser.parse_args()
    dataset = GridUtteranceDataset(args.root, args.split, cache_dir=args.cache_dir)
    try:
        sample = dataset[args.index]
        print(f"speaker={sample.speaker_id} utterance={sample.utterance_id}")
        print(f"frames={sample.frames.shape} dtype={sample.frames.dtype}")
        print(f"indices={sample.frame_indices[0]}..{sample.frame_indices[-1]}")
        print(f"audio_samples={len(sample.audio)} rate={sample.sample_rate}Hz")
        print(f"video_span={sample.video_duration_sec:.3f}s audio={sample.audio_duration_sec:.3f}s")
        if abs(sample.duration_gap_sec) > 0.08:
            print("WARNING: video/audio duration gap exceeds 80ms; verify source synchronization.")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
