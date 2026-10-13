# GRID raw dataset loader

Read-only access to preprocessed GRID Silver TARs and Gold CSV manifests.
**This is not yet a Wav2Lip training window/mel adapter.** No media is committed.

## Expected layout

    DATASET_ROOT/
      gold/train_manifest.csv
      gold/val_manifest.csv
      gold/test_manifest.csv
      silver/shards/s1.tar ... s34.tar (except s21.tar)

The CSV \`shard_path\` is retained only as provenance; it may reference an old
Windows mount or Colab Drive location. The reader resolves shards using
\`DATASET_ROOT/silver/shards/<speaker_id>.tar\` and validates member prefixes.

## Smoke test

Install Python >=3.10 and \`pip install numpy opencv-python pytest\`.
From repo root, run:

    PYTHONPATH=src python -m lipsync.datasets --root "/content/drive/MyDrive/20261002_grid_dataset" --split train --index 0 --cache-dir /content/grid_shard_cache

PowerShell:

    $env:PYTHONPATH="src"
    python -m lipsync.datasets --root "G:\path\to\20261002_grid_dataset" --split train --index 0

\`--cache-dir\` copies only the requested speaker TAR to a local directory
before reading. Use a local SSD in Colab rather than Drive for faster repeated
access. The cache verifies byte size (not checksums), and assumes immutable
source TARs; remove stale caches after regenerating Silver data.

Programmatic usage:

    from lipsync.datasets import GridUtteranceDataset
    dataset = GridUtteranceDataset(root, "train", cache_dir="/content/grid_shard_cache")
    sample = dataset[0]
    print(sample.frames.shape, sample.frame_indices, sample.audio.shape)
    dataset.close()

\`GridSample.frames\` is RGB uint8 [T,H,W,3]. \`audio\` is mono
float32 PCM normalized to [-1,1], and \`frame_indices\` preserves original
source indices even if face detection skipped a frame.

## Critical limitations / follow-up

- The loader checks metadata schema, member existence, decodability, frame
  count/bounds, and audio sample rate. It does **not** validate every shard
  SHA-256, source ground-truth synchronization, or full dataset QC.
- \`video_duration_sec\` measures the source frame-index span; it is not the
  number of successfully detected frames divided by FPS.
- GRID metadata has been observed with ~75 frames at 25 FPS and ~1.19 s audio.
  Investigate temporal offsets and alignment before building Wav2Lip windows.
  The CLI flags duration differences over 80ms without altering audio.
- Training with multiple workers may open separate TAR indices/caches per
  process. Stage shards to SSD before large training runs.
- \`Wav2LipWindowDataset\`, mel features, video rendering, and trained model
  inference are intentionally deferred until the raw loader is validated.

## Tests

    PYTHONPATH=src python -m pytest tests/test_grid_dataset_loader.py -v

Synthetic fixtures only; a real-media smoke test must be executed in the
dataset runtime before claiming the end-to-end real dataset is verified.
