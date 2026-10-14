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

## Quality selection and centralized settings (LSPR-45)

Application defaults are declared in `src/lipsync/setting/config.py` as
`LipSyncSettings`; the repository-root `.env` can override them, and real
environment variables override `.env`:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LIPSYNC_DATA_ROOT` | unset | Gold/Silver dataset root |
| `LIPSYNC_FPS` | `25` | FPS used for duration calculations |
| `LIPSYNC_FRAME_WIDTH` | `96` | RGB output width |
| `LIPSYNC_FRAME_HEIGHT` | `96` | RGB output height |
| `LIPSYNC_MIN_FRAME_COUNT` | unset | Optional metadata-only quality filter |
| `LIPSYNC_DURATION_GAP_THRESHOLD_MS` | `80` | Reporting threshold, **not** a training filter |

Leave `LIPSYNC_MIN_FRAME_COUNT` empty to preserve the existing unfiltered
dataset. For an exploratory training filter, set it to a positive integer
such as `50`. That value is illustrative, not an established optimal cutoff.

```python
from src.lipsync.setting.config import LipSyncSettings
from src.lipsync.datasets import GridUtteranceDataset

settings = LipSyncSettings.from_env()
if settings.data_root is None:
    raise RuntimeError("Set LIPSYNC_DATA_ROOT in .env")

dataset = GridUtteranceDataset(
    root=settings.data_root,
    split="train",
    settings=settings,
)
try:
    print("Manifest entries:", len(dataset.index))
    print("Selected entries:", len(dataset))
finally:
    dataset.close()
```

The filter maps public dataset indices to original Gold manifest indices
and is applied **before** opening any Silver TARs. `_validate_frames`
continues to verify raw TAR/manifest integrity after selection. No manifest
records or shard contents are modified. An utterance with 5 valid frames may
pass basic manifest QC but be excluded by the optional training filter.

**Do not reject utterances because their audio and video durations differ by
80 ms.** That is a diagnostic threshold, not an alignment judgment.
Minimum frame count alone does not guarantee enough consecutive frames;
alignment and missing-frame policies belong to later S05 tasks.

The metadata report CLI uses `LIPSYNC_FPS` and
`LIPSYNC_DURATION_GAP_THRESHOLD_MS` from the same settings module:

```powershell
python -m src.lipsync.evaluation.scripts.DurationGapReporter
```

Tests (synthetic/no Drive):
```powershell
python -m pytest tests/datasets/test_dataset_settings.py -v
python -m pytest tests/ -v -ra
```

## Data model organization

Data objects are separated from processing logic:

- `src/lipsync/datasets/models.py`: decoded `GridSample`.
- `src/lipsync/evaluation/models.py`: `DurationRow` and `SummaryRow`.
- `src/lipsync/setting/config.py`: `LipSyncSettings` and `.env` loading.

Existing imports via `datasets.grid_dataset.GridSample` and
`evaluation.duration_report.DurationRow/SummaryRow` remain supported.
This refactor does not change duration-gap math or media decoding.
