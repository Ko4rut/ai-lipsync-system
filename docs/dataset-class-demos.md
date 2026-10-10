# Dataset class demos (Windows / PowerShell)

The source code lives in `src/lipsync/datasets/` and the project can be invoked from the repository root.

## Dependencies

```powershell
python -m pip install pytest numpy opencv-python python-dotenv
# Only for listening to audio:
python -m pip install sounddevice
```

Configure `LIPSYNC_DATA_ROOT` in a project-root `.env` file, or pass `--root "PATH_TO_GRID_DATASET"`. The root directory must contain `gold/` and `silver/shards/`. Do not commit `.env`.

## Three independent examples in __main__.py

```powershell
# 1. Print CSV metadata and one ManifestRecord.
python -m src.lipsync.datasets --demo manifest --split train --index 0

# 2. Print a JPEG/WAV byte sample and TAR frame indices.
python -m src.lipsync.datasets --demo shard --split train --index 0

# 3. Decode frames and audio; print shape/type and open first frame.
python -m src.lipsync.datasets --demo dataset --split train --index 0

# 4. Play the utterance audio after decoding.
python -m src.lipsync.datasets --demo dataset --preview audio

# 5. Display first frame and then play audio.
python -m src.lipsync.datasets --demo dataset --preview both

# 6. Choose the 11th decoded frame (zero-based position = 10).
python -m src.lipsync.datasets --demo dataset --preview frame --frame-position 10

# Only print sample information (useful on Colab/headless machines).
python -m src.lipsync.datasets --demo dataset --preview none
```

The `--demo` choice calls one of `demo_manifest`, `demo_shard_reader`, or `demo_grid_dataset`. `main()` only selects the example and reads command-line arguments.

`demo_grid_dataset` creates one `GridSample`, prints its metadata, array shapes/dtypes, the first RGB pixel, and the first 10 waveform values. It **does not print full arrays**. The OpenCV window displays a face crop (not an original full video frame); press any key to close it. Playback uses `sounddevice` and requires an audio output device. Audio/image previews are separate demonstration functions and not responsibilities of `GridUtteranceDataset`.

The `GridUtteranceDataset` outputs RGB uint8 frames of shape `(T, H, W, 3)` and mono float32 audio samples of shape `(N,)`. It validates frame count/index bounds, image dimensions, WAV format, and audio sample rate. This is not yet time-aligned training data or a Wav2Lip adapter.

## Tests

```powershell
python -m pytest tests/datasets/test_grid_dataset.py -v
python -m pytest tests/ -v
```

Tests use synthetic data or test doubles. Real GRID preview and playback must be checked on the local Windows machine; no real GRID media is included in the repository.
