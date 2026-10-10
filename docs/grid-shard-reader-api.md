# GridShardReader API

`GridShardReader` receives a `ManifestRecord` from `GoldManifestIndex`, locates `<root>/silver/shards/<speaker_id>.tar`, and reads named members under `<speaker_id>/<utterance_id>/`.

It returns **JPEG/WAV bytes**, not decoded video/audio or rendered frames. Decoding belongs in `GridUtteranceDataset`.

| Method | Result | Reads media bytes? |
| --- | --- | --- |
| `list_frame_indices(record)` | Sorted `list[int]` | No |
| `read_frame(record, index)` | One JPEG `bytes` | Only requested JPEG |
| `read_audio(record)` | WAV `bytes` | Only WAV |
| `read_utterance(record)` | `(dict[int, bytes], bytes)` | All JPEGs and WAV |
| `close()` | `None` | Closes TAR handles |

The frame index is the **original JPEG filename index**, not its position in a sorted list. Missing indices remain missing.

### Example

```python
from src.lipsync.datasets.manifest import GoldManifestIndex
from src.lipsync.datasets.shard_reader import GridShardReader

root = "YOUR_DATASET_ROOT"
manifest = GoldManifestIndex(root, "train")
record = manifest[0]
reader = GridShardReader(root)

try:
    indices = reader.list_frame_indices(record)
    jpeg_bytes = reader.read_frame(record, indices[0])
    wav_bytes = reader.read_audio(record)
    frames, audio = reader.read_utterance(record)
    print(indices[:10], len(jpeg_bytes), len(wav_bytes), len(frames))
finally:
    reader.close()
```

Reader reads TAR members directly without extracting archives. `_get_shard()` indexes member names lazily per speaker and maintains a bounded open-shard LRU cache. `cache_dir` copies requested TARs to local storage using file-size verification only, not hashes. `_read_member()` enforces per-file byte limits and rejects truncated content.

The Gold CSV's `shard_path` is provenance and may point to a stale Windows/Colab location; the speaker ID and dataset root are authoritative. A successful byte read does not prove audio/video synchronization.

Run tests with `python -m pytest tests/datasets/test_shard_reader.py -v`.