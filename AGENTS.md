# AI Agent Instructions — AI LipSync System

Applies to the entire repository. Optimize for understandable, explicit Python, correctness, and incremental learning, not terse or clever code.

## Module ownership
- `src/data/`: original GRID ingestion/preprocessing pipeline; do not modify when editing read-only loader.
- `src/lipsync/datasets/manifest.py`: parse/validate Gold CSV only; returns `ManifestRecord`. Does not read JPEG/WAV.
- `src/lipsync/datasets/shard_reader.py`: locate Silver TAR, index TAR members, read JPEG/WAV **bytes**, manage handles/LRU/optional disk cache. Does not decode or render media and never extracts TARs.
- `src/lipsync/datasets/grid_dataset.py`: combine manifest and shard reader; decode JPEG/WAV into NumPy arrays and validate media.
- `src/lipsync/datasets/error_exception.py`: single source of shared custom exceptions.
- `src/lipsync/datasets/__init__.py`: intentional public exports with no duplicate imports.
- `src/lipsync/audio/`, `video/`, `models/`, `evaluation/`: separate feature extraction, models, and evaluation.
- `tests/datasets/`: focused tests for individual dataset modules.

## Coding style — mandatory
1. Prefer step-by-step `if`/`for` with descriptive intermediate variables. Avoid dense comprehensions, one-line conditional expressions, overly compact Python idioms, unnecessary lambdas and metaprogramming.
2. Each class and method should have one clear responsibility. Extract reusable private helper methods instead of duplicating logic or nesting complex functions.
3. Use explicit type hints and useful docstrings. Add comments for non-obvious implementation decisions; Vietnamese explanatory comments are welcome. Keep identifiers descriptive English.
4. Do not change established public method signatures, exceptions or return types without justification and tests. Surface errors with meaningful context.
5. Use `pathlib.Path`, context managers, and `finally` as necessary. Always release TAR handles and temporary files.
6. Implement the smallest understandable change first. Do not refactor unrelated modules just for style.

## Imports
- Use relative imports **inside** dataset package: `from .error_exception import DatasetReadError`.
- The project currently uses `from src.lipsync.datasets...` in related tests. Preserve a single import convention per execution; do not mix `src.lipsync` with `lipsync` in a process because they can load separate exception class identities.
- Keep `src/` directory. It is the repository's source tree.

## Data safety
- Manifest `shard_path` is provenance; actual shard path is `<root>/silver/shards/<speaker_id>.tar`.
- Utterance members live under `<speaker_id>/<utterance_id>/` with numbered JPEGs and `audio.wav`.
- Never extract TARs to disk. Respect maximum member byte limits; preserve original frame indices, including gaps. Handle missing or malformed data explicitly.
- Never commit actual GRID media, local caches, credentials, `.env`, or machine-specific dataset absolute paths.

## Tests and delivery
- Write deterministic unit tests with `tmp_path` and synthetic TAR/CSV for every new public method and main failure path.
- Real GRID integration tests use `LIPSYNC_DATA_ROOT`, skip when absent, and must not be described as passed without running them.
- Run `python -m pytest tests/datasets/ -v`; where dependencies exist, run `python -m pytest tests/ -v`.
- Keep edits limited to the requested branch. Do not merge or modify `main` unless explicitly requested.
