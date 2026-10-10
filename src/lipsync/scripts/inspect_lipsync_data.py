"""Read-only GRID / AI LipSync dataset structure report.

Works with Google Drive for desktop on Windows or a mounted Google Drive on Colab.
Avoids reading heavy TAR shards or scanning entire CSVs by default.

Run:
  python inspect_lipsync_data.py --root "G:\\...\\20261002_grid_dataset"
  python inspect_lipsync_data.py --root "G:\\...\\20261002_grid_dataset" --out my_report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

DATA_ROOT = Path(
    os.environ["LIPSYNC_DATA_ROOT"]
)

MAX_PREVIEW = 12
META_SAMPLE_LIMIT = 2
MAX_CSV_SAMPLES = 2


def fmt_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(size)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:,.1f} {unit}"
        amount /= 1024
    return str(size)


def sorted_children(folder: Path) -> list[Path]:
    try:
        return sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except (PermissionError, OSError) as exc:
        raise RuntimeError(f"Cannot list {folder.name}: {exc}") from exc


def describe_dir(folder: Path, *, preview: int = MAX_PREVIEW) -> list[str]:
    if not folder.exists():
        return ["  (not found)"]
    if not folder.is_dir():
        return ["  (not a directory)"]
    try:
        children = sorted_children(folder)
    except Exception as exc:
        return [f"  (read error: {type(exc).__name__})"]
    files = [p for p in children if p.is_file()]
    subdirs = [p for p in children if p.is_dir()]
    lines = [f"  directories={len(subdirs)}, files={len(files)}"]
    for p in children[:preview]:
        if p.is_dir():
            lines.append(f"  [DIR]  {p.name}/")
        else:
            try:
                size = fmt_size(p.stat().st_size)
            except OSError:
                size = "size unavailable"
            lines.append(f"  [FILE] {p.name} ({size})")
    if len(children) > preview:
        lines.append(f"  ... and {len(children) - preview} more entries (names omitted)")
    return lines


def summarize_json(path: Path, *, example_sample: bool = False) -> list[str]:
    if not path.is_file():
        return ["  (not found)"]
    try:
        # Reading JSON and CSV metadata is fine; TAR/media is never opened.
        with path.open("r", encoding="utf-8-sig") as fh:
            data: Any = json.load(fh)
    except (OSError, ValueError, UnicodeError) as exc:
        return [f"  (cannot parse JSON: {type(exc).__name__})"]
    if not isinstance(data, dict):
        return [f"  JSON root type: {type(data).__name__}"]

    lines = ["  root keys: " + ", ".join(sorted(data.keys()))]
    for key in ("schema_version", "status", "speaker_id", "strategy", "seed", "partial"):
        value = data.get(key)
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"  {key}: {value}")

    for key in ("summary", "shard", "utterance_counts", "invalid_reason_counts"):
        value = data.get(key)
        if isinstance(value, dict):
            # Print only harmless numeric/boolean properties, never absolute paths.
            numeric = {k: v for k, v in value.items() if isinstance(v, (int, float, bool))}
            lines.append(f"  {key} keys: {', '.join(sorted(value))}")
            if numeric:
                lines.append(f"  {key} numeric fields: {numeric}")

    for key in ("train_speakers", "val_speakers", "test_speakers", "missing_speakers"):
        value = data.get(key)
        if isinstance(value, list):
            lines.append(f"  {key}: count={len(value)}")
            # IDs like s1 / s2 are not private.
            if all(isinstance(v, str) and v.startswith('s') and v[1:].isdigit() for v in value):
                lines.append(f"    ids: {', '.join(value[:40])}")

    if isinstance(data.get("speakers"), dict):
        counts: dict[str, int] = {}
        for item in data["speakers"].values():
            if isinstance(item, dict):
                value = str(item.get("status", "UNKNOWN"))
                counts[value] = counts.get(value, 0) + 1
        lines.append(f"  speakers count: {len(data['speakers'])}; statuses: {counts}")

    samples = data.get("samples")
    if isinstance(samples, list):
        lines.append(f"  samples count: {len(samples)}")
        if example_sample and samples:
            first = next((s for s in samples if isinstance(s, dict)), None)
            if first is not None:
                lines.append(f"  first sample keys: {', '.join(sorted(first.keys()))}")
                safe_keys = ("utterance_id", "valid", "frame_count", "cropped_frame_count", "face_detection_ratio", "audio_sample_rate", "audio_duration_sec", "shard_member_prefix")
                safe_data = {k: first[k] for k in safe_keys if k in first and isinstance(first[k], (str, int, float, bool))}
                if safe_data:
                    lines.append(f"  first sample selected fields: {safe_data}")
    return lines


def summarize_csv(path: Path) -> list[str]:
    if not path.is_file():
        return ["  (not found)"]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames
            lines = ["  columns: " + (", ".join(headers) if headers else "(none)")]
            for i, row in enumerate(reader):
                if i >= MAX_CSV_SAMPLES:
                    break
                # Only report technical IDs, not absolute file paths or private metadata.
                keep = ("split", "speaker_id", "utterance_id", "frame_count", "first_frame_index", "last_frame_index", "audio_sample_rate", "valid")
                sample = {k: row[k] for k in keep if k in row}
                lines.append(f"  sample row {i+1}: {sample}")
            lines.append("  NOTE: total CSV rows not counted (avoids reading full file)")
            return lines
    except (OSError, UnicodeError, csv.Error) as exc:
        return [f"  (cannot parse CSV: {type(exc).__name__})"]


def report(root: Path) -> str:
    rows: list[str] = [
        "AI LipSync / GRID Dataset Structure Report",
        "==========================================",
        "Read-only; no media content read; no files modified; no TAR files opened.",
        "Absolute Google Drive path intentionally omitted from this report.",
        f"Dataset folder: {root.name}",
        "",
    ]

    selected_folders = (
        ("ROOT", root),
        ("BRONZE", root / "bronze"),
        ("SILVER", root / "silver"),
        ("SILVER / shards", root / "silver" / "shards"),
        ("SILVER / metadata", root / "silver" / "metadata"),
        ("GOLD", root / "gold"),
        ("STATE", root / "state"),
        ("LOGS", root / "logs"),
    )

    for name, folder in selected_folders:
        rows.append(f"[{name}]")
        rows.extend(describe_dir(folder))
        rows.append("")

    shards_dir = root / "silver" / "shards"
    if shards_dir.is_dir():
        shards = [p for p in sorted_children(shards_dir) if p.is_file() and p.suffix.lower() == ".tar"]
        rows.append("[SHARDS INVENTORY — NO TAR CONTENT READ]")
        rows.append(f"  TAR files: {len(shards)}")
        total = 0
        for p in shards:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            total += size
        rows.append(f"  Total TAR size (file metadata only): {fmt_size(total)}")
        for p in shards[:40]:
            rows.append(f"  {p.name} ({fmt_size(p.stat().st_size)})")
        if len(shards) > 40:
            rows.append(f"  ... and {len(shards) - 40} more")
        rows.append("")

    metadata_dir = root / "silver" / "metadata"
    if metadata_dir.is_dir():
        metas = [p for p in sorted_children(metadata_dir) if p.is_file() and p.suffix.lower() == ".json"]
        rows.append("[METADATA JSON SCHEMAS — FIRST TWO FILES]")
        rows.append(f"  Metadata JSON files: {len(metas)}")
        for p in metas[:META_SAMPLE_LIMIT]:
            rows.append(f"  File: {p.name}")
            rows.extend(summarize_json(p, example_sample=True))
        rows.append("")

    gold_dir = root / "gold"
    if gold_dir.is_dir():
        rows.append("[GOLD MANIFESTS — CSV HEADERS AND TWO SAMPLE ROWS]")
        manifests = sorted(p for p in gold_dir.glob("*.csv") if p.is_file())
        if not manifests:
            rows.append("  (no CSV manifests found)")
        for p in manifests[:12]:
            rows.append(f"  File: {p.name} ({fmt_size(p.stat().st_size)})")
            rows.extend(summarize_csv(p))
        rows.append("")
        rows.append("[GOLD SPLIT METADATA]")
        rows.extend(summarize_json(gold_dir / "split_metadata.json"))
        rows.append("")

    rows.append("[PIPELINE STATE]")
    rows.extend(summarize_json(root / "state" / "processing_state.json"))
    rows.append("")
    rows.append("[NOTES]")
    rows.append("  No large media files were opened by this script.")
    rows.append("  TAR files were enumerated by filename and file size only.")
    rows.append("  Small metadata/manifest files may be downloaded/cached by Google Drive for desktop.")
    rows.append("  Actual data validity, audio/video contents, and model readiness are NOT verified.")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report GRID data layout without reading TAR media")
    parser.add_argument("--root", default=os.environ.get("LIPSYNC_DATA_ROOT"),
                        help="Path to 20261002_grid_dataset on Google Drive")
    parser.add_argument("--out", default="dataset_pattern_report.txt",
                        help="Local output text report (default: dataset_pattern_report.txt)")
    args = parser.parse_args()

    if not args.root:
        parser.error("Supply --root or set LIPSYNC_DATA_ROOT")

    root = Path(args.root).expanduser()
    if not root.is_dir():
        parser.error(f"Dataset folder does not exist: {root}")
    output = Path(args.out).expanduser().resolve()
    if output == root or root in output.parents:
        parser.error("Please save the report outside Google Drive dataset folder")

    result = report(root)
    output.write_text(result, encoding="utf-8")
    print(f"Report saved: {output}")
    print(f"Report lines: {len(result.splitlines())}")
    print("Send the .txt file here; no TAR/media files are included.")


if __name__ == "__main__":
    main()
