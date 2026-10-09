"""Utility functions shared across pipeline modules.

Kept thin on purpose: only genuinely reusable helpers belong here.
Domain-specific logic lives in the relevant domain module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import HASH_CHUNK_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Safe JSON writes (atomic via temp-file + rename)
# ---------------------------------------------------------------------------

def atomic_json_write(path: Path, data: Any, *, indent: int = 2) -> None:
    """Write *data* to *path* atomically using a temporary file + rename.

    This prevents leaving malformed JSON at *path* if the process is
    interrupted mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".tmp_", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup of the temp file.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: Path) -> None:
    """Create all *paths* and their parents if they do not exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def remove_dir_if_exists(path: Path) -> None:
    """Remove *path* recursively if it exists; silently skip otherwise."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        logger.debug("Removed directory: %s", path)


def remove_file_if_exists(path: Path) -> None:
    """Remove *path* if it exists; silently skip otherwise."""
    if path.exists():
        path.unlink()
        logger.debug("Removed file: %s", path)


# ---------------------------------------------------------------------------
# Consecutive-window detection
# ---------------------------------------------------------------------------

def max_consecutive_run(indices: list[int]) -> int:
    """Return the length of the longest consecutive run in *indices*.

    Example::

        max_consecutive_run([0, 1, 3, 4, 5, 6]) -> 4   (run: 3,4,5,6)
        max_consecutive_run([])                  -> 0
    """
    if not indices:
        return 0
    sorted_idx = sorted(indices)
    best = 1
    current = 1
    for prev, curr in zip(sorted_idx, sorted_idx[1:]):
        if curr == prev + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure root logger with a console handler and optional file handler.

    Safe to call multiple times; additional calls are idempotent with respect
    to duplicate handler prevention.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) and h.stream.name in ("<stdout>", "<stderr>")
                for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(ch)

    # File handler
    if log_file is not None:
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file
                   for h in root.handlers):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            root.addHandler(fh)


# ---------------------------------------------------------------------------
# Human-readable size
# ---------------------------------------------------------------------------

def human_bytes(n: int) -> str:
    """Return *n* bytes as a human-readable string (e.g. '1.23 GiB')."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.2f} PiB"
