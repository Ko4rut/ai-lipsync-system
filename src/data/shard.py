"""Silver shard packaging and safe upload to Google Drive.

Packaging contract:
- One uncompressed .tar per speaker (valid utterances only)
- Created locally first, then uploaded atomically to Drive
- SHA-256 verified after upload
- Existing valid DONE shards are never overwritten without --force
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import SCHEMA_VERSION, PipelineConfig
from .exceptions import PersistenceError, ShardError
from .utils import atomic_json_write, human_bytes, sha256_file

logger = logging.getLogger(__name__)


@dataclass
class ShardResult:
    local_tar_path: Path
    drive_tar_path: Path
    size_bytes: int
    sha256: str


# ---------------------------------------------------------------------------
# TAR packaging
# ---------------------------------------------------------------------------

def package_speaker_shard(
    speaker_id: str,
    silver_local_dir: Path,
    package_dir: Path,
    valid_utterance_ids: List[str],
    *,
    config: PipelineConfig,
) -> Path:
    """Create a local uncompressed .tar containing only valid utterances.

    Internal TAR structure::

        s1/
        └── bbaf2n/
            ├── 0.jpg
            ├── 1.jpg
            └── audio.wav

    Uses a temporary file + rename for transactional safety.
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    tar_path = package_dir / f"{speaker_id}.tar"
    tar_tmp = package_dir / f"{speaker_id}.tar.tmp"

    if tar_tmp.exists():
        tar_tmp.unlink()

    logger.info("[%s] Packaging %d valid utterances into TAR …", speaker_id, len(valid_utterance_ids))

    try:
        with tarfile.open(tar_tmp, "w") as tf:
            for uid in sorted(valid_utterance_ids):
                utt_dir = silver_local_dir / uid
                if not utt_dir.is_dir():
                    logger.warning("[%s] Utterance dir missing: %s (skipping)", speaker_id, utt_dir)
                    continue
                # Add all files under speaker_id/utterance_id/
                for fpath in sorted(utt_dir.iterdir()):
                    arcname = f"{speaker_id}/{uid}/{fpath.name}"
                    tf.add(str(fpath), arcname=arcname)

        tar_tmp.rename(tar_path)
        size = tar_path.stat().st_size
        logger.info("[%s] TAR created: %s (%s)", speaker_id, tar_path, human_bytes(size))
        return tar_path

    except (OSError, tarfile.TarError) as exc:
        if tar_tmp.exists():
            tar_tmp.unlink()
        raise ShardError(f"[{speaker_id}] TAR packaging failed: {exc}") from exc


def verify_shard(tar_path: Path, speaker_id: str, valid_utterance_ids: List[str]) -> None:
    """Verify TAR integrity and that every valid utterance has entries.

    Raises :class:`.ShardError` if verification fails.
    """
    try:
        with tarfile.open(tar_path) as tf:
            members = {m.name for m in tf.getmembers()}
    except (tarfile.TarError, EOFError) as exc:
        raise ShardError(f"[{speaker_id}] TAR is not readable: {exc}") from exc

    missing_utterances: List[str] = []
    for uid in valid_utterance_ids:
        prefix = f"{speaker_id}/{uid}/"
        # There must be at least one member under this prefix
        if not any(m.startswith(prefix) for m in members):
            missing_utterances.append(uid)

    if missing_utterances:
        raise ShardError(
            f"[{speaker_id}] TAR is missing utterances: {missing_utterances[:5]}…"
        )

    logger.info("[%s] TAR verification OK (%d members)", speaker_id, len(members))


# ---------------------------------------------------------------------------
# Safe upload to Google Drive
# ---------------------------------------------------------------------------

def upload_shard(
    speaker_id: str,
    local_tar: Path,
    drive_shards_dir: Path,
    *,
    local_sha256: str,
    config: PipelineConfig,
) -> Path:
    """Copy the local TAR to Google Drive atomically.

    Upload transaction::

        copy to drive_shards_dir/sN.tar.partial
        verify destination size matches local
        rename to drive_shards_dir/sN.tar

    Raises :class:`.PersistenceError` on failure.
    """
    drive_shards_dir.mkdir(parents=True, exist_ok=True)
    drive_dest = drive_shards_dir / f"{speaker_id}.tar"
    drive_partial = drive_shards_dir / f"{speaker_id}.tar.partial"

    local_size = local_tar.stat().st_size
    logger.info(
        "[%s] Uploading %s -> %s (%s) …",
        speaker_id, local_tar, drive_dest, human_bytes(local_size),
    )

    try:
        if drive_partial.exists():
            drive_partial.unlink()
        shutil.copy2(str(local_tar), str(drive_partial))
    except (OSError, shutil.Error) as exc:
        raise PersistenceError(
            f"[{speaker_id}] Failed to copy shard to Drive: {exc}"
        ) from exc

    # Verify partial size
    partial_size = drive_partial.stat().st_size
    if partial_size != local_size:
        drive_partial.unlink()
        raise PersistenceError(
            f"[{speaker_id}] Upload size mismatch: local={local_size}, drive={partial_size}"
        )

    # Rename to final name
    try:
        drive_partial.rename(drive_dest)
    except OSError as exc:
        # On some filesystems rename across mount points fails; fall back to copy+delete
        try:
            shutil.copy2(str(drive_partial), str(drive_dest))
            drive_partial.unlink()
        except Exception as exc2:
            raise PersistenceError(
                f"[{speaker_id}] Could not finalise Drive shard: {exc2}"
            ) from exc2

    logger.info("[%s] Shard uploaded: %s", speaker_id, drive_dest)
    return drive_dest


def upload_speaker_metadata(
    speaker_id: str,
    metadata: dict,
    drive_metadata_dir: Path,
) -> Path:
    """Write speaker JSON metadata to Google Drive atomically."""
    drive_metadata_dir.mkdir(parents=True, exist_ok=True)
    dest = drive_metadata_dir / f"{speaker_id}.json"
    atomic_json_write(dest, metadata)
    logger.info("[%s] Metadata written: %s", speaker_id, dest)
    return dest


def verify_drive_shard(
    drive_tar: Path,
    local_size: int,
    local_sha256: str,
    speaker_id: str,
) -> None:
    """Confirm that the Drive shard matches the local reference.

    Raises :class:`.PersistenceError` on failure.
    """
    if not drive_tar.exists():
        raise PersistenceError(f"[{speaker_id}] Drive shard not found: {drive_tar}")

    drive_size = drive_tar.stat().st_size
    if drive_size != local_size:
        raise PersistenceError(
            f"[{speaker_id}] Drive shard size mismatch: local={local_size}, drive={drive_size}"
        )

    # SHA-256 on Drive is expensive for large shards; do it but log clearly.
    logger.info("[%s] Computing SHA-256 of Drive shard for verification …", speaker_id)
    drive_sha256 = sha256_file(drive_tar)
    if drive_sha256 != local_sha256:
        raise PersistenceError(
            f"[{speaker_id}] Drive shard SHA-256 mismatch. "
            f"local={local_sha256}, drive={drive_sha256}"
        )
    logger.info("[%s] Drive shard verified (sha256: %s…)", speaker_id, drive_sha256[:16])
