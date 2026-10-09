"""Persistent processing state manager for the GRID pipeline.

The state file lives on Google Drive so that it survives Colab restarts.
Writes are atomic (temp-file + rename) to avoid leaving malformed JSON
after an interruption.

State machine transitions::

    PENDING -> DOWNLOADING -> EXTRACTING -> PREPROCESSING -> QC
         -> PACKAGING -> UPLOADING -> DONE
    Any non-DONE state -> FAILED  (on unrecoverable error)
    DONE -> (skipped on resume unless --force)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    RETRYABLE_STATUSES,
    SCHEMA_VERSION,
    TERMINAL_SUCCESS_STATUSES,
    SpeakerStatus,
)
from .exceptions import PersistenceError
from .utils import atomic_json_write, sha256_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-speaker state record
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_speaker_state(speaker_id: str) -> Dict[str, Any]:
    return {
        "speaker_id": speaker_id,
        "status": SpeakerStatus.PENDING,
        "attempt_count": 0,
        "started_at": None,
        "completed_at": None,
        "last_error": None,
        "video_source_used": None,
        "audio_source_used": None,
        "shard_path": None,
        "metadata_path": None,
        "shard_size_bytes": None,
        "shard_sha256": None,
    }


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

class StateManager:
    """Load, query, and persist speaker processing state.

    Parameters
    ----------
    state_file:
        Path to ``processing_state.json`` on Google Drive.
    usable_speakers:
        Ordered list of all usable speaker IDs (e.g. ``['s1', 's2', ...]``).
    """

    def __init__(self, state_file: Path, usable_speakers: List[str]) -> None:
        self._state_file = state_file
        self._usable_speakers = usable_speakers
        self._state: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load state from disk (or initialise fresh state)."""
        if self._state_file.exists():
            try:
                raw = json.loads(self._state_file.read_text(encoding="utf-8"))
                self._state = raw
                logger.info("Loaded processing state from %s", self._state_file)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not read state file (%s); starting fresh: %s",
                    self._state_file, exc,
                )
                self._state = {}
        else:
            logger.info("No state file found; starting fresh at %s", self._state_file)

        # Ensure every usable speaker has an entry.
        speakers_in_state: Dict[str, Any] = self._state.get("speakers", {})
        for sid in self._usable_speakers:
            if sid not in speakers_in_state:
                speakers_in_state[sid] = _empty_speaker_state(sid)

        self._state.setdefault("schema_version", SCHEMA_VERSION)
        self._state.setdefault("dataset", "GRID")
        self._state["speakers"] = speakers_in_state

    def _save(self) -> None:
        """Persist the current state atomically to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_write(self._state_file, self._state)
        except OSError as exc:
            raise PersistenceError(
                f"Failed to save state file {self._state_file}: {exc}"
            ) from exc

    def _speaker(self, speaker_id: str) -> Dict[str, Any]:
        return self._state["speakers"][speaker_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self, speaker_id: str) -> str:
        return self._speaker(speaker_id)["status"]

    def is_done(self, speaker_id: str) -> bool:
        return self.get_status(speaker_id) in TERMINAL_SUCCESS_STATUSES

    def should_retry(self, speaker_id: str) -> bool:
        return self.get_status(speaker_id) in RETRYABLE_STATUSES

    def transition(self, speaker_id: str, new_status: str) -> None:
        """Move *speaker_id* to *new_status* and persist state."""
        spk = self._speaker(speaker_id)
        old = spk["status"]
        spk["status"] = new_status
        if new_status not in (SpeakerStatus.PENDING, SpeakerStatus.DONE):
            # Record start time on first real transition out of PENDING.
            if spk["started_at"] is None:
                spk["started_at"] = _now_iso()
        logger.info("[%s] %s -> %s", speaker_id, old, new_status)
        self._save()

    def record_start(self, speaker_id: str) -> None:
        spk = self._speaker(speaker_id)
        spk["attempt_count"] = spk.get("attempt_count", 0) + 1
        spk["started_at"] = _now_iso()
        spk["completed_at"] = None
        spk["last_error"] = None
        self._save()

    def record_failure(self, speaker_id: str, error: str) -> None:
        spk = self._speaker(speaker_id)
        spk["status"] = SpeakerStatus.FAILED
        spk["last_error"] = error
        logger.error("[%s] FAILED: %s", speaker_id, error)
        self._save()

    def record_done(
        self,
        speaker_id: str,
        *,
        shard_path: str,
        metadata_path: str,
        shard_size_bytes: int,
        shard_sha256: str,
        video_source_used: str,
        audio_source_used: str,
    ) -> None:
        spk = self._speaker(speaker_id)
        spk["status"] = SpeakerStatus.DONE
        spk["completed_at"] = _now_iso()
        spk["last_error"] = None
        spk["shard_path"] = shard_path
        spk["metadata_path"] = metadata_path
        spk["shard_size_bytes"] = shard_size_bytes
        spk["shard_sha256"] = shard_sha256
        spk["video_source_used"] = video_source_used
        spk["audio_source_used"] = audio_source_used
        logger.info("[%s] DONE (shard: %s bytes, sha256: %s...)", speaker_id,
                    shard_size_bytes, shard_sha256[:16])
        self._save()

    def get_all_statuses(self) -> Dict[str, str]:
        return {
            sid: info["status"]
            for sid, info in self._state["speakers"].items()
        }

    def get_speaker_info(self, speaker_id: str) -> Dict[str, Any]:
        return dict(self._speaker(speaker_id))

    def verify_done_artefacts(
        self,
        speaker_id: str,
        *,
        silver_shards_dir: Path,
        silver_metadata_dir: Path,
    ) -> bool:
        """Return True iff this speaker's persistent artefacts verify on disk.

        Resolve artifacts from the configured dataset root rather than the
        absolute paths stored in state, which may point to a previous Drive
        shortcut or mount location.
        """
        spk = self._speaker(speaker_id)
        shard_path = silver_shards_dir / f"{speaker_id}.tar"
        meta_path = silver_metadata_dir / f"{speaker_id}.json"
        expected_sha = spk.get("shard_sha256") or ""
        expected_size = spk.get("shard_size_bytes")

        if not shard_path.exists():
            logger.warning("[%s] Shard missing: %s", speaker_id, shard_path)
            return False
        if not meta_path.exists():
            logger.warning("[%s] Metadata missing: %s", speaker_id, meta_path)
            return False
        if expected_size is not None and shard_path.stat().st_size != expected_size:
            logger.warning("[%s] Shard size mismatch (expected %s, got %s)",
                           speaker_id, expected_size, shard_path.stat().st_size)
            return False
        if expected_sha:
            actual_sha = sha256_file(shard_path)
            if actual_sha != expected_sha:
                logger.warning("[%s] Shard SHA-256 mismatch", speaker_id)
                return False
        return True

    def record_existing_done(
        self,
        speaker_id: str,
        *,
        shard_path: Path,
        metadata_path: Path,
    ) -> None:
        """Reconcile state with already-verified artifacts without reprocessing."""
        spk = self._speaker(speaker_id)
        updates = {
            "status": SpeakerStatus.DONE,
            "shard_path": str(shard_path),
            "metadata_path": str(metadata_path),
            "completed_at": spk.get("completed_at") or _now_iso(),
            "last_error": None,
        }
        changed = any(spk.get(key) != value for key, value in updates.items())
        if not changed:
            return
        spk.update(updates)
        logger.info("[%s] Verified existing artifacts; state reconciled to DONE.", speaker_id)
        self._save()
