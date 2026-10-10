"""Read JPEG and WAV bytes from processed GRID TAR shards without extracting them."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .error_exception import DatasetReadError
from .manifest import ManifestRecord


@dataclass
class _OpenShard:
    """An open TAR archive and its members grouped by utterance folder."""

    archive: tarfile.TarFile
    by_prefix: dict[str, dict[str, tarfile.TarInfo]]


class GridShardReader:
    """Read media bytes for an utterance described by a Gold ManifestRecord.

    The reader indexes TAR member names only when a shard is first accessed.
    Open archives are limited by a small LRU cache. Optionally, a shard can be
    copied to a local cache directory before being opened.

    This class does not decode JPEG/WAV data or create video files.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        cache_dir: str | Path | None = None,
        max_open_shards: int = 2,
        max_member_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")

        if max_member_bytes < 1:
            raise ValueError("max_member_bytes must be positive")

        self.root = Path(root).expanduser().resolve()

        if cache_dir is None:
            self.cache_dir = None
        else:
            self.cache_dir = Path(cache_dir).expanduser().resolve()

        self.max_open_shards = max_open_shards
        self.max_member_bytes = max_member_bytes
        self._pid = os.getpid()
        self._opened: OrderedDict[str, _OpenShard] = OrderedDict()

    def __getstate__(self) -> dict:
        """Exclude open TAR handles when sending the reader to another process."""
        state = self.__dict__.copy()
        state["_opened"] = OrderedDict()
        state["_pid"] = None
        return state

    def close(self) -> None:
        """Close every TAR file currently held by this reader."""
        for shard in self._opened.values():
            shard.archive.close()

        self._opened.clear()

    def __del__(self) -> None:
        if hasattr(self, "_opened"):
            self.close()

    def _local_path(self, speaker: str) -> Path:
        """Find a speaker TAR, optionally copying it to a local cache."""
        source = self.root / "silver" / "shards" / f"{speaker}.tar"

        if not source.is_file():
            raise DatasetReadError(f"Missing Silver shard: {source}")

        if self.cache_dir is None:
            return source

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        destination = self.cache_dir / source.name
        source_size = source.stat().st_size

        if destination.is_file():
            cached_size = destination.stat().st_size
            if cached_size == source_size:
                return destination

        # Copy to a temporary file first. The final cache path is published
        # only after all bytes have been copied successfully.
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{speaker}-",
            suffix=".tmp",
            dir=self.cache_dir,
        )

        try:
            with os.fdopen(file_descriptor, "wb") as output:
                with source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output, length=1024 * 1024)

            copied_size = Path(temporary_name).stat().st_size
            if copied_size != source_size:
                raise DatasetReadError(f"Truncated cached copy for {speaker}")

            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

        return destination

    def _get_shard(self, speaker: str) -> _OpenShard:
        """Open and index a speaker TAR, reusing a cached handle when possible."""
        current_pid = os.getpid()

        if current_pid != self._pid:
            # A worker must not reuse TAR handles inherited from its parent.
            self._opened = OrderedDict()
            self._pid = current_pid

        if speaker in self._opened:
            self._opened.move_to_end(speaker)
            return self._opened[speaker]

        path = self._local_path(speaker)
        archive: tarfile.TarFile | None = None

        try:
            archive = tarfile.open(path, mode="r:")
            by_prefix: dict[str, dict[str, tarfile.TarInfo]] = {}

            for member in archive.getmembers():
                if not member.isfile():
                    continue

                prefix, separator, filename = member.name.rpartition("/")
                if separator == "":
                    continue

                folder = prefix + "/"

                if folder not in by_prefix:
                    by_prefix[folder] = {}

                entries = by_prefix[folder]
                if filename in entries:
                    raise DatasetReadError(f"Duplicate TAR member: {member.name}")

                entries[filename] = member
        except (OSError, tarfile.TarError, DatasetReadError) as error:
            if archive is not None:
                archive.close()

            if isinstance(error, DatasetReadError):
                raise

            raise DatasetReadError(f"Could not index {path}: {error}") from error

        # Limit the number of simultaneously open archives.
        if len(self._opened) >= self.max_open_shards:
            old_speaker, old_shard = self._opened.popitem(last=False)
            old_shard.archive.close()

        opened_shard = _OpenShard(archive=archive, by_prefix=by_prefix)
        self._opened[speaker] = opened_shard

        return opened_shard

    def _get_entries(
        self,
        record: ManifestRecord,
    ) -> tuple[_OpenShard, dict[str, tarfile.TarInfo]]:
        """Locate the TAR members belonging to a single utterance."""
        shard = self._get_shard(record.speaker_id)
        entries = shard.by_prefix.get(record.shard_member_prefix)

        if entries is None:
            raise DatasetReadError(
                f"Utterance missing in TAR: {record.shard_member_prefix}"
            )

        return shard, entries

    def _get_frame_members(
        self,
        record: ManifestRecord,
        entries: dict[str, tarfile.TarInfo],
    ) -> dict[int, tarfile.TarInfo]:
        """Map original numeric frame indices to their TAR entries."""
        frame_members: dict[int, tarfile.TarInfo] = {}

        for filename, member in entries.items():
            if not filename.endswith(".jpg"):
                continue

            frame_number = filename[:-4]
            if not frame_number.isdigit():
                continue

            frame_index = int(frame_number)

            if frame_index in frame_members:
                raise DatasetReadError(
                    f"Duplicate frame index {frame_index} in TAR: "
                    f"{record.shard_member_prefix}"
                )

            frame_members[frame_index] = member

        if not frame_members:
            raise DatasetReadError(
                f"No JPEG frames: {record.shard_member_prefix}"
            )

        return frame_members

    def _read_member(
        self,
        shard: _OpenShard,
        member: tarfile.TarInfo,
    ) -> bytes:
        """Read one TAR member with a size limit and truncation check."""
        if member.size > self.max_member_bytes:
            raise DatasetReadError(f"Oversized TAR member: {member.name}")

        try:
            stream = shard.archive.extractfile(member)

            if stream is None:
                raise DatasetReadError(f"Unreadable TAR member: {member.name}")

            with stream:
                payload = stream.read()
        except (OSError, tarfile.TarError) as error:
            raise DatasetReadError(
                f"Could not read TAR member {member.name}: {error}"
            ) from error

        if len(payload) != member.size:
            raise DatasetReadError(f"Truncated TAR member: {member.name}")

        return payload

    def list_frame_indices(self, record: ManifestRecord) -> list[int]:
        """Return sorted frame indices without reading JPEG bytes."""
        shard, entries = self._get_entries(record)
        frame_members = self._get_frame_members(record, entries)

        frame_indices = list(frame_members.keys())
        frame_indices.sort()

        return frame_indices

    def read_frame(self, record: ManifestRecord, index: int) -> bytes:
        """Read only the JPEG bytes for a requested original frame index."""
        if index < 0:
            raise ValueError("Frame index must be non-negative")

        shard, entries = self._get_entries(record)
        frame_members = self._get_frame_members(record, entries)

        member = frame_members.get(index)
        if member is None:
            raise DatasetReadError(
                f"Frame {index} not found: {record.shard_member_prefix}"
            )

        return self._read_member(shard, member)

    def read_audio(self, record: ManifestRecord) -> bytes:
        """Read only the WAV bytes for the requested utterance."""
        shard, entries = self._get_entries(record)
        audio_member = entries.get("audio.wav")

        if audio_member is None:
            raise DatasetReadError(
                f"Audio missing: {record.shard_member_prefix}"
            )

        return self._read_member(shard, audio_member)

    def read_utterance(
        self,
        record: ManifestRecord,
    ) -> tuple[dict[int, bytes], bytes]:
        """Read all JPEG frames and WAV bytes for one utterance."""
        shard, entries = self._get_entries(record)

        audio_member = entries.get("audio.wav")
        if audio_member is None:
            raise DatasetReadError(
                f"Audio missing: {record.shard_member_prefix}"
            )

        frame_members = self._get_frame_members(record, entries)
        frames: dict[int, bytes] = {}

        for frame_index in sorted(frame_members.keys()):
            member = frame_members[frame_index]
            frames[frame_index] = self._read_member(shard, member)

        audio = self._read_member(shard, audio_member)
        return frames, audio
