"""Read individual GRID utterances from uncompressed TARs; never extract archives."""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .manifest import ManifestRecord
from .error_exception import DatasetReadError

@dataclass
class _OpenShard:
    archive: tarfile.TarFile
    by_prefix: dict[str, dict[str, tarfile.TarInfo]]


class GridShardReader:
    """Lazy TAR index, limited open-file LRU, optional local SSD shard cache.

    The reader is process-aware so forked/spawned DataLoader workers do not
    reuse an open archive handle from another process.
    """

    def __init__(
        self, root: str | Path, *, cache_dir: str | Path | None = None,
        max_open_shards: int = 2, max_member_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if max_open_shards < 1 or max_member_bytes < 1:
            raise ValueError("Reader limits must be positive")
        self.root = Path(root).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.max_open_shards = max_open_shards
        self.max_member_bytes = max_member_bytes
        self._pid = os.getpid()
        self._opened: OrderedDict[str, _OpenShard] = OrderedDict()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_opened"] = OrderedDict()
        state["_pid"] = None
        return state

    def close(self) -> None:
        for shard in self._opened.values():
            shard.archive.close()
        self._opened.clear()

    def __del__(self) -> None:
        if hasattr(self, "_opened"):
            self.close()

    def _local_path(self, speaker: str) -> Path:
        source = self.root / "silver" / "shards" / f"{speaker}.tar"
        if not source.is_file():
            raise DatasetReadError(f"Missing Silver shard: {source}")
        if self.cache_dir is None:
            return source

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dest = self.cache_dir / source.name
        size = source.stat().st_size
        if dest.is_file() and dest.stat().st_size == size:
            return dest
        # Atomic publication; temporary copy lives on the local cache volume.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{speaker}-", suffix=".tmp", dir=self.cache_dir)
        try:
            with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
            if Path(tmp_name).stat().st_size != size:
                raise DatasetReadError(f"Truncated cached copy for {speaker}")
            os.replace(tmp_name, dest)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return dest

    def _get_shard(self, speaker: str) -> _OpenShard:
        pid = os.getpid()
        if pid != self._pid:
            # Child must not close handles inherited from its parent.
            self._opened = OrderedDict()
            self._pid = pid

        if speaker in self._opened:
            self._opened.move_to_end(speaker)
            return self._opened[speaker]

        path = self._local_path(speaker)
        try:
            archive = tarfile.open(path, mode="r:")
            by_prefix: dict[str, dict[str, tarfile.TarInfo]] = {}
            for member in archive.getmembers():
                if not member.isfile() or "/" not in member.name:
                    continue
                prefix, _, name = member.name.rpartition("/")
                by_prefix.setdefault(prefix + "/", {})[name] = member
        except (OSError, tarfile.TarError) as exc:
            raise DatasetReadError(f"Could not index {path}: {exc}") from exc

        if len(self._opened) >= self.max_open_shards:
            _, evicted = self._opened.popitem(last=False)
            evicted.archive.close()
        opened = _OpenShard(archive, by_prefix)
        self._opened[speaker] = opened
        return opened

    def read_utterance(self, record: ManifestRecord) -> tuple[dict[int, bytes], bytes]:
        """Return original-index JPEG bytes and WAV bytes for one utterance."""
        shard = self._get_shard(record.speaker_id)
        entries = shard.by_prefix.get(record.shard_member_prefix)
        if not entries or "audio.wav" not in entries:
            raise DatasetReadError(f"Audio or utterance missing: {record.shard_member_prefix}")

        def read(member: tarfile.TarInfo) -> bytes:
            if member.size > self.max_member_bytes:
                raise DatasetReadError(f"Oversized TAR member: {member.name}")
            stream = shard.archive.extractfile(member)
            if stream is None:
                raise DatasetReadError(f"Unreadable TAR member: {member.name}")
            with stream:
                payload = stream.read()
            if len(payload) != member.size:
                raise DatasetReadError(f"Truncated TAR member: {member.name}")
            return payload

        frames: dict[int, bytes] = {}
        for filename, member in entries.items():
            if not filename.endswith(".jpg") or not filename[:-4].isdigit():
                continue
            index = int(filename[:-4])
            if index in frames:
                raise DatasetReadError(f"Duplicate frame index {index} in TAR")
            frames[index] = read(member)
        if not frames:
            raise DatasetReadError(f"No JPEG frames: {record.shard_member_prefix}")
        return frames, read(entries["audio.wav"])
