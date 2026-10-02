"""Remote download module for the GRID Audio-Visual Speech Corpus.

Supports:
- Primary source: University of Sheffield GRID server
- Fallback source: Zenodo record 3625687
- HTTP Range-based resume
- Streaming (no full-archive RAM load)
- Exponential-backoff retries
- .part temporary files
- Archive integrity verification
"""

from __future__ import annotations

import logging
import os
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

from .config import (
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_CONNECT_TIMEOUT_SEC,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_READ_TIMEOUT_SEC,
    DOWNLOAD_RETRY_BACKOFF_FACTOR,
    DOWNLOAD_RETRY_INITIAL_DELAY_SEC,
    DOWNLOAD_RETRY_MAX_DELAY_SEC,
    SHEFFIELD_AUDIO_URL_TEMPLATE,
    SHEFFIELD_VIDEO_URL_TEMPLATE,
    ZENODO_AUDIO_ARCHIVE_FILENAME,
    ZENODO_RECORD_API_URL,
    ZENODO_SPEAKER_FILENAME_TEMPLATE,
    PipelineConfig,
)
from .exceptions import ArchiveError, DownloadError
from .utils import human_bytes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level HTTP streaming download with resume support
# ---------------------------------------------------------------------------

def _download_url(
    url: str,
    dest: Path,
    *,
    config: PipelineConfig,
    description: str = "",
) -> Path:
    """Stream *url* to *dest*, supporting HTTP Range resume.

    Downloads to ``dest.with_suffix(dest.suffix + '.part')`` and renames
    to *dest* on success.  If *dest* already exists (complete), skips.

    Returns the final destination path.

    Raises :class:`.DownloadError` after retries are exhausted.
    """
    if dest.exists():
        logger.info("Already downloaded: %s (skipping)", dest)
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    label = description or url

    delay = config.download_retry_initial_delay_sec
    last_exc: Optional[Exception] = None

    for attempt in range(1, config.download_max_retries + 1):
        try:
            existing_bytes = part.stat().st_size if part.exists() else 0
            headers: dict[str, str] = {}
            if existing_bytes:
                headers["Range"] = f"bytes={existing_bytes}-"
                logger.info(
                    "[attempt %d/%d] Resuming %s from byte %s",
                    attempt, config.download_max_retries, label, existing_bytes,
                )
            else:
                logger.info(
                    "[attempt %d/%d] Downloading %s", attempt, config.download_max_retries, label
                )

            resp = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(config.download_connect_timeout_sec, config.download_read_timeout_sec),
            )

            # Handle Range not supported — restart from scratch.
            if existing_bytes and resp.status_code == 200:
                logger.debug("Server returned 200 (no Range support); restarting download.")
                existing_bytes = 0
                if part.exists():
                    part.unlink()
            elif resp.status_code not in (200, 206):
                raise DownloadError(
                    f"HTTP {resp.status_code} for {url}"
                )

            content_length = int(resp.headers.get("content-length", 0))
            total = existing_bytes + content_length if content_length else None
            downloaded = existing_bytes

            try:
                from tqdm import tqdm  # type: ignore[import-untyped]
                _pbar = tqdm(
                    total=total,
                    initial=existing_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=Path(url).name,
                    leave=False,
                )
            except ImportError:
                _pbar = None  # type: ignore[assignment]

            mode = "ab" if existing_bytes else "wb"
            with part.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if _pbar is not None:
                            _pbar.update(len(chunk))

            if _pbar is not None:
                _pbar.close()

            if total and downloaded < total:
                raise DownloadError(
                    f"Incomplete download: got {downloaded}/{total} bytes for {url}"
                )

            part.rename(dest)
            logger.info("Downloaded %s -> %s (%s)", label, dest, human_bytes(dest.stat().st_size))
            return dest

        except (requests.RequestException, DownloadError, OSError) as exc:
            last_exc = exc
            logger.warning(
                "[attempt %d/%d] Download failed for %s: %s",
                attempt, config.download_max_retries, label, exc,
            )
            if attempt < config.download_max_retries:
                actual_delay = min(delay, config.download_retry_max_delay_sec)
                logger.info("Retrying in %.1f s …", actual_delay)
                time.sleep(actual_delay)
                delay *= config.download_retry_backoff_factor

    raise DownloadError(
        f"All {config.download_max_retries} download attempts failed for {url}: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Archive verification helpers
# ---------------------------------------------------------------------------

def verify_zip(path: Path) -> None:
    """Raise :class:`.ArchiveError` if *path* is not a valid ZIP."""
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                raise ArchiveError(f"Corrupt ZIP member '{bad}' in {path}")
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"Not a valid ZIP file: {path}") from exc


def verify_tar(path: Path) -> None:
    """Raise :class:`.ArchiveError` if *path* is not a readable TAR."""
    try:
        with tarfile.open(path) as tf:
            # Iterate members to detect truncation.
            members = tf.getmembers()
            if not members:
                raise ArchiveError(f"TAR archive is empty: {path}")
    except (tarfile.TarError, EOFError) as exc:
        raise ArchiveError(f"Not a valid TAR archive: {path}") from exc


def _guess_archive_type(path: Path) -> str:
    """Return 'zip' or 'tar' based on extension/magic; raise if unknown."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return "zip"
    if suffix in (".tar", ".tgz", ".gz", ".bz2"):
        return "tar"
    # Magic bytes fallback
    with path.open("rb") as fh:
        magic = fh.read(6)
    if magic[:4] == b"PK\x03\x04":
        return "zip"
    if magic[:5] == b"ustar" or magic[:2] in (b"\x1f\x8b", b"BZh"):
        return "tar"
    raise ArchiveError(f"Unknown archive type: {path}")


# ---------------------------------------------------------------------------
# Zenodo helpers
# ---------------------------------------------------------------------------

def _fetch_zenodo_files(config: PipelineConfig) -> dict[str, str]:
    """Return ``{filename: download_url}`` for all files in the Zenodo record."""
    logger.info("Fetching Zenodo record metadata from %s", ZENODO_RECORD_API_URL)
    resp = requests.get(
        ZENODO_RECORD_API_URL,
        timeout=(config.download_connect_timeout_sec, config.download_read_timeout_sec),
    )
    if resp.status_code != 200:
        raise DownloadError(
            f"Zenodo API returned HTTP {resp.status_code}: {ZENODO_RECORD_API_URL}"
        )
    data = resp.json()
    files: dict[str, str] = {}
    for f in data.get("files", []):
        fname = f.get("key") or f.get("filename") or ""
        link = (f.get("links") or {}).get("self") or f.get("links", {}).get("download") or ""
        if fname and link:
            files[fname] = link
    if not files:
        raise DownloadError("Could not parse any file entries from Zenodo record metadata.")
    logger.debug("Zenodo record has %d files.", len(files))
    return files


# ---------------------------------------------------------------------------
# Public API: download one speaker's video + audio
# ---------------------------------------------------------------------------

class SpeakerDownloader:
    """Download video and audio archives for one GRID speaker.

    Tries Sheffield primary source first; falls back to Zenodo on failure.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._cfg = config
        self._zenodo_files: Optional[dict[str, str]] = None
        # Cached path to the Zenodo audio_25k.zip (downloaded at most once per session)
        self._zenodo_audio_cache: Optional[Path] = None

    def _zenodo_file_map(self) -> dict[str, str]:
        if self._zenodo_files is None:
            self._zenodo_files = _fetch_zenodo_files(self._cfg)
        return self._zenodo_files

    # ------------------------------------------------------------------
    # Primary source
    # ------------------------------------------------------------------

    def _download_primary_video(self, speaker_id: str) -> Path:
        url = SHEFFIELD_VIDEO_URL_TEMPLATE.format(speaker_id=speaker_id)
        dest = self._cfg.speaker_downloads_dir(speaker_id) / f"{speaker_id}.mpg_vcd.zip"
        return _download_url(url, dest, config=self._cfg, description=f"[primary] video {speaker_id}")

    def _download_primary_audio(self, speaker_id: str) -> Path:
        url = SHEFFIELD_AUDIO_URL_TEMPLATE.format(speaker_id=speaker_id)
        dest = self._cfg.speaker_downloads_dir(speaker_id) / f"{speaker_id}.tar"
        return _download_url(url, dest, config=self._cfg, description=f"[primary] audio {speaker_id}")

    # ------------------------------------------------------------------
    # Fallback source (Zenodo)
    # ------------------------------------------------------------------

    def _download_zenodo_video(self, speaker_id: str) -> Path:
        fname = ZENODO_SPEAKER_FILENAME_TEMPLATE.format(speaker_id=speaker_id)
        files = self._zenodo_file_map()
        if fname not in files:
            raise DownloadError(
                f"Zenodo has no file named '{fname}'. Available: {list(files)[:10]}…"
            )
        url = files[fname]
        dest = self._cfg.speaker_downloads_dir(speaker_id) / fname
        return _download_url(url, dest, config=self._cfg, description=f"[zenodo] video {speaker_id}")

    def _download_zenodo_audio_archive(self) -> Path:
        """Download audio_25k.zip once per session and cache it."""
        if self._zenodo_audio_cache is not None and self._zenodo_audio_cache.exists():
            return self._zenodo_audio_cache

        files = self._zenodo_file_map()
        fname = ZENODO_AUDIO_ARCHIVE_FILENAME
        if fname not in files:
            raise DownloadError(f"Zenodo has no file named '{fname}'.")
        url = files[fname]
        dest = self._cfg.cache_dir / fname
        path = _download_url(url, dest, config=self._cfg, description=f"[zenodo] {fname}")
        self._zenodo_audio_cache = path
        return path

    # ------------------------------------------------------------------
    # Public: download both archives for a speaker
    # ------------------------------------------------------------------

    def download(
        self, speaker_id: str
    ) -> Tuple[Path, Path, str, str]:
        """Download video and audio for *speaker_id*.

        Returns
        -------
        video_archive, audio_archive, video_source, audio_source
        """
        self._cfg.speaker_downloads_dir(speaker_id).mkdir(parents=True, exist_ok=True)

        # --- Video ---
        video_archive: Optional[Path] = None
        video_source = "primary"
        try:
            video_archive = self._download_primary_video(speaker_id)
            verify_zip(video_archive)
        except (DownloadError, ArchiveError) as exc:
            logger.warning(
                "[%s] Primary video failed (%s); trying Zenodo fallback.", speaker_id, exc
            )
            video_source = "zenodo_fallback"
            video_archive = self._download_zenodo_video(speaker_id)
            verify_zip(video_archive)

        # --- Audio ---
        audio_archive: Optional[Path] = None
        audio_source = "primary"
        try:
            audio_archive = self._download_primary_audio(speaker_id)
            verify_tar(audio_archive)
        except (DownloadError, ArchiveError) as exc:
            logger.warning(
                "[%s] Primary audio failed (%s); trying Zenodo audio_25k fallback.", speaker_id, exc
            )
            audio_source = "zenodo_fallback"
            audio_archive = self._download_zenodo_audio_archive()
            verify_zip(audio_archive)

        assert video_archive is not None
        assert audio_archive is not None
        return video_archive, audio_archive, video_source, audio_source
