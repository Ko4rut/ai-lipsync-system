"""Custom exceptions for the GRID preprocessing pipeline.

All pipeline-specific exceptions descend from ``GRIDPipelineError`` so that
callers can catch the entire family with a single clause when needed.
"""

from __future__ import annotations


class GRIDPipelineError(Exception):
    """Base class for all GRID pipeline errors."""


class DownloadError(GRIDPipelineError):
    """Raised when a download fails after all retries are exhausted."""


class ArchiveError(GRIDPipelineError):
    """Raised when an archive cannot be opened or is corrupt."""


class SourceLayoutError(GRIDPipelineError):
    """Raised when an extracted archive has an unexpected directory structure."""


class PairingError(GRIDPipelineError):
    """Raised for fatal pairing failures (individual utterances use QC instead)."""


class PreprocessingError(GRIDPipelineError):
    """Raised for speaker-level preprocessing infrastructure failures."""


class QCError(GRIDPipelineError):
    """Raised for speaker-level QC infrastructure failures."""


class ShardError(GRIDPipelineError):
    """Raised for TAR packaging or validation failures."""


class PersistenceError(GRIDPipelineError):
    """Raised for upload or persistent artefact verification failures."""


class FaceDetectionSetupError(GRIDPipelineError):
    """Raised when the face detection model or weights cannot be initialised."""


class DriveNotMountedError(GRIDPipelineError):
    """Raised when Google Drive is not mounted at the expected path."""
