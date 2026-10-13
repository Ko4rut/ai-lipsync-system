class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass


class ManifestError(DatasetError, ValueError):
    """Invalid or inconsistent Gold manifest."""
    pass


class DatasetReadError(DatasetError):
    """Failed to read or decode dataset content."""
    pass