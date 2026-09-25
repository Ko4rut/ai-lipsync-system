"""Planning skeleton for the research data pipeline."""


class DataCrawler:
    """Collect source media and provenance metadata into the Bronze layer."""

    pass


class BronzeStage:
    """Preserve raw downloads, source metadata and checksums."""

    pass


class SilverStage:
    """Validate, deduplicate, segment and normalize collected media."""

    pass


class GoldStage:
    """Build versioned manifests and train/validation/test splits."""

    pass


class DatasetPipeline:
    """Coordinate collection and promotion from Bronze to Silver to Gold."""

    pass
