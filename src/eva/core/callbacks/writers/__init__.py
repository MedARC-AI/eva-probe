"""Writers callbacks API."""

from eva.core.callbacks.writers.embeddings import (
    ClassificationEmbeddingsInMemoryWriter,
    ClassificationMultiEmbeddingsInMemoryWriter,
    ClassificationEmbeddingsWriter,
    SegmentationEmbeddingsWriter,
)

__all__ = [
    "ClassificationEmbeddingsWriter",
    "ClassificationEmbeddingsInMemoryWriter",
    "ClassificationMultiEmbeddingsInMemoryWriter",
    "SegmentationEmbeddingsWriter",
]
