"""Embedding callback writers."""

from eva.core.callbacks.writers.embeddings.classification import ClassificationEmbeddingsWriter
from eva.core.callbacks.writers.embeddings.classification_in_memory import (
    ClassificationEmbeddingsInMemoryWriter,
)
from eva.core.callbacks.writers.embeddings.classification_multi_in_memory import (
    ClassificationMultiEmbeddingsInMemoryWriter,
)
from eva.core.callbacks.writers.embeddings.segmentation import SegmentationEmbeddingsWriter

__all__ = [
    "ClassificationEmbeddingsWriter",
    "ClassificationEmbeddingsInMemoryWriter",
    "ClassificationMultiEmbeddingsInMemoryWriter",
    "SegmentationEmbeddingsWriter",
]
