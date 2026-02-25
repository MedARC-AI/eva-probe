"""Embedding cllassification datasets API."""

from eva.core.data.datasets.classification.embeddings import EmbeddingsClassificationDataset
from eva.core.data.datasets.classification.in_memory_embeddings import (
    InMemoryEmbeddingsClassificationDataset,
)
from eva.core.data.datasets.classification.in_memory_multi_embeddings import (
    InMemoryMultiEmbeddingsClassificationDataset,
)
from eva.core.data.datasets.classification.multi_embeddings import (
    MultiEmbeddingsClassificationDataset,
)

__all__ = [
    "EmbeddingsClassificationDataset",
    "InMemoryEmbeddingsClassificationDataset",
    "InMemoryMultiEmbeddingsClassificationDataset",
    "MultiEmbeddingsClassificationDataset",
]
