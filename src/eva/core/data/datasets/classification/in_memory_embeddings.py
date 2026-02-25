"""In-memory embeddings dataset for classification."""

from typing import Callable, Dict, Literal, Tuple

import torch
from typing_extensions import override

from eva.core.data.datasets import base


class InMemoryEmbeddingsClassificationDataset(base.MapDataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Classification dataset backed by tensors populated during `predict`."""

    def __init__(
        self,
        root: str | None = None,
        manifest_file: str | None = None,
        split: Literal["train", "val", "test"] | None = None,
        column_mapping: Dict[str, str] | None = None,
        embeddings_transforms: Callable | None = None,
        target_transforms: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        `root`, `manifest_file` and `column_mapping` are accepted for config compatibility.
        """
        super().__init__()

        del root, manifest_file, column_mapping

        if split is None:
            raise ValueError("`split` must be provided.")
        self._split = split
        self._embeddings_transforms = embeddings_transforms
        self._target_transforms = target_transforms

        self._embeddings: torch.Tensor | None = None
        self._targets: torch.Tensor | None = None

    def set_data(self, embeddings: torch.Tensor, targets: torch.Tensor) -> None:
        """Sets the in-memory tensors."""
        if embeddings.ndim < 2:
            raise ValueError(
                f"Expected `embeddings` with at least 2 dims, but got shape {embeddings.shape}."
            )
        if targets.ndim != 1:
            raise ValueError(f"Expected 1D `targets`, but got shape {targets.shape}.")
        if embeddings.shape[0] != targets.shape[0]:
            raise ValueError(
                "The number of embeddings and targets must match, "
                f"but got {embeddings.shape[0]} and {targets.shape[0]}."
            )

        self._embeddings = embeddings.contiguous()
        self._targets = targets.to(torch.int64).contiguous()

    @override
    def configure(self) -> None:
        pass

    @override
    def validate(self) -> None:
        if self._embeddings is None or self._targets is None:
            raise ValueError(
                f"No in-memory embeddings found for `{self._split}` split. "
                "Run `eva predict_fit` with `InMemoryEmbeddingsClassificationDataset`."
            )
        if self._embeddings.shape[0] != self._targets.shape[0]:
            raise ValueError(
                "The number of embeddings and targets must match, "
                f"but got {self._embeddings.shape[0]} and {self._targets.shape[0]}."
            )

    def load_embeddings(self, index: int) -> torch.Tensor:
        """Returns the `index`'th embedding sample."""
        if self._embeddings is None:
            raise ValueError("Dataset tensors were not initialized.")
        return self._embeddings[index]

    def load_target(self, index: int) -> torch.Tensor:
        """Returns the `index`'th target sample."""
        if self._targets is None:
            raise ValueError("Dataset tensors were not initialized.")
        return self._targets[index]

    @override
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.load_embeddings(index)
        target = self.load_target(index)

        if self._embeddings_transforms is not None:
            embeddings = self._embeddings_transforms(embeddings)
        if self._target_transforms is not None:
            target = self._target_transforms(target)

        return embeddings, target

    @override
    def __len__(self) -> int:
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])
