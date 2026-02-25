"""In-memory dataset class for grouped (MIL) embeddings."""

from typing import Callable, Dict, Literal, Tuple

import torch
from typing_extensions import override

from eva.core.data.datasets import base


class InMemoryMultiEmbeddingsClassificationDataset(
    base.MapDataset[Tuple[torch.Tensor, torch.Tensor]]
):
    """Dataset class where each sample corresponds to multiple cached embeddings."""

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

        self._multi_ids: list[str] = []
        self._embeddings_by_id: Dict[str, torch.Tensor] = {}
        self._targets_by_id: Dict[str, torch.Tensor] = {}

    def set_data(
        self,
        embeddings_by_id: Dict[str, torch.Tensor],
        targets_by_id: Dict[str, torch.Tensor],
    ) -> None:
        """Sets the grouped in-memory tensors."""
        if len(embeddings_by_id) == 0:
            raise ValueError("Received no grouped embeddings.")
        if set(embeddings_by_id) != set(targets_by_id):
            raise ValueError("Group ids in embeddings and targets do not match.")

        checked_embeddings = {}
        checked_targets = {}
        for group_id, embeddings in embeddings_by_id.items():
            if embeddings.ndim != 2:
                raise ValueError(
                    f"Expected 2D tensor for `{group_id}`, but got shape {embeddings.shape}."
                )
            target = targets_by_id[group_id].reshape(-1)
            if target.numel() != 1:
                raise ValueError(f"Expected scalar class target for `{group_id}`.")

            checked_embeddings[group_id] = embeddings
            checked_targets[group_id] = target.squeeze(0).to(torch.int64)

        self._multi_ids = list(checked_embeddings.keys())
        self._embeddings_by_id = checked_embeddings
        self._targets_by_id = checked_targets

    @override
    def configure(self) -> None:
        pass

    @override
    def validate(self) -> None:
        if len(self._multi_ids) == 0:
            raise ValueError(
                f"No in-memory grouped embeddings found for `{self._split}` split. "
                "Run `eva predict_fit` with `InMemoryMultiEmbeddingsClassificationDataset`."
            )

    def load_embeddings(self, index: int) -> torch.Tensor:
        """Returns stacked embeddings of one MIL bag."""
        return self._embeddings_by_id[self._multi_ids[index]]

    def load_target(self, index: int) -> torch.Tensor:
        """Returns target of one MIL bag."""
        return self._targets_by_id[self._multi_ids[index]]

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
        return len(self._multi_ids)
