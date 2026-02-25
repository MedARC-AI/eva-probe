"""Embeddings writer for MIL classification without disk I/O."""

import os
from typing import Any, Dict, List, Sequence

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks
from loguru import logger
from torch import nn
from typing_extensions import override

from eva.core.models.modules.typings import INPUT_BATCH


class ClassificationMultiEmbeddingsInMemoryWriter(callbacks.BasePredictionWriter):
    """Caches grouped embeddings in memory for MIL `predict_fit` workflows."""

    def __init__(
        self,
        output_dir: str | None = None,
        backbone: nn.Module | None = None,
        dataloader_idx_map: Dict[int, str] | None = None,
        metadata_keys: List[str] | None = None,
        overwrite: bool = False,
        save_every_n: int = 100,
    ) -> None:
        """Initializes the writer.

        The signature intentionally mirrors `ClassificationEmbeddingsWriter` for config compatibility.
        """
        super().__init__(write_interval="batch")

        del output_dir, overwrite, save_every_n

        self._backbone = backbone
        self._dataloader_idx_map = dataloader_idx_map or {}
        self._group_key = "wsi_id" if not metadata_keys else metadata_keys[0]

        self._embeddings: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        self._targets: Dict[str, Dict[str, torch.Tensor]] = {}

    @override
    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer

        self._embeddings = {}
        self._targets = {}

        if self._backbone is not None:
            self._backbone = self._backbone.to(pl_module.device)
            self._backbone.eval()

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int],
        batch: INPUT_BATCH,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        del pl_module, batch_idx

        split = self._dataloader_idx_map.get(dataloader_idx)
        if split is None:
            raise KeyError(
                f"Predict dataloader index `{dataloader_idx}` is missing from `dataloader_idx_map`."
            )

        dataset = trainer.predict_dataloaders[dataloader_idx].dataset  # type: ignore
        _, targets, metadata = INPUT_BATCH(*batch)
        if not isinstance(targets, torch.Tensor):
            raise ValueError(f"Targets ({type(targets)}) should be `torch.Tensor`.")

        with torch.no_grad():
            embeddings = self._backbone(prediction) if self._backbone is not None else prediction

        if not isinstance(embeddings, torch.Tensor):
            raise ValueError(
                f"Embeddings ({type(embeddings)}) should be `torch.Tensor` for classification."
            )
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.shape[0] != targets.shape[0]:
            raise ValueError(
                "The number of embeddings and targets must match, "
                f"but got {embeddings.shape[0]} and {targets.shape[0]}."
            )

        split_embeddings = self._embeddings.setdefault(split, {})
        split_targets = self._targets.setdefault(split, {})

        for local_idx, global_idx in enumerate(batch_indices[: len(embeddings)]):
            group_id = _resolve_group_id(
                dataset=dataset,
                global_idx=global_idx,
                metadata=metadata,
                local_idx=local_idx,
                group_key=self._group_key,
            )

            split_embeddings.setdefault(group_id, []).append(embeddings[local_idx].detach().cpu())

            target = targets[local_idx].detach().cpu().to(torch.int64).reshape(-1)
            if target.numel() != 1:
                raise ValueError("Expected scalar class target for MIL.")
            target = target.squeeze(0)

            previous_target = split_targets.get(group_id)
            if previous_target is not None and not torch.equal(previous_target, target):
                raise ValueError(
                    f"Inconsistent targets for group `{group_id}`: "
                    f"{previous_target.item()} vs {target.item()}."
                )
            split_targets[group_id] = target

    @override
    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.datamodule is None:
            raise ValueError("Trainer datamodule is missing.")

        for split, grouped_embeddings in self._embeddings.items():
            dataset = getattr(trainer.datamodule.datasets, split, None)
            if dataset is None:
                raise ValueError(f"Datamodule does not define dataset split `{split}`.")
            if not hasattr(dataset, "set_data"):
                raise TypeError(
                    f"Dataset for split `{split}` must implement `set_data(embeddings, targets)`."
                )

            embeddings_by_group = {}
            patch_count = 0
            for group_id, chunks in grouped_embeddings.items():
                embeddings = torch.stack(chunks, dim=0).contiguous()
                embeddings_by_group[group_id] = embeddings
                patch_count += embeddings.shape[0]

            dataset.set_data(embeddings_by_group, self._targets[split])
            logger.info(
                "Cached %d groups (%d embeddings) in memory for split `%s`.",
                len(embeddings_by_group),
                patch_count,
                split,
            )

        self._embeddings = {}
        self._targets = {}


def _resolve_group_id(
    dataset: Any,
    global_idx: int,
    metadata: Dict[str, Any] | None,
    local_idx: int,
    group_key: str,
) -> str:
    if metadata is not None and group_key in metadata:
        item = metadata[group_key]
        if isinstance(item, torch.Tensor):
            value = item[local_idx]
            return str(value.item() if value.numel() == 1 else value)
        if isinstance(item, (list, tuple)):
            return str(item[local_idx])
        raise TypeError(f"Unsupported metadata type for `{group_key}`: {type(item)}")

    data_name = dataset.filename(global_idx)
    return os.path.splitext(data_name)[0]
