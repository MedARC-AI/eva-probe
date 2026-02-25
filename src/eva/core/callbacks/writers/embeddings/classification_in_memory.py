"""Embeddings writer for classification without disk I/O."""

from typing import Any, Dict, List, Sequence

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks
from loguru import logger
from torch import nn
from typing_extensions import override

from eva.core.models.modules.typings import INPUT_BATCH


class ClassificationEmbeddingsInMemoryWriter(callbacks.BasePredictionWriter):
    """Caches generated classification embeddings in memory for fast `predict_fit`."""

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

        del output_dir, metadata_keys, overwrite, save_every_n

        self._backbone = backbone
        self._dataloader_idx_map = dataloader_idx_map or {}
        self._embeddings: Dict[str, List[torch.Tensor]] = {}
        self._targets: Dict[str, List[torch.Tensor]] = {}

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
        del trainer, pl_module, batch_indices, batch_idx

        split = self._dataloader_idx_map.get(dataloader_idx)
        if split is None:
            raise KeyError(
                f"Predict dataloader index `{dataloader_idx}` is missing from `dataloader_idx_map`."
            )

        _, targets, _ = INPUT_BATCH(*batch)
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

        self._embeddings.setdefault(split, []).append(embeddings.detach().cpu())
        self._targets.setdefault(split, []).append(targets.detach().cpu().to(torch.int64))

    @override
    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.datamodule is None:
            raise ValueError("Trainer datamodule is missing.")

        for split, embeddings_chunks in self._embeddings.items():
            dataset = getattr(trainer.datamodule.datasets, split, None)
            if dataset is None:
                raise ValueError(f"Datamodule does not define dataset split `{split}`.")
            if not hasattr(dataset, "set_data"):
                raise TypeError(
                    f"Dataset for split `{split}` must implement `set_data(embeddings, targets)`."
                )

            embeddings = torch.cat(embeddings_chunks, dim=0).contiguous()
            targets = torch.cat(self._targets[split], dim=0).contiguous()
            dataset.set_data(embeddings, targets)
            logger.info(f"Cached {len(targets)} embeddings in memory for split `{split}`.")

        self._embeddings = {}
        self._targets = {}
