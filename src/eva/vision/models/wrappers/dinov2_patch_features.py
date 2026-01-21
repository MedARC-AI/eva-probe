"""DINOv2 backbone wrapper that returns patch feature maps for segmentation."""

from typing import List

import torch
from torch import nn

from eva.core.models.wrappers import _utils as wrapper_utils


class DinoV2PatchFeatures(nn.Module):
    """Loads a DINOv2 backbone and returns patch feature maps as a list."""

    def __init__(
        self,
        repo_or_dir: str,
        model: str,
        checkpoint_path: str,
        pretrained: bool = True,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self._model = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=model,
            pretrained=pretrained,
        )
        if checkpoint_path:
            wrapper_utils.load_model_weights(self._model, checkpoint_path)
        self._n_layers = n_layers

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = self._model.get_intermediate_layers(x, n=self._n_layers, reshape=True)
        return list(outputs)
