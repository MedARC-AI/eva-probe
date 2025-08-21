"""Helper function from models defined with a function locally."""


from typing import Any, Callable, Dict

import jsonargparse
import torch
from torch import nn
from typing_extensions import override

from eva.core.models.wrappers import _utils, base


class ModelFromLocal(base.BaseModel[torch.Tensor, torch.Tensor]):
    """Wrapper class for models which are initialized from functions.

    This is helpful for initializing models in a `.yaml` configuration file.
    """

    def __init__(
        self,
        local_repo_path: str,
        model_name: str,
        checkpoint_path: str,
    ) -> None:
        """Initializes and constructs the model.

        Args:
            path: The path to the callable object (class or function).
            arguments: The extra callable function / class arguments.
            checkpoint_path: The path to the checkpoint to load the model
                weights from. This is currently only supported for torch
                model checkpoints. For other formats, the checkpoint loading
                should be handled within the provided callable object in <path>.
            transforms: The transforms to apply to the output tensor
                produced by the model.
        """
        #super().__init__(transforms=transforms)
        super().__init__()

        self.local_repo = local_repo_path
        self.model_name = model_name
        self.weights_path = checkpoint_path

        self.load_model()

    @override
    def load_model(self) -> None:
        #Only loads models from local.
        #After a model has been loaded, it will be copied into local cache.
        model = torch.hub.load(self.local_repo, self.model_name, source = 'local', weights = self.weights_path)
        print("Loaded model from local weights", self.weights_path)
        self._model = model        
        return

