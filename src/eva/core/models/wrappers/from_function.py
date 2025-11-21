"""Helper function from models defined with a function."""

from pathlib import Path
from typing import Any, Callable, Dict

import sys
import torch
from torch import nn
from typing_extensions import override

from eva.core.models.wrappers import _utils, base


class ModelFromFunction(base.BaseModel[torch.Tensor, torch.Tensor]):
    """Wrapper class for models which are initialized from functions.

    This is helpful for initializing models in a `.yaml` configuration file.
    """

    def __init__(
        self,
        path: Callable[..., nn.Module],
        arguments: Dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
        transforms: Callable | None = None,
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
        super().__init__(transforms=transforms)

        self._path = path
        self._arguments = arguments
        self._checkpoint_path = checkpoint_path

        self.load_model()

    @override
    def load_model(self) -> None:
        if not self._checkpoint_path:
            raise ValueError("Model checkpoint is required; none was provided.")

        checkpoint_path = Path(self._checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        repo_root = Path(__file__).resolve().parents[5].parent
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        from importlib import import_module

        backbones = import_module("dinov3.hub.backbones")
        model_name = None
        if isinstance(self._arguments, dict):
            model_name = self._arguments.get("model")
        model_name = model_name or "dinov3_vith16plus"
        if not hasattr(backbones, model_name):
            raise ValueError(f"Unsupported backbone '{model_name}' for dinov3.")

        model_fn = getattr(backbones, model_name)
        model = model_fn(pretrained=False)
        _utils.load_model_weights(model, str(checkpoint_path))
        self._model = model
