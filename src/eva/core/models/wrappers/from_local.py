"""Helper function from models defined with a function locally."""


from typing import Any, Callable, Dict

import jsonargparse
import torch
from torch import nn
from typing_extensions import override

from eva.core.models.wrappers import _utils, base


#Add lora code here
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank: int, alpha: float):
        super().__init__()

        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha
    
        # Freeze the original linear layer's weights
        # Not needed right now
        #self.linear.weight.requires_grad = False
        #if self.linear.bias is not None:
        #    self.linear.bias.requires_grad = False

        # LoRA's low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))

        #We will do this differently...
        #We will have A and B be the same shape, but have a frozen gaussian matrix between them
        

        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass
        original_output = self.linear(x)

        # LoRA forward pass: (x * A^T) * B^T
        lora_output = (self.alpha / self.rank) * (x @ self.lora_A.T @ self.lora_B.T)
        #print(self.lora_B)
        return original_output + lora_output

# Define LoRA hyperparameters
lora_rank = 8
lora_alpha = 16

def apply_lora_to_model(model: nn.Module, rank: int, alpha: float):
    for name, module in model.named_modules():
        # Target the q and v projection linear layers in attention blocks
        if isinstance(module, torch.nn.Linear):
            # Get the parent module
            parent_name = ".".join(name.split('.')[:-1])
            parent_module = model
            for part in parent_name.split('.'):
                parent_module = getattr(parent_module, part)
            # Replace the original linear layer with the LoRALinear wrapper
            loralin = LoRALinear(module, rank, alpha)
            setattr(parent_module, name.split('.')[-1], LoRALinear(module, rank, alpha))
            print(f"✅ Applied LoRA to: {name}", flush = True)

class ModelFromLocal(base.BaseModel[torch.Tensor, torch.Tensor]):
    """Wrapper class for models which are initialized from functions.

    This is helpful for initializing models in a `.yaml` configuration file.
    """

    def __init__(
        self,
        local_repo_path: str,
        model_name: str,
        checkpoint_path: str,
        apply_lora: bool = False
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
        for param in model.parameters():
            param.requires_grad = False
        #Set all parameters to off
        apply_lora_to_model(model, lora_rank, lora_alpha)

        #for param in model.parameters():
        #    print(param.requires_grad)

        #print(model, flush = True)

        #exit()


        return

