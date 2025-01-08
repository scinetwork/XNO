import torch
import torch.nn as nn

class ActivationWrapper(nn.Module):
    """
    Wraps any callable (e.g., built-in function, lambda, or torch.nn.functional)
    so it can be used as an nn.Module.
    """
    def __init__(self, activation_func):
        super().__init__()
        self.activation_func = activation_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation_func(x)