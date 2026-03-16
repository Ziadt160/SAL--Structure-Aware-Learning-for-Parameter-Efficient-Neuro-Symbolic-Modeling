import torch
import torch.nn as nn
import random
from typing import Dict, Callable, Union, Any

# Type alias for activation functions
ActivationFunc = Callable[[torch.Tensor], torch.Tensor]
CompositeOpFunc = Callable[[ActivationFunc, ActivationFunc, str], ActivationFunc]

def create_composite_op(op1_func: ActivationFunc, op2_func: ActivationFunc, operation: str = 'compose') -> ActivationFunc:
    """Creates a new activation function by combining two existing ones with soft clipping."""
    
    def _soft_clip(t: torch.Tensor) -> torch.Tensor:
        # Soft clip to prevent INF/NaN, keeping gradient flow
        # Limit approx +/- 1,000,000
        return 1e6 * torch.tanh(t / 1e6)

    if operation == 'compose':
        return lambda x: _soft_clip(op2_func(op1_func(x)))
    elif operation == 'multiply':
        return lambda x: _soft_clip(op1_func(x) * op2_func(x))
    elif operation == 'add':
        return lambda x: _soft_clip(op1_func(x) + op2_func(x))
    # Default fallback
    return op1_func

# Predefined Activation Tools
ACTIVATIONS: Dict[str, Union[nn.Module, ActivationFunc]] = {
    'identity': lambda x: x,
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'sin': lambda x: torch.sin(x * torch.pi),
    'cos': lambda x: torch.cos(x),
    'gaussian': lambda x: torch.exp(-x**2),
    'square': lambda x: x**2,
    'leaky_relu': nn.LeakyReLU(),
}
