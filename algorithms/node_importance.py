import torch
import torch.nn as nn

def calculate_node_importance(node: nn.Module) -> float:
    """
    Computes node importance based on accumulated gradient norms.
    """
    call_count = getattr(node, 'call_count', 0)
    grad_accumulator = getattr(node, 'grad_accumulator', 0.0)
    
    if call_count == 0: 
        return 0.0
    return grad_accumulator / call_count

def reset_importance_metrics(node: nn.Module):
    """
    Resets the statistical counters for importance scoring.
    """
    if hasattr(node, 'grad_accumulator'):
        node.grad_accumulator = 0.0
    if hasattr(node, 'call_count'):
        node.call_count = 0
