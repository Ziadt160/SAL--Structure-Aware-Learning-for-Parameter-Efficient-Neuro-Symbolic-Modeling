import random
import torch
import torch.nn as nn
from typing import Dict, Optional, Set, Any, Tuple

def perform_structural_mutation(node: nn.Module, forbidden_ops: Optional[Set[str]] = None, logger_prefix: str = "GAI") -> Tuple[bool, Optional[str]]:
    """
    Extracted logic for mutating the activation function of a symbolic node.
    """
    if forbidden_ops is None: forbidden_ops = set()
    
    # Access tools and op_name from the node
    tools = getattr(node, 'tools', {})
    current_op = getattr(node, 'op_name', 'identity')
    
    # Filter available choices
    choices = set(tools.keys()) - {current_op} - forbidden_ops
    
    if not choices: 
        return False, None
    
    old_op = current_op
    new_op = random.choice(list(choices))
    
    # Update the node
    node.op_name = new_op
    print(f"    [{logger_prefix}] -> Mutating: {old_op.upper()} ===> {new_op.upper()}", flush=True)
    
    # Weight reset logic (to mitigate gradient shock)
    with torch.no_grad():
        if hasattr(node, 'linear'):
            nn.init.uniform_(node.linear.weight, -0.05, 0.05)
            nn.init.zeros_(node.linear.bias)
            if node.linear.in_features == node.linear.out_features:
                node.linear.weight.add_(torch.eye(node.linear.in_features) * 0.9)
            
    return True, old_op
