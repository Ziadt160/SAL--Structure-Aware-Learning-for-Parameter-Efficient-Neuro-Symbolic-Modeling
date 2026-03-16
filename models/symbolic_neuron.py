import torch
import torch.nn as nn
import random
from typing import Dict, Optional, Set, Any, Tuple
from .activations import ACTIVATIONS, create_composite_op

class MatrixSymbolicNode(nn.Module):
    """
    A single node in the Matrix Chain. 
    It applies a specific activation function (tool) and can be evolved.
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 op_name: str = 'identity', 
                 track_gradients: bool = True, 
                 dropout: float = 0.0, 
                 allowed_tools: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.op_name = op_name
        self.linear = nn.Linear(in_dim, out_dim)
        
        # Initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        self.dropout = nn.Dropout(dropout)
        self.track_gradients = track_gradients
        
        # Metrics for evolution
        self.grad_accumulator = 0.0
        self.call_count = 0
        
        # Tools available for mutation
        self.tools = allowed_tools if allowed_tools is not None else ACTIVATIONS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        z = self.dropout(z)
        
        # Apply symbolic tool
        if self.op_name in self.tools:
            act = self.tools[self.op_name]
            out = act(z) if isinstance(act, nn.Module) else act(z)
        else:
            out = z
            
        # Hook for importance scoring
        if self.training and self.track_gradients and out.requires_grad:
            out.retain_grad()
            self._register_hook(out)
            
        return out

    def _register_hook(self, tensor: torch.Tensor):
        def hook(grad):
            self.grad_accumulator += grad.norm().item()
            self.call_count += 1
        tensor.register_hook(hook)

    def get_importance(self) -> float:
        """Returns the average gradient norm as an importance score."""
        if self.call_count == 0: return 0.0
        return self.grad_accumulator / self.call_count

    def reset_metrics(self):
        self.grad_accumulator = 0.0
        self.call_count = 0

    def mutate(self, forbidden_ops: Optional[Set[str]] = None, logger_prefix: str = "GAI") -> Tuple[bool, Optional[str]]:
        """
        Mutates the activation function of this node.
        Returns (True, old_op) if successful, (False, None) otherwise.
        """
        if forbidden_ops is None: forbidden_ops = set()
        
        # Filter available choices
        choices = set(self.tools.keys()) - {self.op_name} - forbidden_ops
        
        # Chance to discover a NEW composite tool
        if random.random() < 0.05:
            self._try_discover_tool(logger_prefix, choices)
        
        if not choices: 
            return False, None
        
        old_op = self.op_name
        self.op_name = random.choice(list(choices))
        print(f"    [{logger_prefix}] -> Mutating: {old_op.upper()} ===> {self.op_name.upper()}", flush=True)
        
        # Reset weights to near-identity or small values to mitigate 'gradient shock'
        with torch.no_grad():
            nn.init.uniform_(self.linear.weight, -0.05, 0.05)
            nn.init.zeros_(self.linear.bias)
            
            # If dimensions align, add identity bias to pass signal through
            if self.linear.in_features == self.linear.out_features:
                self.linear.weight.add_(torch.eye(self.linear.in_features) * 0.9)
            
        return True, old_op

    def _try_discover_tool(self, logger_prefix: str, choices: Set[str]):
        """Helper to create new composite tools."""
        
        # Enforce Tool Limit (Max 25)
        MAX_TOOLS = 25
        if len(self.tools) >= MAX_TOOLS:
             # Identify custom tools (not in base ACTIVATIONS)
             custom_keys = [k for k in self.tools.keys() if k not in ACTIVATIONS]
             if custom_keys:
                 # Prune a random custom tool to make space
                 to_remove = random.choice(custom_keys)
                 del self.tools[to_remove]
                 # Also remove from valid choices if present
                 if to_remove in choices: choices.remove(to_remove)
                 print(f"    [{logger_prefix}] [Pruning] Removed unused tool '{to_remove}' to save memory.", flush=True)

        base_ops = list(self.tools.keys())
        base_ops = list(self.tools.keys())
        op1 = random.choice(base_ops)
        op2 = random.choice(base_ops)

        # Randomly select composition type
        comp_type = random.choice(['compose', 'multiply', 'add'])
        
        # Formatting name based on type
        if comp_type == 'compose':
            new_name = f"{op1}_of_{op2}"
        elif comp_type == 'multiply':
            new_name = f"{op1}_x_{op2}"
        elif comp_type == 'add':
            new_name = f"{op1}_plus_{op2}"
        
        if new_name not in self.tools:
            print(f"    [{logger_prefix}] [Discovery] Invented new composite tool: {new_name} ({comp_type})", flush=True)
            func1 = self.tools[op1]
            func2 = self.tools[op2]
            
            # Normalize to callables
            f1 = func1.forward if isinstance(func1, nn.Module) else func1
            f2 = func2.forward if isinstance(func2, nn.Module) else func2
            
            self.tools[new_name] = create_composite_op(f1, f2, operation=comp_type)
            choices.add(new_name)
