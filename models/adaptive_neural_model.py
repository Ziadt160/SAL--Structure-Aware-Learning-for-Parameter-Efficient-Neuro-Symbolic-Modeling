import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sympy
from typing import Dict, List, Optional, Tuple, Any
from .symbolic_neuron import MatrixSymbolicNode

class MatrixChain(nn.Module):
    """
    A sequence of MatrixSymbolicNodes representing a single 'Expert' or 'Chain'.
    """
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float = 0.1, allowed_tools: Optional[Dict[str, Any]] = None, fixed_ops: Optional[List[str]] = None):
        super().__init__()
        self.layers = nn.ModuleList()
        
        if fixed_ops:
            assert len(fixed_ops) == depth, f"Fixed ops length ({len(fixed_ops)}) must match depth ({depth})"
        # 1. Force the Input Layer to be Non-Linear
        starters = ['tanh', 'relu', 'sin', 'gaussian', 'sigmoid']
        
        # Override with fixed ops if provided
        op0 = fixed_ops[0] if fixed_ops else random.choice(starters)
        self.layers.append(MatrixSymbolicNode(input_dim, hidden_dim, op0, dropout=dropout, allowed_tools=allowed_tools))
        
        # 2. Force Hidden Layers to be Non-Linear
        for i in range(depth - 1):
            op = fixed_ops[i+1] if fixed_ops else random.choice(starters)
            self.layers.append(MatrixSymbolicNode(hidden_dim, hidden_dim, op, dropout=dropout, allowed_tools=allowed_tools))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def export_formula(self, input_vars: List[str]) -> str:
        """
        Reconstructs the symbolic formula of the network using SymPy.
        """
        assert len(input_vars) == self.layers[0].linear.in_features, "Input vars count must match input dim"
        
        # Initialize symbolic state
        sym_state = [sympy.Symbol(v) for v in input_vars]
        
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, MatrixSymbolicNode): continue
            
            # 1. Linear Transform (W * x + b)
            W = layer.linear.weight.detach().cpu().numpy()
            b = layer.linear.bias.detach().cpu().numpy()
            
            new_state = []
            for out_idx in range(layer.linear.out_features):
                # Dot product
                term = b[out_idx]
                for in_idx in range(layer.linear.in_features):
                    term += W[out_idx, in_idx] * sym_state[in_idx]
                new_state.append(term)
            
            # 2. Activation Function (Symbolic)
            op_name = layer.op_name
            
            # Recursive helper to apply op
            def apply_op(val, op):
                if op == 'identity': return val
                if op == 'tanh': return sympy.tanh(val)
                if op == 'relu': return sympy.Max(0, val)
                if op == 'sigmoid': return 1 / (1 + sympy.exp(-val))
                if op == 'sin': return sympy.sin(val * sympy.pi) # Note the PI scaling
                if op == 'cos': return sympy.cos(val)
                if op == 'gaussian': return sympy.exp(-val**2)
                if op == 'square': return val**2
                if op == 'leaky_relu': return sympy.Piecewise((val, val >= 0), (0.01 * val, True))
                
                # Composites (parsed by name)
                if '_of_' in op:
                    parts = op.split('_of_')
                    return apply_op(apply_op(val, parts[1]), parts[0])
                if '_x_' in op:
                    parts = op.split('_x_')
                    return apply_op(val, parts[0]) * apply_op(val, parts[1])
                if '_plus_' in op:
                    parts = op.split('_plus_')
                    return apply_op(val, parts[0]) + apply_op(val, parts[1])

                return val # Default/Unknown
            
            sym_state = [apply_op(s, op_name) for s in new_state]
            
        return str(sym_state)

class MatrixGGLEN(nn.Module):
    """
    Standard GAI Architecture: An ensemble of parallel chains summed together.
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int = 1, 
                 hidden_dim: int = 32, 
                 num_chains: int = 2, 
                 chain_depth: int = 3, 
                 dropout: float = 0.0, 
                 allowed_tools: Optional[Dict[str, Any]] = None,
                 fixed_structure: Optional[List[List[str]]] = None):
        super().__init__()
        
        if fixed_structure:
             assert len(fixed_structure) == num_chains, "Fixed structure must define ops for all chains"
        
        # Store for resizing/rebuilding
        self.hparams_dict = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dim': hidden_dim,
            'num_chains': num_chains,
            'chain_depth': chain_depth,
            'dropout': dropout,
            'allowed_tools': allowed_tools
        }
        
        self.chains = nn.ModuleList([
            MatrixChain(input_dim, hidden_dim, chain_depth, dropout, allowed_tools, fixed_ops=fixed_structure[i] if fixed_structure else None) 
            for i in range(num_chains)
        ])
        self.final = nn.Linear(hidden_dim, output_dim)

    def resize(self, new_hidden_dim: int):
        """Creates a new instance with the same structure but new hidden dim."""
        new_params = self.hparams_dict.copy()
        new_params['hidden_dim'] = new_hidden_dim
        new_params['fixed_structure'] = self.get_structure()
        return MatrixGGLEN(**new_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chain_sum = 0
        for chain in self.chains:
            chain_sum = chain_sum + chain(x)
        return self.final(chain_sum)

    def evolve_structure(self, history_tracker: Dict, logger_prefix: str = "GAI", strategy: str = 'importance') -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """
        Selects a node to mutate.
        """
        candidates = []
        for c_idx, chain in enumerate(self.chains):
            for l_idx, node in enumerate(chain.layers): 
                score = node.get_importance()
                candidates.append((score, c_idx, l_idx))
                node.reset_metrics()
        
        if strategy == 'importance':
            # Target least important nodes
            candidates.sort(key=lambda x: x[0])
        elif strategy == 'random':
            random.shuffle(candidates)
        
        for score, c_id, l_id in candidates:
            forbidden = history_tracker.get((c_id, l_id), set())
            
            log_strat = f"Imp: {score:.4f}" if strategy == 'importance' else "Random"
            print(f"  [{logger_prefix}] [Evolution] Targeting Chain {c_id} Layer {l_id} ({log_strat})...", flush=True)
                
            node = self.chains[c_id].layers[l_id] 
            success, old_op = node.mutate(forbidden, logger_prefix)
            if success: 
                return c_id, l_id, old_op
                
        return None, None, None

    def get_structure(self) -> List[List[str]]:
        """Returns the current list of operations for all chains."""
        structure = []
        for chain in self.chains:
            ops = [layer.op_name for layer in chain.layers] # type: ignore
            structure.append(ops)
        return structure

class GatedMatrixGGLEN(nn.Module):
    """
    Mixture of Experts (MoE) GAI Architecture.
    """
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int = 1, 
                 hidden_dim: int = 32, 
                 num_chains: int = 2, 
                 chain_depth: int = 3, 
                 dropout: float = 0.0, 
                 allowed_tools: Optional[Dict[str, Any]] = None,
                 fixed_structure: Optional[List[List[str]]] = None):
        super().__init__()
        
        self.hparams_dict = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dim': hidden_dim,
            'num_chains': num_chains,
            'chain_depth': chain_depth,
            'dropout': dropout,
            'allowed_tools': allowed_tools
        }
        
        # 1. Experts
        self.chains = nn.ModuleList([
            MatrixChain(input_dim, hidden_dim, chain_depth, dropout, allowed_tools, fixed_ops=fixed_structure[i] if fixed_structure else None) 
            for i in range(num_chains)
        ])
        
        # 2. Gate (Manager)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_chains)
        )
        
        # 3. Aggregation
        self.final = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A. Gate
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=1) # [B, N]
        
        # B. Experts
        chain_outputs_list = [chain(x) for chain in self.chains] 
        expert_outputs = torch.stack(chain_outputs_list, dim=1) # [B, N, H]
        
        # C. Weighted Sum
        weights_expanded = gate_weights.unsqueeze(2) # [B, N, 1]
        weighted_experts = weights_expanded * expert_outputs # [B, N, H]
        consensus = torch.sum(weighted_experts, dim=1) # [B, H]
        
        # D. Final
        consensus = self.dropout_layer(consensus)
        return self.final(consensus)

    def get_structure(self) -> List[List[str]]:
        """Returns the current list of operations for all chains."""
        structure = []
        for chain in self.chains:
            ops = [layer.op_name for layer in chain.layers] # type: ignore
            structure.append(ops)
        return structure

    def resize(self, new_hidden_dim: int):
        """Creates a new instance with the same structure but new hidden dim."""
        new_params = self.hparams_dict.copy()
        new_params['hidden_dim'] = new_hidden_dim
        new_params['fixed_structure'] = self.get_structure()
        return GatedMatrixGGLEN(**new_params)

    def evolve_structure(self, history_tracker: Dict, logger_prefix: str = "GAI-MoE", strategy: str = 'importance') -> Tuple[Optional[int], Optional[int], Optional[str]]:
        candidates = []
        for c_idx, chain in enumerate(self.chains):
            for l_idx, node in enumerate(chain.layers):
                score = node.get_importance()
                candidates.append((score, c_idx, l_idx))
                node.reset_metrics()
        
        if strategy == 'importance':
            candidates.sort(key=lambda x: x[0])
        elif strategy == 'random':
            random.shuffle(candidates)
            
        for score, c_id, l_id in candidates:
            forbidden = history_tracker.get((c_id, l_id), set())
            
            log_strat = f"Imp: {score:.4f}" if strategy == 'importance' else "Random"
            print(f"  [{logger_prefix}] [Evolution] Targeting Expert {c_id} Layer {l_id} ({log_strat})...", flush=True)
                
            node = self.chains[c_id].layers[l_id]
            success, old_op = node.mutate(forbidden, logger_prefix)
            if success: return c_id, l_id, old_op
            
        return None, None, None

class SymbolicJudge(nn.Module):
    """
    Automated Metric Learning Module.
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, depth: int = 2, dropout: float = 0.0, allowed_tools: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.chain = MatrixChain(input_dim, hidden_dim, depth, dropout, allowed_tools)
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.chain(x)
        score = self.final(features)
        return score 

    def evolve_structure(self, history_tracker: Dict, logger_prefix: str = "Judge", strategy: str = 'importance') -> Tuple[Optional[int], Optional[int], Optional[str]]:
        candidates = []
        for l_idx, node in enumerate(self.chain.layers): 
            score = node.get_importance()
            candidates.append((score, 0, l_idx)) 
            node.reset_metrics()
        
        if strategy == 'importance':
            candidates.sort(key=lambda x: x[0])
        elif strategy == 'random':
            random.shuffle(candidates)
            
        for score, c_id, l_id in candidates:
            forbidden = history_tracker.get((c_id, l_id), set())
            print(f"  [{logger_prefix}] [Evolution] Targeting Layer {l_id} (Imp: {score:.4f})...", flush=True)
            node = self.chain.layers[l_id] 
            success, old_op = node.mutate(forbidden, logger_prefix)
            if success: return c_id, l_id, old_op
            
        return None, None, None
