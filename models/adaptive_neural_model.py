"""Chain-ensemble architectures whose per-node operators are searchable."""

import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
from typing import Dict, List, Optional, Tuple, Any

from .symbolic_neuron import MatrixSymbolicNode
from .activations import to_sympy, SEARCH_BASIS

# Operators a freshly built node may start from. Kept non-linear so a chain
# is not born as a pure affine map.
STARTERS: List[str] = ['tanh', 'relu', 'sin', 'gaussian']

# export_formula expands into a flat sum of products; the term count grows as
# input_dim * hidden_dim ** (depth - 1). Refuse rather than hang.
MAX_FORMULA_TERMS = 200_000


class MatrixChain(nn.Module):
    """A sequence of MatrixSymbolicNodes -- one 'expert'."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 depth: int,
                 dropout: float = 0.1,
                 allowed_tools: Optional[Dict[str, Any]] = None,
                 fixed_ops: Optional[List[str]] = None,
                 rng: Optional[random.Random] = None,
                 importance_mode: str = 'taylor'):
        super().__init__()
        rng = rng or random
        if fixed_ops:
            depth = len(fixed_ops)

        self.layers = nn.ModuleList()
        for i in range(depth):
            op = fixed_ops[i] if fixed_ops else rng.choice(STARTERS)
            in_d = input_dim if i == 0 else hidden_dim
            self.layers.append(MatrixSymbolicNode(
                in_d, hidden_dim, op, dropout=dropout,
                allowed_tools=allowed_tools, importance_mode=importance_mode))

    @property
    def depth(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def append_identity_node(self, dropout: float = 0.0,
                             importance_mode: str = 'taylor'):
        """Append an exact identity node -- function-preserving in eval mode."""
        hidden = self.layers[-1].linear.out_features
        node = MatrixSymbolicNode(hidden, hidden, 'identity', dropout=dropout,
                                  allowed_tools=self.layers[-1].tools,
                                  importance_mode=importance_mode)
        node.make_identity()
        self.layers.append(node)

    def pop_node(self) -> bool:
        """Remove the last node. Not function-preserving."""
        if len(self.layers) <= 1:
            return False
        self.layers = nn.ModuleList(list(self.layers)[:-1])
        return True

    def export_formula(self, input_vars: List[str], include_clip: bool = True,
                       max_terms: int = MAX_FORMULA_TERMS) -> str:
        """Symbolic form of this chain's hidden output, as a list of expressions.

        Uses the same operator parse as `forward` (models.activations.to_sympy),
        so the expression is the function the network actually computes rather
        than a re-derivation that can drift out of sync.
        """
        n_in = self.layers[0].linear.in_features
        if len(input_vars) != n_in:
            raise ValueError(f"Expected {n_in} input vars, got {len(input_vars)}")
        self._check_formula_size(max_terms)

        state = [sympy.Symbol(v) for v in input_vars]
        for layer in self.layers:
            W = layer.linear.weight.detach().cpu().numpy()
            b = layer.linear.bias.detach().cpu().numpy()
            pre = []
            for o in range(layer.linear.out_features):
                terms = [sympy.Float(float(b[o]))]
                terms += [sympy.Float(float(W[o, i])) * state[i]
                          for i in range(layer.linear.in_features)]
                pre.append(sympy.Add(*terms))
            state = [to_sympy(layer.op_name, e, include_clip=include_clip)
                     for e in pre]
        return str(state)

    def _check_formula_size(self, max_terms: int):
        n_in = self.layers[0].linear.in_features
        hidden = self.layers[0].linear.out_features
        est = n_in * (hidden ** (len(self.layers) - 1))
        if est > max_terms:
            raise ValueError(
                f"Refusing to expand formula: ~{est:,} terms per hidden unit "
                f"(input_dim={n_in}, hidden_dim={hidden}, depth={len(self.layers)}). "
                f"Symbolic export is only tractable for small inputs; reduce "
                f"dimensions or raise max_terms deliberately."
            )


class _ChainEnsembleBase(nn.Module):
    """Shared search/serialisation behaviour for the chain ensembles."""

    def get_structure(self) -> List[List[str]]:
        return [[layer.op_name for layer in chain.layers] for chain in self.chains]

    def set_structure(self, structure: List[List[str]]):
        if len(structure) != len(self.chains):
            raise ValueError("Structure must have one op list per chain")
        for chain, ops in zip(self.chains, structure):
            if len(ops) != len(chain.layers):
                raise ValueError("Op list length must match chain depth")
            for layer, op in zip(chain.layers, ops):
                layer.set_op(op)

    def iter_nodes(self):
        for c, chain in enumerate(self.chains):
            for l, node in enumerate(chain.layers):
                yield c, l, node

    def node_count(self) -> int:
        return sum(len(chain.layers) for chain in self.chains)

    def set_tracking(self, flag: bool):
        """Enable/disable importance accumulation on every node.

        The hooks add two extra reductions plus .item() calls per node per
        backward pass, which is a large fraction of step time for small models.
        The search only needs importance for its final report, so it keeps
        tracking off and switches it on for one measurement pass.
        """
        for _, _, node in self.iter_nodes():
            node.track_gradients = flag

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- checkpointing ----------------------------------------------------
    def checkpoint(self) -> Dict[str, Any]:
        """A complete checkpoint.

        `op_name` is a plain Python string, not a parameter or buffer, so it
        is absent from state_dict(). Saving weights alone would therefore lose
        the discovered architecture entirely -- the one thing the search
        produces. Structure is stored alongside.
        """
        return {
            'state_dict': {k: v.detach().cpu().clone()
                           for k, v in self.state_dict().items()},
            'structure': self.get_structure(),
            'hparams': dict(self.hparams_dict),
            'class': type(self).__name__,
        }

    def save(self, path: str):
        torch.save(self.checkpoint(), path)

    @classmethod
    def from_checkpoint(cls, ckpt: Dict[str, Any]):
        hp = dict(ckpt['hparams'])
        hp['fixed_structure'] = ckpt['structure']
        model = cls(**hp)
        model.load_state_dict(ckpt['state_dict'])
        model.set_structure(ckpt['structure'])
        return model

    @classmethod
    def load(cls, path: str):
        return cls.from_checkpoint(torch.load(path, map_location='cpu'))

    # -- topology moves ---------------------------------------------------
    def deepen_chain(self, chain_idx: int) -> bool:
        """Append an identity node to one chain. Function-preserving (eval mode)."""
        chain = self.chains[chain_idx]
        chain.append_identity_node(
            dropout=self.hparams_dict.get('dropout', 0.0),
            importance_mode=self.hparams_dict.get('importance_mode', 'taylor'))
        return True

    def shrink_chain(self, chain_idx: int) -> bool:
        return self.chains[chain_idx].pop_node()

    def prune_chain(self, chain_idx: int) -> bool:
        if len(self.chains) <= 1:
            return False
        keep = [c for i, c in enumerate(self.chains) if i != chain_idx]
        self.chains = nn.ModuleList(keep)
        self.hparams_dict['num_chains'] = len(keep)
        self._on_chain_removed(chain_idx)
        return True

    def _on_chain_removed(self, chain_idx: int):
        pass

    def resize(self, new_hidden_dim: int):
        params = dict(self.hparams_dict)
        params['hidden_dim'] = new_hidden_dim
        params['fixed_structure'] = self.get_structure()
        params['num_chains'] = len(self.chains)
        return type(self)(**params)

    # -- legacy random search --------------------------------------------
    def evolve_structure(self, history_tracker: Dict, logger_prefix: str = "GAI",
                         strategy: str = 'importance',
                         rng: Optional[random.Random] = None,
                         reset_weights: bool = True
                         ) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """Pick one node and randomly mutate it. Returns (chain, layer, old_op).

        reset_weights=True re-initialises the mutated node near identity, which
        is what the original code always did -- the flag existed on mutate() but
        evolve_structure never passed it, so 'transfer the weights into the new
        operator instead' was unreachable and had never been measured. Only the
        mutated node is ever touched either way; the rest of the network keeps
        its weights."""
        rng = rng or random
        candidates = []
        for c, l, node in self.iter_nodes():
            candidates.append((node.get_importance(), c, l))
            node.reset_metrics()

        if strategy == 'importance':
            candidates.sort(key=lambda t: t[0])
        elif strategy == 'random':
            rng.shuffle(candidates)

        for score, c_id, l_id in candidates:
            forbidden = history_tracker.get((c_id, l_id), set())
            tag = f"Imp: {score:.3e}" if strategy == 'importance' else "Random"
            print(f"  [{logger_prefix}] [Evolution] Targeting Chain {c_id} "
                  f"Layer {l_id} ({tag})...", flush=True)
            ok, old_op = self.chains[c_id].layers[l_id].mutate(
                forbidden, logger_prefix, reset_weights=reset_weights, rng=rng)
            if ok:
                return c_id, l_id, old_op
        return None, None, None


class MatrixGGLEN(_ChainEnsembleBase):
    """Ensemble of parallel chains, summed, then a linear read-out."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 32,
                 num_chains: int = 2,
                 chain_depth: int = 3,
                 dropout: float = 0.0,
                 allowed_tools: Optional[Dict[str, Any]] = None,
                 fixed_structure: Optional[List[List[str]]] = None,
                 rng: Optional[random.Random] = None,
                 importance_mode: str = 'taylor',
                 readout: str = 'sum'):
        super().__init__()
        if fixed_structure:
            num_chains = len(fixed_structure)
        if readout not in ('sum', 'concat'):
            raise ValueError("readout must be 'sum' or 'concat'")

        self.hparams_dict = {
            'input_dim': input_dim, 'output_dim': output_dim,
            'hidden_dim': hidden_dim, 'num_chains': num_chains,
            'chain_depth': chain_depth, 'dropout': dropout,
            'allowed_tools': allowed_tools, 'importance_mode': importance_mode,
            'readout': readout,
        }
        self.readout = readout
        self.chains = nn.ModuleList([
            MatrixChain(input_dim, hidden_dim, chain_depth, dropout,
                        allowed_tools,
                        fixed_ops=fixed_structure[i] if fixed_structure else None,
                        rng=rng, importance_mode=importance_mode)
            for i in range(num_chains)
        ])
        self.final = nn.Linear(self._readout_width(num_chains, hidden_dim),
                               output_dim)

    def _readout_width(self, num_chains: int, hidden_dim: int) -> int:
        return hidden_dim if self.readout == 'sum' else num_chains * hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [chain(x) for chain in self.chains]
        if self.readout == 'concat':
            return self.final(torch.cat(outs, dim=1))
        total = outs[0]
        for o in outs[1:]:
            total = total + o
        return self.final(total)

    def add_chain(self, copy_from: int = -1) -> bool:
        """Add an expert whose output is exactly zero -- function-preserving.

        Ops are copied from an existing chain (duplicate-and-diverge) rather
        than drawn at random, so the move is deterministic.
        """
        hp = self.hparams_dict
        template = self.get_structure()[copy_from]
        chain = MatrixChain(hp['input_dim'], hp['hidden_dim'], len(template),
                            hp['dropout'], hp['allowed_tools'],
                            fixed_ops=list(template),
                            importance_mode=hp.get('importance_mode', 'taylor'))
        chain.layers[-1].zero_out()   # emits 0 -> the aggregate is unchanged
        self.chains.append(chain)
        hp['num_chains'] = len(self.chains)
        if self.readout == 'concat':
            self._resize_readout_columns(insert_at=len(self.chains) - 1)
        return True

    def _resize_readout_columns(self, insert_at: Optional[int] = None,
                                remove_at: Optional[int] = None):
        """Grow/shrink the concat read-out by one chain's block of columns."""
        h = self.hparams_dict['hidden_dim']
        old = self.final
        width = self._readout_width(len(self.chains), h)
        new = nn.Linear(width, old.out_features)
        with torch.no_grad():
            new.weight.zero_()
            new.bias.copy_(old.bias)
            if insert_at is not None:
                # new chain contributes nothing yet -> its columns stay zero
                new.weight[:, :insert_at * h].copy_(old.weight[:, :insert_at * h])
            elif remove_at is not None:
                keep = [j for j in range(old.in_features)
                        if not (remove_at * h <= j < (remove_at + 1) * h)]
                new.weight.copy_(old.weight[:, keep])
        self.final = new

    def _on_chain_removed(self, chain_idx: int):
        if self.readout == 'concat':
            self._resize_readout_columns(remove_at=chain_idx)

    def export_formula(self, input_vars: List[str], include_clip: bool = True,
                       max_terms: int = MAX_FORMULA_TERMS) -> str:
        """Symbolic form of the WHOLE model: sum over chains, then read-out.

        The previous version exported chains[0] only and dropped both the
        cross-chain sum and the final linear layer, so the printed formula was
        not the model's function even when it rendered.
        """
        n_in = self.hparams_dict['input_dim']
        if len(input_vars) != n_in:
            raise ValueError(f"Expected {n_in} input vars, got {len(input_vars)}")
        for chain in self.chains:
            chain._check_formula_size(max_terms)

        syms = [sympy.Symbol(v) for v in input_vars]
        per_chain = []
        for chain in self.chains:
            state = list(syms)
            for layer in chain.layers:
                W = layer.linear.weight.detach().cpu().numpy()
                b = layer.linear.bias.detach().cpu().numpy()
                pre = []
                for o in range(layer.linear.out_features):
                    terms = [sympy.Float(float(b[o]))]
                    terms += [sympy.Float(float(W[o, i])) * state[i]
                              for i in range(layer.linear.in_features)]
                    pre.append(sympy.Add(*terms))
                state = [to_sympy(layer.op_name, e, include_clip=include_clip)
                         for e in pre]
            per_chain.append(state)

        if self.readout == 'concat':
            hidden_total = [e for state in per_chain for e in state]
        else:
            hidden_total = list(per_chain[0])
            for state in per_chain[1:]:
                hidden_total = [a + b for a, b in zip(hidden_total, state)]

        Wf = self.final.weight.detach().cpu().numpy()
        bf = self.final.bias.detach().cpu().numpy()
        outs = []
        for o in range(self.final.out_features):
            terms = [sympy.Float(float(bf[o]))]
            terms += [sympy.Float(float(Wf[o, h])) * hidden_total[h]
                      for h in range(len(hidden_total))]
            outs.append(sympy.Add(*terms))
        return str(outs)


class GatedMatrixGGLEN(_ChainEnsembleBase):
    """Mixture of experts: a gate softmax-weights the chain outputs."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 32,
                 num_chains: int = 2,
                 chain_depth: int = 3,
                 dropout: float = 0.0,
                 allowed_tools: Optional[Dict[str, Any]] = None,
                 fixed_structure: Optional[List[List[str]]] = None,
                 rng: Optional[random.Random] = None,
                 importance_mode: str = 'taylor'):
        super().__init__()
        if fixed_structure:
            num_chains = len(fixed_structure)

        self.hparams_dict = {
            'input_dim': input_dim, 'output_dim': output_dim,
            'hidden_dim': hidden_dim, 'num_chains': num_chains,
            'chain_depth': chain_depth, 'dropout': dropout,
            'allowed_tools': allowed_tools, 'importance_mode': importance_mode,
        }
        self.chains = nn.ModuleList([
            MatrixChain(input_dim, hidden_dim, chain_depth, dropout,
                        allowed_tools,
                        fixed_ops=fixed_structure[i] if fixed_structure else None,
                        rng=rng, importance_mode=importance_mode)
            for i in range(num_chains)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_chains),
        )
        self.final = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gate(x), dim=1)              # [B, N]
        experts = torch.stack([c(x) for c in self.chains], 1)  # [B, N, H]
        consensus = torch.sum(weights.unsqueeze(2) * experts, dim=1)
        return self.final(self.dropout_layer(consensus))

    def add_chain(self, copy_from: int = -1) -> bool:
        """Add an expert, gated almost entirely off.

        Only APPROXIMATELY function-preserving: the new gate logit is set to
        -30, giving it a softmax weight around 1e-13, which perturbs the other
        experts' normalised weights by the same order (well inside float32
        resolution).
        """
        hp = self.hparams_dict
        template = self.get_structure()[copy_from]
        chain = MatrixChain(hp['input_dim'], hp['hidden_dim'], len(template),
                            hp['dropout'], hp['allowed_tools'],
                            fixed_ops=list(template),
                            importance_mode=hp.get('importance_mode', 'taylor'))
        chain.layers[-1].zero_out()
        self.chains.append(chain)
        hp['num_chains'] = len(self.chains)
        self._grow_gate(len(self.chains))
        return True

    def _grow_gate(self, num_chains: int):
        """Extend the gate to `num_chains` logits, gating new experts off."""
        old = self.gate[-1]
        new = nn.Linear(old.in_features, num_chains)
        with torch.no_grad():
            new.weight.zero_()
            new.bias.fill_(-30.0)
            n = min(old.out_features, num_chains)
            new.weight[:n].copy_(old.weight[:n])
            new.bias[:n].copy_(old.bias[:n])
        self.gate[-1] = new

    def _on_chain_removed(self, chain_idx: int):
        """Drop the pruned expert's gate row, keeping the others aligned.

        Copying the first N rows would silently reassign experts when a middle
        chain is pruned (pruning chain 1 of 3 must keep gate rows 0 and 2, not
        rows 0 and 1).
        """
        old = self.gate[-1]
        keep = [i for i in range(old.out_features) if i != chain_idx]
        new = nn.Linear(old.in_features, len(keep))
        with torch.no_grad():
            new.weight.copy_(old.weight[keep])
            new.bias.copy_(old.bias[keep])
        self.gate[-1] = new


class SymbolicJudge(nn.Module):
    """Scores whether an input lies on a learned manifold."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, depth: int = 2,
                 dropout: float = 0.0,
                 allowed_tools: Optional[Dict[str, Any]] = None,
                 rng: Optional[random.Random] = None):
        super().__init__()
        self.chain = MatrixChain(input_dim, hidden_dim, depth, dropout,
                                 allowed_tools, rng=rng)
        self.final = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.chain(x))

    def get_structure(self) -> List[List[str]]:
        return [[layer.op_name for layer in self.chain.layers]]

    def evolve_structure(self, history_tracker: Dict, logger_prefix: str = "Judge",
                         strategy: str = 'importance',
                         rng: Optional[random.Random] = None,
                         reset_weights: bool = True
                         ) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        rng = rng or random
        candidates = []
        for l, node in enumerate(self.chain.layers):
            candidates.append((node.get_importance(), 0, l))
            node.reset_metrics()

        if strategy == 'importance':
            candidates.sort(key=lambda t: t[0])
        elif strategy == 'random':
            rng.shuffle(candidates)

        for score, c_id, l_id in candidates:
            forbidden = history_tracker.get((c_id, l_id), set())
            print(f"  [{logger_prefix}] [Evolution] Targeting Layer {l_id} "
                  f"(Imp: {score:.3e})...", flush=True)
            ok, old_op = self.chain.layers[l_id].mutate(
                forbidden, logger_prefix, reset_weights=reset_weights, rng=rng)
            if ok:
                return c_id, l_id, old_op
        return None, None, None
