"""
A quantum circuit wearing the interface training/structure_search.py expects.

The point of this file is that structure_search is NOT modified. If the search
machinery is general -- the probe protocol, exhaustive enumeration, restarts
after selection, the sweep-revert guard -- then swapping the operator basis for
a gate basis should be a matter of providing the same handful of methods.

States are carried as real tensors of shape [batch, 2 * 2**n] holding
[Re | Im], because MSELoss does not accept complex input. The circuit itself is
complex throughout.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .quantum_ops import (CDTYPE, GATE_CLASSES, SEARCH_GATES, QuantumCircuit,
                          QuantumGateNode)


def to_real(psi: torch.Tensor) -> torch.Tensor:
    """[batch, 2**n] complex -> [batch, 2*2**n] real."""
    return torch.cat([psi.real, psi.imag], dim=1)


def to_complex(vec: torch.Tensor) -> torch.Tensor:
    """[batch, 2*2**n] real -> [batch, 2**n] complex."""
    d = vec.shape[1] // 2
    return torch.complex(vec[:, :d], vec[:, d:])


class QuantumCircuitModel(nn.Module):
    """Learns a target unitary from its action on states.

    forward(X) applies the circuit to each input state in X. With y = U_target|x>
    this becomes ordinary supervised regression, so the existing search loop,
    loss, and validation protocol all apply unchanged.
    """

    def __init__(self,
                 input_dim: int,              # 2 * 2**n_qubits, for interface parity
                 output_dim: int = None,
                 hidden_dim: int = 3,         # n_qubits (named for interface parity)
                 num_chains: int = 1,
                 chain_depth: int = 6,
                 dropout: float = 0.0,
                 allowed_tools: Optional[Dict] = None,
                 fixed_structure: Optional[List[List[str]]] = None,
                 rng=None,
                 importance_mode: str = 'taylor',
                 readout: str = 'sum'):
        super().__init__()
        n_qubits = hidden_dim
        if fixed_structure:
            num_chains = len(fixed_structure)
        if num_chains != 1:
            raise ValueError("quantum model uses a single circuit; num_chains=1")

        self.n_qubits = n_qubits
        self.hparams_dict = {
            'input_dim': input_dim, 'output_dim': output_dim,
            'hidden_dim': n_qubits, 'num_chains': 1,
            'chain_depth': chain_depth, 'dropout': 0.0,
            'allowed_tools': None, 'importance_mode': importance_mode,
            'readout': readout,
        }
        gen = torch.Generator()
        gen.manual_seed(rng.randint(0, 2 ** 31 - 1) if rng is not None else 0)
        self.chains = nn.ModuleList([
            QuantumCircuit(n_qubits, chain_depth,
                           fixed_ops=fixed_structure[0] if fixed_structure else None,
                           generator=gen)])

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        psi = to_complex(x.to(torch.float64))
        u = self.chains[0].unitary()
        out = psi @ u.transpose(0, 1)
        return to_real(out).to(x.dtype)

    def unitary(self) -> torch.Tensor:
        return self.chains[0].unitary()

    # -- interface for structure_search -----------------------------------
    def get_structure(self) -> List[List[str]]:
        return [[l.op_name for l in self.chains[0].layers]]

    def set_structure(self, structure: List[List[str]]):
        ops = structure[0]
        if len(ops) != len(self.chains[0].layers):
            raise ValueError("op list length must match circuit depth")
        for layer, op in zip(self.chains[0].layers, ops):
            layer.set_op(op)

    def iter_nodes(self):
        for l, node in enumerate(self.chains[0].layers):
            yield 0, l, node

    def node_count(self) -> int:
        return len(self.chains[0].layers)

    def param_count(self) -> int:
        """Trainable angles only. A slot holding a non-parameterised gate (I, H,
        CNOT, CZ) still owns a theta, so this counts the angles that actually
        affect the circuit -- otherwise every candidate looks the same size."""
        from .quantum_ops import GATE_BASIS
        return sum(1 for _, _, nd in self.iter_nodes()
                   if GATE_BASIS[nd.op_name][1])

    def set_tracking(self, flag: bool):
        for _, _, node in self.iter_nodes():
            node.track_gradients = flag

    def resize(self, new_hidden_dim: int):
        params = dict(self.hparams_dict)
        params['hidden_dim'] = new_hidden_dim
        params['fixed_structure'] = self.get_structure()
        params['num_chains'] = 1
        return QuantumCircuitModel(**params)

    def gate_count(self) -> int:
        """Non-identity gates -- the quantum notion of model size."""
        return sum(1 for _, _, nd in self.iter_nodes() if nd.op_name != 'i')

    def two_qubit_count(self) -> int:
        """Entangling gates: the expensive resource on real hardware."""
        return sum(1 for _, _, nd in self.iter_nodes()
                   if nd.op_name in ('cnot', 'cz'))

    def structure_classes(self) -> List[str]:
        return [GATE_CLASSES[op] for op in self.get_structure()[0]]


# --------------------------------------------------------------------------
def random_state_batch(n_qubits: int, batch: int, generator=None) -> torch.Tensor:
    """Haar-ish random pure states: complex Gaussian, normalised."""
    d = 2 ** n_qubits
    re = torch.randn(batch, d, dtype=torch.float64, generator=generator)
    im = torch.randn(batch, d, dtype=torch.float64, generator=generator)
    psi = torch.complex(re, im)
    psi = psi / psi.abs().pow(2).sum(1, keepdim=True).sqrt()
    return psi


def unitary_learning_task(n_qubits: int, target_ops: List[str], n_states: int,
                          seed: int = 0):
    """Inputs = random states, targets = U_target applied to them.

    The target unitary is built from a KNOWN gate sequence, so an exact solution
    is guaranteed to exist inside the search space at this depth. That makes it
    the direct analogue of the sin(pi*x^2)+x^2 recovery test: we can ask whether
    the search finds an answer we know is reachable, rather than only whether it
    beats a baseline.
    """
    g = torch.Generator().manual_seed(seed)
    target = QuantumCircuit(n_qubits, len(target_ops), fixed_ops=target_ops,
                            generator=g)
    with torch.no_grad():
        u = target.unitary()
    psi = random_state_batch(n_qubits, n_states, g)
    with torch.no_grad():
        out = psi @ u.transpose(0, 1)
    X = to_real(psi).numpy()
    Y = to_real(out).numpy()
    return X, Y, u, target.state_dict()


def process_fidelity(u: torch.Tensor, v: torch.Tensor) -> float:
    """|Tr(U^dag V)| / dim -- 1.0 means equal up to global phase."""
    d = u.shape[0]
    return float((torch.trace(u.conj().transpose(0, 1) @ v).abs() / d).item())
