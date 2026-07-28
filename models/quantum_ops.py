"""
Quantum gate basis for the structure search.

The mapping from the neural version, term for term:

    basis operator          ->  gate: RX(t), RY(t), RZ(t), H, CNOT, CZ, I
    node weights (W, b)     ->  rotation angle t, trained by gradient descent
    a_of_b   (compose)      ->  U_a . U_b        operator product  = DEPTH
    a_x_b    (multiply)     ->  U_a (tensor) U_b                   = WIDTH
    soft clip 1e6*tanh(..)  ->  unnecessary; unitarity bounds everything
    chain                   ->  a sub-circuit
    final linear read-out   ->  the measured observable / output states

Two things carry over for free. Unitarity removes the need for the soft-clipping
hack entirely -- no gate can blow up. And the equivalence-class analysis built
for activations maps onto gate identities: RZ.RY.RZ spans SU(2), so a run of
three arbitrary rotations on one wire is redundant with itself, exactly the way
sin/cos and tanh/sigmoid were redundant given the surrounding affine maps.

`a_plus_b` (linear combination of unitaries) is deliberately ABSENT from this
first version. LCU is not a native circuit operation: it needs an ancilla
register, PREPARE and SELECT oracles, and it only succeeds probabilistically.
Adding it makes the search multi-objective (fidelity against ancilla count
against success probability), and its natural home is algorithm synthesis rather
than variational ansatz design. That is the part of the idea whose soundness is
least established, so it is left out until the rest is shown to work.

Scope: the state vector is 2**n_qubits, so this is exact and fast to ~10 qubits
and infeasible beyond ~20 regardless of how good the search is. That ceiling is
the simulator's, not the method's.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

CDTYPE = torch.complex128

_I2 = torch.eye(2, dtype=CDTYPE)
_X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE)
_Z = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE)
_H = torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE) / math.sqrt(2)

# name -> (arity, parameterised?)
GATE_BASIS: Dict[str, Tuple[int, bool]] = {
    'i':    (1, False),
    'h':    (1, False),
    'rx':   (1, True),
    'ry':   (1, True),
    'rz':   (1, True),
    'cnot': (2, False),
    'cz':   (2, False),
}

# One representative per class, for the search. `i` is kept because "do nothing
# here" is a meaningful choice -- it is how the search shortens a circuit.
SEARCH_GATES: List[str] = ['i', 'h', 'rx', 'ry', 'rz', 'cnot', 'cz']

# Gate identities that make two names interchangeable given free neighbours,
# the direct analogue of EQUIVALENCE_CLASSES in models/activations.py.
GATE_CLASSES: Dict[str, str] = {
    'i': 'identity', 'h': 'basis_change',
    'rx': 'rotation', 'ry': 'rotation', 'rz': 'rotation',
    'cnot': 'entangler', 'cz': 'entangler',
}


def one_qubit_matrix(name: str, theta: torch.Tensor) -> torch.Tensor:
    """2x2 matrix for a single-qubit gate."""
    if name == 'i':
        return _I2.clone()
    if name == 'h':
        return _H.clone()
    half = (theta.to(torch.float64) / 2).to(CDTYPE)
    c, s = torch.cos(half), torch.sin(half)
    if name == 'rx':
        return torch.stack([torch.stack([c, -1j * s]),
                            torch.stack([-1j * s, c])])
    if name == 'ry':
        return torch.stack([torch.stack([c, -s]),
                            torch.stack([s, c])])
    if name == 'rz':
        e_m, e_p = torch.exp(-1j * half), torch.exp(1j * half)
        z = torch.zeros((), dtype=CDTYPE)
        return torch.stack([torch.stack([e_m, z]), torch.stack([z, e_p])])
    raise ValueError(f"not a single-qubit gate: {name!r}")


def embed_one(mat: torch.Tensor, wire: int, n: int) -> torch.Tensor:
    """Lift a 2x2 gate on `wire` to the full 2**n space."""
    out = torch.ones((1, 1), dtype=CDTYPE)
    for q in range(n):
        out = torch.kron(out, mat if q == wire else _I2)
    return out


def embed_two(name: str, control: int, target: int, n: int) -> torch.Tensor:
    """CNOT / CZ on (control, target), built by projector decomposition:
    |0><0|_c (tensor) I + |1><1|_c (tensor) op_t."""
    p0 = torch.tensor([[1, 0], [0, 0]], dtype=CDTYPE)
    p1 = torch.tensor([[0, 0], [0, 1]], dtype=CDTYPE)
    op = _X if name == 'cnot' else _Z

    term0 = torch.ones((1, 1), dtype=CDTYPE)
    term1 = torch.ones((1, 1), dtype=CDTYPE)
    for q in range(n):
        term0 = torch.kron(term0, p0 if q == control else _I2)
        if q == control:
            term1 = torch.kron(term1, p1)
        elif q == target:
            term1 = torch.kron(term1, op)
        else:
            term1 = torch.kron(term1, _I2)
    return term0 + term1


class QuantumGateNode(nn.Module):
    """One gate slot: a searchable gate type plus its trainable angle.

    Mirrors MatrixSymbolicNode's interface (`op_name` property with a
    recompiling setter, `snapshot`/`restore`) so training.structure_search can
    drive it without modification.
    """

    def __init__(self, n_qubits: int, wires: Tuple[int, ...],
                 op_name: str = 'i', generator: Optional[torch.Generator] = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.wires = tuple(wires)
        self.theta = nn.Parameter(
            torch.empty(1, dtype=torch.float64).uniform_(-math.pi, math.pi,
                                                         generator=generator))
        self._op_name = op_name
        self.track_gradients = False

    @property
    def op_name(self) -> str:
        return self._op_name

    @op_name.setter
    def op_name(self, name: str):
        self.set_op(name)

    def set_op(self, name: str):
        if name not in GATE_BASIS:
            raise ValueError(f"unknown gate {name!r}; known: {sorted(GATE_BASIS)}")
        self._op_name = name

    def matrix(self) -> torch.Tensor:
        arity, _ = GATE_BASIS[self._op_name]
        if arity == 1:
            return embed_one(one_qubit_matrix(self._op_name, self.theta[0]),
                             self.wires[0], self.n_qubits)
        # A two-qubit gate needs two distinct wires; degrade to identity if the
        # slot only has one, so every gate remains selectable at every slot.
        if len(self.wires) < 2 or self.wires[0] == self.wires[1]:
            return torch.eye(2 ** self.n_qubits, dtype=CDTYPE)
        return embed_two(self._op_name, self.wires[0], self.wires[1],
                         self.n_qubits)

    # -- interface expected by structure_search ---------------------------
    def snapshot(self) -> Dict:
        return {'op_name': self._op_name, 'theta': self.theta.detach().clone()}

    def restore(self, snap: Dict):
        with torch.no_grad():
            self.theta.copy_(snap['theta'])
        self.set_op(snap['op_name'])

    def get_importance(self) -> float:
        return 0.0

    def get_metrics(self) -> Dict[str, float]:
        return {'taylor': 0.0, 'gradnorm': 0.0, 'calls': 0}

    def reset_metrics(self):
        pass


class QuantumCircuit(nn.Module):
    """A sequence of gate slots -- the analogue of MatrixChain.

    Slots cycle over the wires so that every qubit is touched and adjacent
    pairs are available for entanglers, which fixes the LAYOUT and leaves only
    the gate TYPE to be searched. That is deliberately the same restriction the
    neural version has: one operator choice per layer, layout fixed.
    """

    def __init__(self, n_qubits: int, depth: int,
                 fixed_ops: Optional[List[str]] = None,
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        self.n_qubits = n_qubits
        if fixed_ops:
            depth = len(fixed_ops)
        # Start from random gates, not all-identity. An all-identity circuit is
        # a constant map with no parameter in the graph at all, so the first
        # backward pass has nothing to differentiate. This mirrors MatrixChain
        # starting from random non-linear operators rather than from identity.
        starters = ['rx', 'ry', 'rz', 'h', 'cnot']
        self.layers = nn.ModuleList()
        for i in range(depth):
            w = (i % n_qubits, (i + 1) % n_qubits)
            if fixed_ops:
                op = fixed_ops[i]
            else:
                j = int(torch.randint(len(starters), (1,),
                                      generator=generator).item())
                op = starters[j]
            self.layers.append(QuantumGateNode(n_qubits, w, op, generator))

    def unitary(self) -> torch.Tensor:
        """Product of the slot matrices -- `_of_` composition, i.e. depth."""
        u = torch.eye(2 ** self.n_qubits, dtype=CDTYPE)
        for layer in self.layers:
            u = layer.matrix() @ u
        return u
