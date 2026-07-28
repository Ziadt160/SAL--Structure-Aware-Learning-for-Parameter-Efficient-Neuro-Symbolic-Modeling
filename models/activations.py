"""
Operator library for Structure-Aware Learning.

Design note
-----------
An operator name is parsed ONCE into a small expression tree, and two
interpreters walk that same tree: one emits a torch callable (used by
`forward`), one emits a SymPy expression (used by `export_formula`).

This is deliberate. Previously the two paths parsed operator names
independently and disagreed -- `sin_of_square` executed as `sin(x)**2` but
exported as `sin(x**2)`. Sharing the parse makes that class of bug
unrepresentable: if `to_torch` and `to_sympy` ever diverge, the round-trip
test in tests/test_operators.py fails.

Grammar (no parentheses; precedence lowest to highest, left-associative):
    add      A_plus_B   ->  A(x) + B(x)
    multiply A_x_B      ->  A(x) * B(x)
    compose  A_of_B     ->  A(B(x))
Atoms are the keys of BASIS_OPS.
"""

import torch
import torch.nn.functional as F
import sympy
from typing import Callable, Dict, List, Tuple, Any

ActivationFunc = Callable[[torch.Tensor], torch.Tensor]
OpTree = Tuple[Any, ...]  # ('atom', str) | ('add'|'mul'|'of', OpTree, OpTree)

# Soft-clip bound. Composites of `square`/`gaussian` can overflow float32;
# 1e6 * tanh(x / 1e6) is ~identity well inside the bound but saturates
# smoothly outside it, so gradients keep flowing instead of becoming NaN.
CLIP = 1e6


def soft_clip_torch(t: torch.Tensor) -> torch.Tensor:
    return CLIP * torch.tanh(t / CLIP)


def soft_clip_sympy(e):
    return CLIP * sympy.tanh(e / CLIP)


# --------------------------------------------------------------------------
# Basis operators. Each entry pairs a torch implementation with a SymPy one.
# The two MUST be numerically identical -- tests/test_operators.py checks it.
# --------------------------------------------------------------------------
BASIS_OPS: Dict[str, Dict[str, Callable]] = {
    'identity':   {'torch': lambda x: x,
                   'sympy': lambda e: e},
    'tanh':       {'torch': torch.tanh,
                   'sympy': sympy.tanh},
    'relu':       {'torch': torch.relu,
                   'sympy': lambda e: sympy.Max(0, e)},
    'sigmoid':    {'torch': torch.sigmoid,
                   'sympy': lambda e: 1 / (1 + sympy.exp(-e))},
    'sin':        {'torch': lambda x: torch.sin(x * torch.pi),
                   'sympy': lambda e: sympy.sin(e * sympy.pi)},
    'cos':        {'torch': torch.cos,
                   'sympy': sympy.cos},
    'gaussian':   {'torch': lambda x: torch.exp(-x ** 2),
                   'sympy': lambda e: sympy.exp(-e ** 2)},
    'square':     {'torch': lambda x: x ** 2,
                   'sympy': lambda e: e ** 2},
    'leaky_relu': {'torch': lambda x: F.leaky_relu(x, 0.01),
                   'sympy': lambda e: sympy.Piecewise((e, e >= 0), (0.01 * e, True))},
}

# --------------------------------------------------------------------------
# Equivalence classes.
#
# Every operator is preceded by a trainable affine map (Wx + b) and followed
# by another linear layer, so some operators are EXACTLY interchangeable --
# the surrounding weights can absorb the difference:
#
#   cos(Wx + b)  ==  sin(pi * (W'x + b'))   with W' = W/pi, b' = b/pi + 1/2
#   sigmoid(u)   ==  1/2 + 1/2 * tanh(u/2)  -- scale and offset are absorbed
#                                              by the following linear layer
#
# relu/leaky_relu are close but not exact. Grouping them matters when
# measuring structural agreement across seeds: two runs that pick `sin` vs
# `cos` differ in name only, and a naive string comparison would report
# instability that carries no functional content.
# --------------------------------------------------------------------------
EQUIVALENCE_CLASSES: Dict[str, str] = {
    'identity': 'identity',
    'tanh': 'saturating', 'sigmoid': 'saturating',
    'relu': 'rectifier', 'leaky_relu': 'rectifier',
    'sin': 'periodic', 'cos': 'periodic',
    'gaussian': 'bump',
    'square': 'quadratic',
}

# One representative per class -- the default search space. Dropping the
# redundant aliases shrinks the space without losing any expressivity.
SEARCH_BASIS: List[str] = ['identity', 'tanh', 'relu', 'sin', 'gaussian', 'square']

_SEPARATORS: Tuple[Tuple[str, str], ...] = (('_plus_', 'add'), ('_x_', 'mul'), ('_of_', 'of'))


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_op(name: str) -> OpTree:
    """Parse an operator name into an expression tree.

    Raises ValueError on an unknown atom. This is deliberate: the previous
    implementation silently fell back to the identity, so a typo or an
    unrecognised composite produced a formula that looked plausible and was
    wrong. Failing loudly is the whole point for an interpretability claim.
    """
    for sep, tag in _SEPARATORS:
        idx = name.find(sep)
        if idx > 0:
            left, right = name[:idx], name[idx + len(sep):]
            if left and right:
                return (tag, parse_op(left), parse_op(right))
    if name not in BASIS_OPS:
        raise ValueError(
            f"Unknown operator atom {name!r}. Known atoms: {sorted(BASIS_OPS)}"
        )
    return ('atom', name)


def composite_name(a: str, b: str, kind: str) -> str:
    """Canonical name for a composite operator.

    `mul` and `add` are commutative, so their operands are sorted -- this
    stops `sin_x_cos` and `cos_x_sin` from both occupying the search space
    as if they were different operators.
    """
    if kind == 'of':
        return f"{a}_of_{b}"
    if kind == 'mul':
        lo, hi = sorted((a, b))
        return f"{lo}_x_{hi}"
    if kind == 'add':
        lo, hi = sorted((a, b))
        return f"{lo}_plus_{hi}"
    raise ValueError(f"Unknown composite kind {kind!r}")


def op_depth(name: str) -> int:
    """Nesting depth of an operator: 1 for an atom, 2 for a composite of atoms."""
    def _d(tree: OpTree) -> int:
        if tree[0] == 'atom':
            return 1
        return 1 + max(_d(tree[1]), _d(tree[2]))
    return _d(parse_op(name))


def canonical_class(name: str) -> str:
    """Map an operator to its equivalence class, recursively for composites.

    Used to compare architectures across seeds without counting
    functionally-meaningless differences (sin vs cos, tanh vs sigmoid).
    """
    def _c(tree: OpTree) -> str:
        if tree[0] == 'atom':
            return EQUIVALENCE_CLASSES[tree[1]]
        left, right = _c(tree[1]), _c(tree[2])
        if tree[0] in ('mul', 'add'):
            left, right = sorted((left, right))
        return f"{left}_{tree[0]}_{right}"
    return _c(parse_op(name))


# --------------------------------------------------------------------------
# Interpreters -- both walk the tree produced by parse_op
# --------------------------------------------------------------------------
def to_torch(name: str) -> ActivationFunc:
    """Compile an operator name to a torch callable."""
    tree = parse_op(name)

    def _build(t: OpTree) -> ActivationFunc:
        if t[0] == 'atom':
            return BASIS_OPS[t[1]]['torch']
        f, g = _build(t[1]), _build(t[2])
        if t[0] == 'of':
            return lambda x: soft_clip_torch(f(g(x)))
        if t[0] == 'mul':
            return lambda x: soft_clip_torch(f(x) * g(x))
        return lambda x: soft_clip_torch(f(x) + g(x))

    return _build(tree)


def to_sympy(name: str, expr, include_clip: bool = True):
    """Compile an operator name to a SymPy expression applied to `expr`.

    include_clip mirrors the soft clipping that `to_torch` applies at every
    composite node. Keep it True when you need the exported formula to match
    the network numerically; set it False only for display, where it is a
    good approximation for |x| << 1e6.
    """
    tree = parse_op(name)
    clip = soft_clip_sympy if include_clip else (lambda e: e)

    def _build(t: OpTree, e):
        if t[0] == 'atom':
            return BASIS_OPS[t[1]]['sympy'](e)
        if t[0] == 'of':
            return clip(_build(t[1], _build(t[2], e)))
        if t[0] == 'mul':
            return clip(_build(t[1], e) * _build(t[2], e))
        return clip(_build(t[1], e) + _build(t[2], e))

    return _build(tree, expr)


# --------------------------------------------------------------------------
# Candidate generation for the search
# --------------------------------------------------------------------------
def basis_candidates(basis: List[str] = None) -> List[str]:
    return list(basis if basis is not None else SEARCH_BASIS)


def composite_candidates(anchor: str, basis: List[str] = None) -> List[str]:
    """Composites built from `anchor` and each basis operator.

    The search uses this as a second tier: first find the best atom for a
    node, then try composing it. Enumerating every pair of every basis
    operator would be ~78 candidates per node; anchoring on the tier-1
    winner keeps it to ~20 while still exploring composition, and -- unlike
    the previous 5%-probability random discovery -- the set is deterministic,
    so the same seed produces the same search.
    """
    basis = basis_candidates(basis)
    out: List[str] = []
    for other in basis:
        for kind in ('of', 'mul', 'add'):
            for a, b in ((anchor, other), (other, anchor)):
                # identity composites collapse to an existing candidate
                if kind == 'of' and 'identity' in (a, b):
                    continue
                if kind in ('mul', 'add') and a == b:
                    continue
                name = composite_name(a, b, kind)
                if name not in out:
                    out.append(name)
    return out


def build_registry(basis: List[str] = None) -> Dict[str, ActivationFunc]:
    """A fresh operator -> callable dict.

    Returns a NEW dict every call. The previous implementation handed every
    node a reference to one module-level dict, so a composite discovered by
    one node appeared in every other node of every model in the process --
    including baselines -- and results depended on execution order.
    """
    return {name: to_torch(name) for name in basis_candidates(basis)}


# Back-compat alias. Read-only: callers must not mutate this.
ACTIVATIONS: Dict[str, ActivationFunc] = {name: to_torch(name) for name in BASIS_OPS}


def create_composite_op(op1_func, op2_func, operation: str = 'compose') -> ActivationFunc:
    """Deprecated: kept so older scripts import cleanly.

    Note the argument order matches the name it is given by the caller --
    for 'compose', op1 is the OUTER function, i.e. op1(op2(x)). The old
    implementation applied them in the opposite order from its own naming.
    """
    if operation == 'compose':
        return lambda x: soft_clip_torch(op1_func(op2_func(x)))
    if operation == 'multiply':
        return lambda x: soft_clip_torch(op1_func(x) * op2_func(x))
    if operation == 'add':
        return lambda x: soft_clip_torch(op1_func(x) + op2_func(x))
    return op1_func
