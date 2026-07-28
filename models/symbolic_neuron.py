"""A single searchable node: an affine map followed by a mutable operator."""

import copy
import math
import random
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Set, Tuple

from .activations import build_registry, to_torch, SEARCH_BASIS


class MatrixSymbolicNode(nn.Module):
    """Linear -> dropout -> symbolic operator, where the operator is searchable.

    `op_name` is a property: assigning to it recompiles the cached operator,
    so external code that does `node.op_name = 'relu'` stays correct.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 op_name: str = 'identity',
                 track_gradients: bool = True,
                 dropout: float = 0.0,
                 allowed_tools: Optional[Dict[str, Any]] = None,
                 importance_mode: str = 'taylor'):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.dropout = nn.Dropout(dropout)
        self.track_gradients = track_gradients
        self.importance_mode = importance_mode

        # A private copy. Sharing one dict across nodes (the previous
        # behaviour) let a composite discovered by one node leak into every
        # other node in the process and made results order-dependent.
        if allowed_tools is None:
            self.tools: Dict[str, Any] = build_registry(SEARCH_BASIS)
        else:
            self.tools = dict(allowed_tools)

        self._op_name = op_name
        self._op_fn = to_torch(op_name)
        if op_name not in self.tools:
            self.tools[op_name] = self._op_fn

        self._grad_norm_sum = 0.0
        self._taylor_sum = 0.0
        self._call_count = 0

        # How a mutated non-square node is re-initialised, and whether
        # square/gaussian are nudged off their stationary point at z=0.
        # Both defaults are the ORIGINAL behaviour; see
        # reset_weights_near_identity for the measurements that put them back.
        self.reset_scale = 'small'            # 'small' | 'xavier'
        self.reset_offset_stationary = False

        # Homotopy swap state (see begin_swap)
        self._swap_from = None
        self._swap_to = None
        self._swap_target = None
        self._swap_t = 0.0

    # -- operator ---------------------------------------------------------
    @property
    def op_name(self) -> str:
        return self._op_name

    @op_name.setter
    def op_name(self, name: str):
        self.set_op(name)

    def set_op(self, name: str):
        """Set the operator, compiling it once instead of per forward pass."""
        self._op_fn = to_torch(name)          # raises on an unknown operator
        self._op_name = name
        self.tools.setdefault(name, self._op_fn)

    # -- homotopy operator swap -------------------------------------------
    def begin_swap(self, new_op: str):
        """Start a function-preserving transition from the current operator.

        Implements the P-activation family of Network Morphism (Wei et al.,
        ICML 2016, arXiv:1603.01670), generalised from `phi -> identity` to an
        arbitrary operator swap:

            g_t(z) = (1 - t) * g_old(z) + t * g_new(z)

        At t=0 the node computes EXACTLY what it computed before, so the swap
        costs nothing: no loss shock, nothing to recover from, and the
        accept/reject test measures the operator instead of measuring
        re-initialisation damage. Annealing t: 0 -> 1 hands the node over to the
        new operator continuously, with the weights adapting as it goes.

        This is what makes the reset-vs-transfer question moot -- at t=0 the
        existing weights are trivially correct for the current function.
        """
        self._swap_from = self._op_fn
        self._swap_to = to_torch(new_op)
        self._swap_target = new_op
        self._swap_t = 0.0

    def set_swap_t(self, t: float):
        """Advance the transition. At t >= 1 the new operator is installed."""
        if self._swap_to is None:
            return
        self._swap_t = float(min(max(t, 0.0), 1.0))
        if self._swap_t >= 1.0:
            target = self._swap_target
            self._clear_swap()
            self.set_op(target)

    def cancel_swap(self):
        """Abandon a transition and keep the original operator."""
        self._clear_swap()

    def _clear_swap(self):
        self._swap_from = None
        self._swap_to = None
        self._swap_target = None
        self._swap_t = 0.0

    @property
    def swapping(self) -> bool:
        return self._swap_to is not None

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.linear(x))
        if self._swap_to is not None:
            t = self._swap_t
            out = (1.0 - t) * self._swap_from(z) + t * self._swap_to(z)
        else:
            out = self._op_fn(z)

        if self.training and self.track_gradients and out.requires_grad:
            self._attach_probe(out)
        return out

    def _attach_probe(self, tensor: torch.Tensor):
        """Accumulate importance statistics during backward.

        No retain_grad() here -- the tensor hook receives the gradient
        directly, so retaining a .grad on every intermediate activation was
        pure memory overhead.
        """
        activation = tensor.detach()

        def hook(grad):
            with torch.no_grad():
                self._grad_norm_sum += grad.norm().item()
                self._taylor_sum += (grad * activation).abs().mean().item()
                self._call_count += 1

        tensor.register_hook(hook)

    # -- importance -------------------------------------------------------
    def get_importance(self) -> float:
        """Importance of this node.

        'taylor' (default) accumulates mean |dL/dz * z|, the first-order
        estimate of how much the loss would change if this node's output were
        zeroed. 'gradnorm' is the original mean ||dL/dz||.

        gradnorm is kept only for A/B comparison -- it has a systematic depth
        bias. Backpropagated gradients shrink with distance from the loss, so
        the earliest layer in a chain reliably scores lowest regardless of its
        actual contribution, and an importance-guided search using it spends
        nearly all its proposals on the first layer. Multiplying by the
        activation removes most of that bias, because a node that matters
        has large activations where the gradient is large.
        """
        if self._call_count == 0:
            return 0.0
        if self.importance_mode == 'gradnorm':
            return self._grad_norm_sum / self._call_count
        return self._taylor_sum / self._call_count

    def get_metrics(self) -> Dict[str, float]:
        if self._call_count == 0:
            return {'taylor': 0.0, 'gradnorm': 0.0, 'calls': 0}
        return {
            'taylor': self._taylor_sum / self._call_count,
            'gradnorm': self._grad_norm_sum / self._call_count,
            'calls': self._call_count,
        }

    def reset_metrics(self):
        self._grad_norm_sum = 0.0
        self._taylor_sum = 0.0
        self._call_count = 0

    # -- weights ----------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Capture operator + weights so a candidate probe can be undone."""
        return {
            'op_name': self._op_name,
            'weight': self.linear.weight.detach().clone(),
            'bias': self.linear.bias.detach().clone(),
        }

    def restore(self, snap: Dict[str, Any]):
        with torch.no_grad():
            self.linear.weight.copy_(snap['weight'])
            self.linear.bias.copy_(snap['bias'])
        self.set_op(snap['op_name'])

    def reset_weights_near_identity(self):
        """Small random weights plus an identity component where shapes allow.

        Used when a mutation should start from a clean slate. The search
        path deliberately does NOT call this: probing candidates from the
        SAME weights is what makes the comparison about the operator rather
        than about which reinitialisation got luckier.

        Two properties of the original looked like defects on inspection. Only
        one of them was, and the "fix" to the other had to be reverted:

        1. NOT a defect. The identity component applies only when
           in_features == out_features, so a non-square node gets a bare
           U(-0.05, 0.05) draw -- for hidden_dim=32, input_dim=3 a standard
           deviation ~8x below Xavier. Replacing it with a correctly scaled
           Xavier draw is the textbook choice and it MEASURED WORSE:
           ground-truth recovery on sin(pi*x1)+x2^2 fell from 3/8 seeds to 0/8
           (best test 7.6e-16 -> 2.7e-04). At width 1 a node is Linear(2->1),
           where Xavier gives U(-1.414, 1.414) against U(-0.05, 0.05) -- 28x
           wider. For sin(pi*w*x) a near-zero w starts almost linear and can
           grow smoothly into the correct frequency; a wide random w lands in a
           wrong frequency basin that gradient descent cannot leave. The narrow
           draw is doing real work as a start-simple-and-grow prior, so `small`
           is the default. `reset_scale='xavier'` restores the wide draw for
           architectures where under-scaling actually costs something.

        2. `square` and `gaussian` were placed at a stationary point of their
           OWN activation: d/dz z^2 = 2z and d/dz e^(-z^2) = -2z e^(-z^2), both
           zero at z=0. Offsetting the bias so they wake on a sloped part of the
           curve measured NEUTRAL -- 3/8 -> 2/8 recovery with an identical best
           loss (7.647e-16 vs 7.857e-16), i.e. one seed, inside the noise at
           n=8. Kept as an option, default off: the mechanism is real but the
           benefit is unmeasured, and this routine has now been shown once to be
           more delicate than it looks.
        """
        with torch.no_grad():
            fan_in, fan_out = self.linear.in_features, self.linear.out_features
            if fan_in == fan_out:
                nn.init.uniform_(self.linear.weight, -0.05, 0.05)
                self.linear.weight.add_(torch.eye(fan_in) * 0.9)
            elif self.reset_scale == 'xavier':
                bound = math.sqrt(6.0 / (fan_in + fan_out))
                nn.init.uniform_(self.linear.weight, -bound, bound)
            else:
                nn.init.uniform_(self.linear.weight, -0.05, 0.05)
            nn.init.zeros_(self.linear.bias)
            if (self.reset_offset_stationary
                    and self._op_name in ('square', 'gaussian')):
                self.linear.bias.fill_(0.5)

    def make_identity(self):
        """Make this node an exact identity map (requires in_dim == out_dim).

        Used for function-preserving topology growth: a node appended this
        way leaves the network's output bit-for-bit unchanged.
        """
        if self.linear.in_features != self.linear.out_features:
            raise ValueError(
                f"make_identity needs a square node, got "
                f"{self.linear.in_features}->{self.linear.out_features}"
            )
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(self.linear.in_features))
            nn.init.zeros_(self.linear.bias)
        self.set_op('identity')

    def zero_out(self):
        """Make this node emit exactly zero, whatever its input."""
        with torch.no_grad():
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        self.set_op('identity')

    # -- legacy mutation --------------------------------------------------
    def mutate(self,
               forbidden_ops: Optional[Set[str]] = None,
               logger_prefix: str = "GAI",
               reset_weights: bool = True,
               rng: Optional[random.Random] = None) -> Tuple[bool, Optional[str]]:
        """Random operator mutation (legacy path used by GAIOptimizer).

        `rng` accepts an explicit generator so the caller can make the search
        reproducible; the module-level `random` is used otherwise. The
        stochastic 'discover a new composite with 5% probability' behaviour
        was removed -- composite candidates are now enumerated deterministically
        by training.structure_search. Note that this removes the ALGORITHMIC
        source of irreproducibility only; runs still differ across
        OMP_NUM_THREADS values because BLAS reductions are order-dependent. See
        training.structure_search.seed_everything.
        """
        forbidden_ops = forbidden_ops or set()
        rng = rng or random
        choices = sorted(set(self.tools) - {self._op_name} - forbidden_ops)
        if not choices:
            return False, None

        old_op = self._op_name
        self.set_op(rng.choice(choices))
        print(f"    [{logger_prefix}] -> Mutating: {old_op.upper()} ===> "
              f"{self._op_name.upper()}", flush=True)
        if reset_weights:
            self.reset_weights_near_identity()
        return True, old_op
