"""
Does the search earn its keep as the operator space grows?

Every recovery measurement in this repo lives at ONE scale: 2 nodes over a
6-operator basis, i.e. 21 unordered assignments. At that size the search is
solving a problem that does not need solving -- you can enumerate the whole
space, and the measured result says exactly that:

    correct operators + restarts   28/28   (100%)
    the search                      3/28   ( 11%)
    random operators + restarts      0/28   (  0%)

Search vs random is p = 0.24: not established. But a null result at 21
configurations is weak evidence about a method whose entire premise is spaces
too large to enumerate. THIS is the regime that matters, and nothing has tested
it.

The design. `random + restarts` degrades in a known way as the space grows --
its hit rate is the chance of drawing the true assignment, which falls off a
cliff (1/21 at 2 nodes, ~1/126 at 3, ~1/1300 at 4). A search with real selection
power should degrade far more slowly. So the diagnostic is not the raw recovery
rate at any one scale, it is **how the search-to-chance ratio moves with scale**:

    ratio grows      ->  the search genuinely selects operators; its value is
                         concentrated exactly where enumeration is infeasible,
                         which is the method's claim.
    ratio flat/falls ->  the search has no selection power that survives scale,
                         and at small scale you should just enumerate.

Targets are generated FROM the architecture (a teacher with a known operator
assignment and fixed weights), so exact recovery is guaranteed representable at
every depth -- no hand-derived closed form needed, and the ground truth is known
by construction.

Arms, all given an identical epoch budget:

    oracle      operators forced to the teacher's, restarts. Feasibility gate:
                if this is not near 100%, the scale is unrecoverable for
                reasons unrelated to search and its cell is uninformative.
    search      GAIOptimizer, one long trajectory (the method as shipped)
    random      random operators, restarts (the chance baseline)

Run:  python experiments/scaling_regime.py --nodes 2 4 --seeds 12
      python experiments/scaling_regime.py --nodes 2 --seeds 8 --arms oracle
"""

import argparse
import itertools
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.gai_recovery import EXACT, arm_params
from models.activations import SEARCH_BASIS
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import seed_everything
from training.trainer import GAIOptimizer

BASIS = list(SEARCH_BASIS)

# Ground-truth operator assignment per total node count, as [chain][layer].
#
# The deeper teachers pad with `identity` rather than nesting more nonlinearities,
# and that choice is forced by measurement, not taste. A NESTED teacher at 4
# nodes -- [['sin','gaussian'], ['square','identity']], i.e. gaussian(sin(.)) --
# is not recoverable even with the operators handed over:
#
#     2 nodes,  sin + square                    oracle 4/4   1.3e-15
#     4 nodes,  identity-padded (same function) oracle 4/4   3.1e-15
#     4 nodes,  gaussian-of-sin (nested)        oracle 0/4   4.9e-03   <- and
#                                                  0/4 at 4x the budget too
#
# So depth per se costs nothing; NESTED NONLINEARITY costs everything. That is
# worth stating plainly because composition is the method's premise -- the
# `a_of_b` composite operators exist precisely for nested targets, and those are
# the targets whose weights cannot be optimised to machine precision even when
# the operator assignment is correct.
#
# Padding keeps the FUNCTION easy while making the SEARCH hard (1,296
# assignments at 4 nodes against 36 at 2), which is the only configuration in
# which the search question can be asked at scale at all.
TEACHERS = {
    2: [['sin'], ['square']],
    4: [['sin', 'identity'], ['square', 'identity']],
    6: [['sin', 'identity', 'identity'], ['square', 'identity', 'identity']],
}

# Kept for the record; see the note above.
NESTED_TEACHER_4 = [['sin', 'gaussian'], ['square', 'identity']]


def build(structure, hidden_dim=1, seed=0):
    return MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=hidden_dim,
                       fixed_structure=structure, rng=random.Random(seed))


def make_teacher_data(n_nodes, n=900, seed=0, hidden_dim=1):
    """Target = output of a teacher with the ground-truth operators.

    Guarantees the student CAN represent the target exactly at this depth,
    which a hand-derived closed form does not, and gives a known answer to
    recover at any scale.
    """
    structure = TEACHERS[n_nodes]
    seed_everything(seed)
    teacher = build(structure, hidden_dim=hidden_dim, seed=seed)
    with torch.no_grad():
        # Simple, well-conditioned weights: near-identity plus a small offset,
        # so the teacher computes something non-degenerate that a student
        # starting near zero can still climb toward.
        for ci, li, node in teacher.iter_nodes():
            nn.init.zeros_(node.linear.bias)
            w = node.linear.weight
            nn.init.zeros_(w)
            for i in range(w.shape[0]):
                w[i, (ci + i) % w.shape[1]] = 1.0
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    with torch.no_grad():
        Y = teacher(torch.as_tensor(X)).numpy().astype(np.float32)
    a, b = int(0.6 * n), int(0.8 * n)
    return (X[:a], Y[:a]), (X[a:b], Y[a:b]), (X[b:], Y[b:]), structure


def _test_loss(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        return float(nn.MSELoss()(model(torch.as_tensor(Xte)),
                                  torch.as_tensor(yte)))


def run_search(seed, epochs, data, structure, hidden_dim=1):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    seed_everything(seed)
    shape = [[None] * len(c) for c in structure]
    model = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=hidden_dim,
                        num_chains=len(shape), chain_depth=len(shape[0]),
                        rng=random.Random(seed))
    p = arm_params('GAI-A')
    p['legacy_sa'] = True
    p['mutation_mode'] = 'reset'
    opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **p)
    opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name=f'search/n{seed}')
    ops = [n.op_name for _, _, n in opt.model.iter_nodes()]
    return _test_loss(opt.model, Xte, yte), ops


def run_fixed(seed, epochs, data, structure, k, forced, hidden_dim=1, lr=0.01):
    """K restarts with operators either forced to truth or drawn at random."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    flat_truth = [op for chain in structure for op in chain]
    per = max(1, epochs // k)
    Xtr_t, ytr_t = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    Xva_t, yva_t = torch.as_tensor(Xva), torch.as_tensor(yva)
    crit = nn.MSELoss()
    best_val, best_test, best_ops = float('inf'), float('nan'), None
    for r in range(k):
        s = seed * 1000 + r
        seed_everything(s)
        model = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=hidden_dim,
                            num_chains=len(structure),
                            chain_depth=len(structure[0]),
                            rng=random.Random(s))
        ops = (list(flat_truth) if forced
               else [random.choice(BASIS) for _ in flat_truth])
        for i, (_, _, node) in enumerate(model.iter_nodes()):
            node.set_op(ops[i])
            node.reset_weights_near_identity()
        o = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(per):
            model.train(); o.zero_grad()
            crit(model(Xtr_t), ytr_t).backward(); o.step()
        model.eval()
        with torch.no_grad():
            v = float(crit(model(Xva_t), yva_t))
        if v < best_val:
            best_val, best_test, best_ops = v, _test_loss(model, Xte, yte), ops
    return best_test, best_ops


def chance_rate(n_nodes):
    """P(drawing the true assignment) for one random draw, order-sensitive."""
    return 1.0 / (len(BASIS) ** n_nodes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nodes', nargs='+', type=int, default=[2, 4],
                    choices=sorted(TEACHERS))
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=6000)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--n', type=int, default=900)
    ap.add_argument('--arms', nargs='+', default=['oracle', 'search', 'random'],
                    choices=['oracle', 'search', 'random'])
    args = ap.parse_args()

    print("=" * 100)
    print(" DOES THE SEARCH EARN ITS KEEP AS THE OPERATOR SPACE GROWS?")
    print(f" teacher-generated targets; {args.seeds} seeds;"
          f" {args.epochs} epochs per arm; restarts k={args.k}")
    print("=" * 100)
    print(f"{'nodes':>5} {'space':>9} {'arm':<8} {'recovery':>9} "
          f"{'median test':>13} {'op-set exact':>13}")
    print("-" * 100)

    grid = {}
    for n_nodes in args.nodes:
        tr, va, te, structure = make_teacher_data(n_nodes, n=args.n)
        data = (tr, va, te)
        flat_truth = [op for c in structure for op in c]
        space = len(BASIS) ** n_nodes

        for arm in args.arms:
            losses, exact_ops = [], 0
            for s in range(args.seeds):
                if arm == 'search':
                    t, ops = run_search(s, args.epochs, data, structure)
                else:
                    t, ops = run_fixed(s, args.epochs, data, structure, args.k,
                                       forced=(arm == 'oracle'))
                losses.append(t)
                # Order-INSENSITIVE. The chains are summed, so [sin, square]
                # and [square, sin] are the same model; an order-sensitive
                # comparison undercounts correct selections (it reported 1/16
                # on a cell whose recovery column was already 2/16, which is
                # impossible if the two agree).
                exact_ops += (sorted(ops) == sorted(flat_truth))
            hits = sum(1 for t in losses if t < EXACT)
            grid[(n_nodes, arm)] = (hits, float(np.median(losses)), exact_ops)
            print(f"{n_nodes:>5} {space:>9,} {arm:<8} {hits:>4}/{args.seeds}   "
                  f"{np.median(losses):>13.3e} {exact_ops:>8}/{args.seeds}",
                  flush=True)
        print("-" * 100)

    print("\n" + "=" * 100)
    print(" READING IT")
    print("=" * 100)
    print("  'oracle' is a FEASIBILITY GATE. If it is not near 100% at a given")
    print("  size, nothing at that size is recoverable for reasons unrelated to")
    print("  search, and the search/random cells there say nothing.\n")
    for n_nodes in args.nodes:
        if (n_nodes, 'search') in grid and (n_nodes, 'random') in grid:
            o = grid.get((n_nodes, 'oracle'), (None,))[0]
            se = grid[(n_nodes, 'search')][0]
            ra = grid[(n_nodes, 'random')][0]
            gate = ('' if o is None else
                    f"  [oracle {o}/{args.seeds}"
                    f"{' - GATE FAILED, cell uninformative' if o < args.seeds * 0.5 else ''}]")
            print(f"  {n_nodes} nodes ({len(BASIS)**n_nodes:,} assignments): "
                  f"search {se}/{args.seeds}  random {ra}/{args.seeds}"
                  f"   chance/draw {chance_rate(n_nodes):.2%}{gate}")
    print("\n  Compare the search-to-random gap ACROSS sizes, not within one.")
    print("  Growing gap  -> real selection power, concentrated where")
    print("                  enumeration is infeasible (the method's claim).")
    print("  Flat/shrinking -> no selection power that survives scale; at small")
    print("                  sizes enumerate instead.")


if __name__ == '__main__':
    main()
