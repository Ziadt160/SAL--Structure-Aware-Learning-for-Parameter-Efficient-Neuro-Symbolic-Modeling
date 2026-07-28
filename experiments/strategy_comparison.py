"""
Which node-selection strategy is best -- and does gradient norm help at all?

This is the question the original design answered with "mutate the node with the
lowest gradient norm", and which had never actually been tested here. I removed
importance ranking from the search on the strength of an argument about depth
bias plus the repo's own pendulum sweep. An argument is not a measurement.

WHEN RANKING CAN MATTER

With exhaustive enumeration, or with greedy visiting every node, node ranking is
irrelevant -- everything gets searched. Ranking matters exactly when the budget
does not stretch to all nodes, which is the regime that matters for scaling,
because enumeration is len(basis) ** n_nodes and dies past MLP depth.

So this runs a model deep enough that searching every node is expensive, gives
every strategy the SAME budget (`node_budget` nodes per sweep, same number of
sweeps), and varies only which nodes that budget is spent on:

  taylor_low      lowest  mean |dL/dz * z|   -- "repair the weakest node"
  taylor_high     highest first              -- "spend it where there is leverage"
  gradnorm_low    lowest  mean ||dL/dz||     -- the ORIGINAL metric
  gradnorm_high   highest first
  random          shuffle                    -- the control that has to be beaten
  all             no budget limit            -- reference upper bound, more compute

Every budgeted arm costs the same, so any difference is attributable to the
ranking rule. `random` is the bar: a ranking rule that cannot beat shuffling is
not carrying information.

Run:  python experiments/strategy_comparison.py --seeds 3
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.mlp_comparison import task_analytic, task_lorenz
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything)

TASKS = {'lorenz': task_lorenz, 'analytic': task_analytic}

# (label, node_order, uses the budget?)
STRATEGIES = [
    ('gradnorm_low',  'gradnorm_low',  True),   # the original design
    ('gradnorm_high', 'gradnorm_high', True),
    ('taylor_low',    'taylor_low',    True),
    ('taylor_high',   'taylor_high',   True),
    ('random',        'random',        True),    # the control
    ('all (ref)',     'all',           False),  # upper bound, more compute
]


def split(X, y):
    n = len(X)
    a, b = int(0.6 * n), int(0.8 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='lorenz', choices=list(TASKS))
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--n', type=int, default=4500)
    ap.add_argument('--width', type=int, default=8)
    ap.add_argument('--depth', type=int, default=6,
                    help='deep enough that searching every node is costly '
                         'and 6**depth enumeration is infeasible')
    ap.add_argument('--budget', type=int, default=2,
                    help='nodes searched per sweep by the budgeted arms')
    ap.add_argument('--sweeps', type=int, default=3)
    ap.add_argument('--probe-epochs', type=int, default=150)
    args = ap.parse_args()

    X, Y = TASKS[args.task](n=args.n)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
    data = (Xtr, ytr, Xva, yva, Xte, yte)

    print("=" * 88)
    print(f" NODE-SELECTION STRATEGY -- {args.task} | MLP depth={args.depth} "
          f"width={args.width}")
    print(f" budget={args.budget}/{args.depth} nodes per sweep x {args.sweeps} "
          f"sweeps | {args.seeds} seeds")
    print(f" enumeration would need 6**{args.depth} = {6 ** args.depth:,} "
          f"configurations -- infeasible, which is why ranking matters here")
    print("=" * 88)

    rows = {}
    for seed in range(args.seeds):
        print(f"\n[seed {seed}]", flush=True)
        for label, order, budgeted in STRATEGIES:
            seed_everything(seed)
            cfg = SearchConfig(
                seed=seed, search_mode='greedy',
                node_order=order,
                node_budget=args.budget if budgeted else 0,
                warmup_epochs=300, probe_epochs=args.probe_epochs,
                probe_restarts=1, consolidate_epochs=250,
                max_op_sweeps=args.sweeps, use_composites=False,
                topology_rounds=0, allow_growth=False, allow_pruning=False,
                final_restarts=3, compress=False, verbose=False)
            m = MatrixGGLEN(input_dim=X.shape[1], output_dim=Y.shape[1],
                            hidden_dim=args.width, num_chains=1,
                            chain_depth=args.depth, rng=random.Random(seed))
            tr = StructureSearch(m, *data, config=cfg).run()
            probed = sorted(l for _, l in tr.nodes_probed)
            rows.setdefault(label, []).append(
                (tr.test_loss, tr.train_epochs, len(set(tr.nodes_changed)),
                 tr.final_structure[0], probed))
            print(f"  {label:<14} test={tr.test_loss:.4e} "
                  f"epochs={tr.train_epochs:>6} "
                  f"probed_layers={probed} "
                  f"changed={len(set(tr.nodes_changed))}", flush=True)

    print("\n" + "=" * 88)
    print(f"{'strategy':<15} {'median test':>13} {'best':>12} {'worst':>12} "
          f"{'epochs':>8} {'vs random':>10}")
    print("-" * 88)
    rand_med = float(np.median([r[0] for r in rows['random']]))
    for label, _, _ in STRATEGIES:
        ts = [r[0] for r in rows[label]]
        med = float(np.median(ts))
        print(f"{label:<15} {med:>13.4e} {min(ts):>12.4e} {max(ts):>12.4e} "
              f"{int(np.mean([r[1] for r in rows[label]])):>8} "
              f"{rand_med / med:>9.2f}x")

    print("\n" + "=" * 88)
    print(" VERDICT")
    print("=" * 88)
    best = min((l for l, _, b in STRATEGIES if b),
               key=lambda l: np.median([r[0] for r in rows[l]]))
    print(f"  best budgeted strategy: {best}")
    for label, _, budgeted in STRATEGIES:
        if not budgeted:
            continue
        med = float(np.median([r[0] for r in rows[label]]))
        ratio = rand_med / med
        if label == 'random':
            print(f"  {label:<14} {med:.4e}   (the control)")
        else:
            tag = ('carries information' if ratio > 1.15 else
                   'WORSE than shuffling' if ratio < 0.87 else
                   'indistinguishable from shuffling')
            print(f"  {label:<14} {med:.4e}   {ratio:.2f}x vs random -- {tag}")
    ref = float(np.median([r[0] for r in rows['all (ref)']]))
    print(f"\n  searching every node: {ref:.4e} at "
          f"{int(np.mean([r[1] for r in rows['all (ref)']]))} epochs "
          f"({args.depth / args.budget:.1f}x the probe cost)")
    print(f"  best budgeted arm reaches "
          f"{ref / np.median([r[0] for r in rows[best]]):.2f}x of that.")

    # Which layers does each rule actually spend the budget on? This is where a
    # depth bias would show up: a rule that always picks the same end of the
    # network is not adapting, it is following the gradient-magnitude gradient.
    print("\n" + "=" * 88)
    print(" WHICH LAYERS DID EACH RULE PICK?  (layer index -> times chosen)")
    print("=" * 88)
    for label, _, budgeted in STRATEGIES:
        if not budgeted:
            continue
        counts = {}
        for r in rows[label]:
            for l in r[4]:
                counts[l] = counts.get(l, 0) + 1
        bars = ' '.join(f"L{l}:{counts.get(l, 0)}" for l in range(args.depth))
        distinct = len(counts)
        print(f"  {label:<14} {bars}   ({distinct}/{args.depth} distinct layers)")


if __name__ == '__main__':
    main()
