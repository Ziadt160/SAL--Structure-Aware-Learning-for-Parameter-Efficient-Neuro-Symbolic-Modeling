"""
Is the gap to SINDy a BUDGET gap or a KIND-OF-METHOD gap?

The honest way to answer "would more epochs and a hyperparameter grid get us
there" is to scale the budget and watch the curve, rather than argue about it.

Two things are measured:

1. BUDGET SCALING. Same configuration, training budget multiplied 1x / 3x / 10x.
   If test loss keeps falling roughly in proportion, budget is the lever and a
   grid search is worth running. If it plateaus, the gap is structural and no
   amount of compute closes it.

2. PRECISION FLOOR. The same task in float32 and float64. On the pure
   polynomial target we already reach 5.7e-14 against SINDy's 1.1e-14, which is
   suspiciously close to the float32 floor. If float64 moves it and float32 does
   not, we are precision-limited, not budget-limited -- and epochs are
   irrelevant.

Reference bars, from experiments/sota_baselines.py:
    lorenz       SINDy 4.12e-15   (exact; linear in a degree-2 library)
    polynomial   SINDy 1.10e-14   (exact)

Run:  python experiments/budget_scaling.py
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.sota_baselines import TASKS, split
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything)

SINDY_BAR = {'lorenz': 4.1187e-15, 'polynomial': 1.0989e-14,
             'gauss_of_sin': 6.145e-06, 'sin_of_square': 4.858e-15,
             'square_of_sin': 3.368e-15}


def run(data, seed, mult, width, depth, dtype=torch.float32):
    """One run at `mult` times the reference budget."""
    torch.set_default_dtype(dtype)
    Xtr, ytr, Xva, yva, Xte, yte = [a.astype(
        np.float64 if dtype == torch.float64 else np.float32) for a in data]
    seed_everything(seed)
    cfg = SearchConfig(
        seed=seed, search_mode='exhaustive', use_composites=True,
        exhaustive_refine_composites=True,
        topology_rounds=2, allow_growth=True, allow_pruning=True,
        topology_requires_op_gain=False, max_chains=3, max_depth=4,
        exhaustive_max_configs=1300,
        warmup_epochs=int(300 * mult),
        exhaustive_screen_epochs=int(120 * mult),
        consolidate_epochs=int(250 * mult),
        probe_epochs=int(400 * mult),
        exhaustive_verify_top=4, max_op_sweeps=2,
        final_restarts=6, compress=False, verbose=False)
    m = MatrixGGLEN(input_dim=Xtr.shape[1], output_dim=ytr.shape[1],
                    hidden_dim=width, num_chains=1, chain_depth=depth,
                    rng=random.Random(seed))
    tr = StructureSearch(m, Xtr, ytr, Xva, yva, Xte, yte, config=cfg).run()
    torch.set_default_dtype(torch.float32)
    return tr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='polynomial', choices=list(TASKS))
    ap.add_argument('--mults', nargs='+', type=float, default=[1, 3, 10])
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--width', type=int, default=8)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--n', type=int, default=4500)
    ap.add_argument('--float64', action='store_true',
                    help='also run the 1x budget in double precision')
    args = ap.parse_args()

    fn, kind = TASKS[args.task]
    X, Y = fn(n=args.n)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
    data = (Xtr, ytr, Xva, yva, Xte, yte)
    bar = SINDY_BAR.get(args.task, float('nan'))

    print("=" * 84)
    print(f" BUDGET SCALING -- {args.task} [{kind}]")
    print(f" SINDy reference: {bar:.4e}")
    print("=" * 84)
    print(f"{'budget':>8} {'epochs':>10} {'median test':>13} {'best':>13} "
          f"{'vs SINDy':>12} {'gain vs 1x':>11}")
    print("-" * 84)

    base = None
    for mult in args.mults:
        losses, eps = [], []
        for seed in range(args.seeds):
            tr = run(data, seed, mult, args.width, args.depth)
            losses.append(tr.test_loss)
            eps.append(tr.train_epochs)
        med = float(np.median(losses))
        if base is None:
            base = med
        print(f"{mult:>7.0f}x {int(np.mean(eps)):>10} {med:>13.4e} "
              f"{min(losses):>13.4e} {med / bar:>11.2e}x {base / med:>10.2f}x",
              flush=True)

    if args.float64:
        losses = []
        for seed in range(args.seeds):
            tr = run(data, seed, 1.0, args.width, args.depth, torch.float64)
            losses.append(tr.test_loss)
        med = float(np.median(losses))
        print(f"{'fp64 1x':>8} {'':>10} {med:>13.4e} {min(losses):>13.4e} "
              f"{med / bar:>11.2e}x {base / med:>10.2f}x")

    print("\n" + "=" * 84)
    print(" READING THIS")
    print("=" * 84)
    print("  'gain vs 1x' near 1.0 across the rows means the budget is NOT the")
    print("  binding constraint, and a hyperparameter grid will not close the")
    print("  gap either -- both spend compute on a limit that is not compute.")
    print("  If fp64 moves the number while extra epochs do not, we are")
    print("  precision-limited and the comparison is settled: a direct linear")
    print("  solve in the right basis cannot be beaten by iterative descent.")


if __name__ == '__main__':
    main()
