"""
Can a better OPTIMISER beat SINDy, where more epochs alone could not?

The premise for this experiment is a correction. I claimed the 9-order gap on
Lorenz was structural -- a direct linear solve against non-convex descent. That
is wrong in an important way: with `identity -> square -> identity`, the model
computes `sum_k w_k (a_k . x + c_k)^2`, and signed combinations of squared linear
forms span ALL quadratic polynomials -- including the xy and xz products Lorenz
needs. An exact solution exists in the hypothesis space at hidden_dim=8.

Landing 9 orders above a reachable optimum is an OPTIMISATION failure. And the
optimiser has been Adam throughout, which is the wrong tool for the last few
orders of a smooth least-squares objective: a first-order method with an
adaptive step plateaus around 1e-5..1e-7. Quasi-Newton with a strong-Wolfe line
search reaches 1e-12 and beyond. Adam-then-LBFGS is standard in scientific ML.

Axes swept, all with the architecture search already fixed at its best known
configuration:

  optimiser   adam            |  adam + L-BFGS refinement
  precision   float32         |  float64
  budget      1x              |  5x

CONFIG IS SELECTED ON VALIDATION and only then scored on test, so this is a
hyperparameter search rather than test-set fitting. SINDy gets its own tuning in
sota_baselines.py (two libraries x four thresholds, also chosen on validation).

Reference bars:  lorenz 4.12e-15   gauss_of_sin 6.15e-06   polynomial 1.10e-14

Run:  python experiments/optimizer_search.py --task lorenz
"""

import argparse
import itertools
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


def run_one(data, seed, width, depth, lbfgs, dtype, mult):
    torch.set_default_dtype(dtype)
    npdt = np.float64 if dtype == torch.float64 else np.float32
    d = [a.astype(npdt) for a in data]
    seed_everything(seed)
    cfg = SearchConfig(
        seed=seed, search_mode='exhaustive', use_composites=True,
        exhaustive_refine_composites=True,
        topology_rounds=2, allow_growth=True, allow_pruning=True,
        topology_requires_op_gain=False, max_chains=3, max_depth=3,
        exhaustive_max_configs=300,
        warmup_epochs=int(300 * mult),
        exhaustive_screen_epochs=int(100 * mult),
        consolidate_epochs=int(250 * mult),
        probe_epochs=int(200 * mult),
        exhaustive_verify_top=3, max_op_sweeps=2,
        final_restarts=4, lbfgs_iters=lbfgs,
        compress=False, verbose=False)
    m = MatrixGGLEN(input_dim=d[0].shape[1], output_dim=d[1].shape[1],
                    hidden_dim=width, num_chains=1, chain_depth=depth,
                    rng=random.Random(seed))
    tr = StructureSearch(m, *d, config=cfg).run()
    torch.set_default_dtype(torch.float32)
    return tr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='lorenz', choices=list(TASKS))
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--width', type=int, default=8)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--n', type=int, default=4500)
    ap.add_argument('--lbfgs', nargs='+', type=int, default=[0, 500])
    ap.add_argument('--mults', nargs='+', type=float, default=[1, 5])
    ap.add_argument('--fp64', action='store_true', default=True)
    args = ap.parse_args()

    fn, kind = TASKS[args.task]
    X, Y = fn(n=args.n)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
    data = (Xtr, ytr, Xva, yva, Xte, yte)
    bar = SINDY_BAR[args.task]

    dtypes = [torch.float32] + ([torch.float64] if args.fp64 else [])
    grid = list(itertools.product(args.lbfgs, dtypes, args.mults))

    print("=" * 94)
    print(f" OPTIMISER SEARCH -- {args.task} [{kind}]")
    print(f" SINDy bar: {bar:.4e}   |   {len(grid)} configs x {args.seeds} seeds")
    print("=" * 94)
    print(f"{'lbfgs':>6} {'dtype':>8} {'budget':>7} {'median val':>13} "
          f"{'median test':>13} {'best test':>13} {'vs SINDy':>11}")
    print("-" * 94)

    results = []
    for lb, dt, mult in grid:
        vals, tests = [], []
        for seed in range(args.seeds):
            tr = run_one(data, seed, args.width, args.depth, lb, dt, mult)
            vals.append(tr.val_loss)
            tests.append(tr.test_loss)
        mv, mt = float(np.median(vals)), float(np.median(tests))
        name = 'fp64' if dt == torch.float64 else 'fp32'
        print(f"{lb:>6} {name:>8} {mult:>6.0f}x {mv:>13.4e} {mt:>13.4e} "
              f"{min(tests):>13.4e} {mt / bar:>10.2e}x", flush=True)
        results.append((mv, mt, min(tests), lb, name, mult))

    # Selection on VALIDATION -- the whole point of doing it this way.
    best = min(results, key=lambda r: r[0])
    print("\n" + "=" * 94)
    print(" VERDICT")
    print("=" * 94)
    print(f"  config selected on validation: lbfgs={best[3]} {best[4]} "
          f"budget={best[5]:.0f}x")
    print(f"  its test loss:      {best[1]:.4e}")
    print(f"  SINDy:              {bar:.4e}")
    if best[1] < bar:
        print(f"  -> OPERATOR SEARCH WINS by {bar / best[1]:.3g}x")
    else:
        print(f"  -> SINDy still wins by {best[1] / bar:.3g}x")
    fp32_1x = [r for r in results if r[3] == 0 and r[4] == 'fp32'
               and r[5] == min(args.mults)]
    if fp32_1x:
        base = fp32_1x[0][1]
        print(f"\n  improvement over plain Adam / fp32 / 1x ({base:.4e}): "
              f"{base / best[1]:.3g}x")
        print("  If that factor is large, the earlier conclusion that the gap")
        print("  was structural was wrong, and it was an optimiser choice.")


if __name__ == '__main__':
    main()
