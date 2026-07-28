"""
The one structural advantage: input dimension.

SINDy beat operator search on every low-dimensional target tested, often at
machine precision, and the earlier "compositional niche" argument mostly
collapsed -- polynomials plus multi-frequency trig plus Gaussians span far more
than the things that merely LOOK compositional, and trig identities dissolve
most of the rest.

But SINDy has a weakness that no library choice repairs: **the library grows
combinatorially with input dimension.** A degree-3 polynomial library over d
inputs has O(d^3) terms:

      d =  3  ->        20 terms
      d = 10  ->       286
      d = 50  ->    23,426
      d = 200 -> 1,373,701

Past a few dozen inputs the design matrix stops fitting in memory, the sparse
regression becomes badly conditioned, and the "interpretable handful of terms"
promise dies with it. Operator search costs O(d) in the first layer and O(1)
after, so it is indifferent to d.

This measures where the crossover is. It is the honest place to look for a niche,
with one caveat stated up front: at high d the relevant competitor stops being
SINDy and becomes gradient boosting and plain MLPs -- and on this repo's own
E. coli data XGBoost already reached ~95% against this model's ~87%. So a real
niche here, not an easy win.

Run:  python experiments/dimension_scaling.py --dims 3 10 30
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.mlp_comparison import (Gaussian, Sin, build_mlp, count,
                                        train_plain)
from experiments.sota_baselines import poly_library, run_sindy, split
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything)

ACTS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sin': Sin, 'gaussian': Gaussian}


def sparse_composite_target(n, d, seed=0):
    """A target that depends on only a few of d inputs, compositionally.

        y = sin(pi * x0^2) + exp(-x1^2) + 0.5 * x2 * x3

    The first term is genuinely out-of-library; the rest are in it. Only 4 of d
    inputs matter, so the difficulty scales purely through library size -- the
    function itself is unchanged as d grows, which isolates the dimension effect
    from task difficulty.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.2, 1.2, (n, d)).astype(np.float32)
    y = (np.sin(np.pi * X[:, 0] ** 2)
         + np.exp(-X[:, 1] ** 2)
         + 0.5 * X[:, 2] * X[:, 3])
    return X, y.reshape(-1, 1).astype(np.float32)


def library_size(d, degree=3, include_trig=True):
    """Closed form, because CONSTRUCTING the library is itself infeasible at
    scale -- an attempt to build it at d=784 did not finish in five minutes,
    which is the argument of this experiment in miniature.

    Terms: 1 constant, d linear, C(d+1,2) quadratic, C(d+2,3) cubic (both with
    repetition), plus 2d trigonometric.
    """
    n = 1 + d
    if degree >= 2:
        n += d * (d + 1) // 2
    if degree >= 3:
        n += d * (d + 1) * (d + 2) // 6
    if include_trig:
        n += 2 * d
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dims', nargs='+', type=int, default=[4, 12, 40])
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--width', type=int, default=8)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--restarts', type=int, default=3)
    args = ap.parse_args()

    print("=" * 96)
    print(" DIMENSION SCALING -- y = sin(pi*x0^2) + exp(-x1^2) + 0.5*x2*x3")
    print(" The target never changes; only the number of irrelevant inputs does.")
    print("=" * 96)
    print(f"{'d':>4} {'lib terms':>10} {'SINDy':>12} {'SINDy s':>8} "
          f"{'MLP-best':>12} {'search':>12} {'search s':>9}   winner")
    print("-" * 96)

    for d in args.dims:
        X, Y = sparse_composite_target(args.n, d)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
        data = (Xtr, ytr, Xva, yva, Xte, yte)
        T = torch.from_numpy
        tt = (T(Xtr), T(ytr), T(Xva), T(yva), T(Xte), T(yte))

        # -- SINDy, richest library it can afford
        t0 = time.time()
        try:
            s_mse, s_terms, _ = run_sindy(
                Xtr.astype(np.float64), ytr.astype(np.float64),
                Xte.astype(np.float64), yte.astype(np.float64),
                Xva=Xva.astype(np.float64), yva=yva.astype(np.float64),
                rich=(d <= 12))
        except (MemoryError, np.linalg.LinAlgError) as exc:
            s_mse, s_terms = float('nan'), -1
            print(f"    SINDy failed at d={d}: {type(exc).__name__}")
        s_time = time.time() - t0

        # -- MLP with the best of four global activations, restart-matched
        cands = []
        for name, act in ACTS.items():
            for r in range(args.restarts):
                seed_everything(r * 101)
                m = build_mlp(d, 1, args.width, args.depth, act)
                v, t = train_plain(m, *tt, epochs=args.epochs)
                cands.append((v, t))
        mlp_mse = min(cands)[1]

        # -- operator search, best known configuration
        t0 = time.time()
        losses = []
        for seed in range(args.seeds):
            seed_everything(seed)
            cfg = SearchConfig(
                seed=seed, search_mode='exhaustive', use_composites=True,
                exhaustive_refine_composites=True,
                topology_rounds=2, allow_growth=True, allow_pruning=True,
                topology_requires_op_gain=False, max_chains=3, max_depth=3,
                exhaustive_max_configs=250, exhaustive_screen_epochs=100,
                exhaustive_verify_top=3, warmup_epochs=300,
                consolidate_epochs=250, probe_epochs=200,
                max_op_sweeps=2, final_restarts=args.restarts,
                compress=False, verbose=False)
            m = MatrixGGLEN(input_dim=d, output_dim=1, hidden_dim=args.width,
                            num_chains=1, chain_depth=args.depth,
                            rng=random.Random(seed))
            losses.append(StructureSearch(m, *data, config=cfg).run().test_loss)
        o_mse = float(np.median(losses))
        o_time = time.time() - t0

        cands_all = {'SINDy': s_mse, 'MLP': mlp_mse, 'search': o_mse}
        winner = min((k for k, v in cands_all.items() if not np.isnan(v)),
                     key=lambda k: cands_all[k])
        print(f"{d:>4} {library_size(d):>10} {s_mse:>12.3e} {s_time:>7.1f}s "
              f"{mlp_mse:>12.3e} {o_mse:>12.3e} {o_time:>8.0f}s   {winner}",
              flush=True)

    print("\n" + "=" * 96)
    print(" WHAT WOULD MAKE THIS A RESULT")
    print("=" * 96)
    print("  SINDy winning at small d and losing at large d, with the crossover")
    print("  identified. If SINDy wins at every d that fits in memory, the")
    print("  dimension argument fails too and the method has no measured niche.")
    print("  If the plain MLP wins everywhere, the operator search is the part")
    print("  that is not carrying weight -- which is the more likely outcome")
    print("  given it lost to a tuned MLP on the teacher task by 2.7x.")


if __name__ == '__main__':
    main()
