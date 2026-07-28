"""
Does operator COMPOSITION earn its cost on a target containing products?

Lorenz derivatives are the strongest case for the idea in this repo:

    dx/dt = 10(y - x)              linear
    dy/dt = x(28 - z) - y          contains x*z
    dz/dt = x*y - (8/3)z           contains x*y

No single global activation can express a product of two inputs. The `_x_`
multiply composites can. If per-layer operator search has a real niche, this is
where it should show -- and the earlier comparison ran with composites DISABLED,
so it never tested the relevant tool.

All arms use MLP topology (num_chains=1), because the chain ensemble measured as
a liability: at matched parameters `chain-fixed` lost to a plain MLP on every
task. num_chains=1 makes MatrixGGLEN literally an MLP whose per-layer
activations are searchable.

  mlp-best-act    ordinary MLP, one global activation, best of 4 on validation
  mlp+search      per-layer operator search, basis only
  mlp+search+comp per-layer operator search, basis + composites

Run:  python experiments/lorenz_composites.py --seeds 3
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.mlp_comparison import (Gaussian, Sin, build_mlp, count,
                                        task_lorenz, train_plain)
from models.activations import canonical_class
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything)

ACTS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sin': Sin, 'gaussian': Gaussian}


def split(X, y, val_frac=0.2):
    """Contiguous: a random split leaks, since neighbouring trajectory points
    are nearly identical.

    val_frac is exposed because structure selection is far more sensitive to
    validation size than weight training is. Choosing the best of ~25 candidates
    at each of several nodes is itself a fit to the validation split, and a small
    contiguous slice of one trajectory is not a representative sample of the
    attractor.
    """
    n = len(X)
    test_frac = 0.2
    a = int((1.0 - val_frac - test_frac) * n)
    b = int((1.0 - test_frac) * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--widths', nargs='+', type=int, default=[8, 16])
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--n', type=int, default=1500, help='trajectory length')
    ap.add_argument('--val-frac', type=float, default=0.2,
                    help='validation fraction used for structure selection')
    args = ap.parse_args()

    X, Y = task_lorenz(n=args.n)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y, args.val_frac)
    T = torch.from_numpy
    tt = (T(Xtr), T(ytr), T(Xva), T(yva), T(Xte), T(yte))
    data = (Xtr, ytr, Xva, yva, Xte, yte)

    print("=" * 86)
    print(" LORENZ DERIVATIVES -- does composition help where the target has products?")
    print(f" {args.seeds} seeds | MLP topology (num_chains=1), depth={args.depth}")
    print(f" n={args.n} | train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
    print("=" * 86)

    rows = {}
    for seed in range(args.seeds):
        print(f"[seed {seed}]", flush=True)

        # arm 1: ordinary MLP, one global activation, selected on val
        for w in args.widths:
            cands = []
            for act_name, act in ACTS.items():
                seed_everything(seed)
                m = build_mlp(3, 3, w, args.depth, act)
                v, t = train_plain(m, *tt, epochs=args.epochs)
                cands.append((v, t, count(m), act_name))
            v, t, p, a = min(cands)
            rows.setdefault(('mlp-best-act', w), []).append((t, p, a))
            print(f"    mlp-best-act  w{w}: test={t:.4e} params={p} picked={a}",
                  flush=True)

        # arms 2 and 3: per-layer operator search, without/with composition
        for use_comp in (False, True):
            label = 'mlp+search+comp' if use_comp else 'mlp+search'
            for w in args.widths:
                seed_everything(seed)
                cfg = SearchConfig(
                    seed=seed, warmup_epochs=300, probe_epochs=40,
                    probe_restarts=args.restarts, consolidate_epochs=250,
                    max_op_sweeps=2, use_composites=use_comp,
                    topology_rounds=0, allow_growth=False,
                    allow_pruning=False, compress=False, verbose=False)
                m = MatrixGGLEN(input_dim=3, output_dim=3, hidden_dim=w,
                                num_chains=1, chain_depth=args.depth,
                                rng=random.Random(seed))
                tr = StructureSearch(m, *data, config=cfg).run()
                rows.setdefault((label, w), []).append(
                    (tr.test_loss, tr.params, tr.final_structure[0]))
                print(f"    {label:<16} w{w}: test={tr.test_loss:.4e} "
                      f"params={tr.params} epochs={tr.train_epochs} "
                      f"{tr.final_structure[0]}", flush=True)

    print("\n" + "=" * 86)
    print(f"{'arm':<18} {'w':>4} {'params':>7} {'median test':>13} "
          f"{'best test':>12}   detail")
    print("-" * 86)
    for key in sorted(rows, key=lambda k: (k[1], k[0])):
        rs = rows[key]
        ts = [r[0] for r in rs]
        detail = rs[0][2]
        if isinstance(detail, list):
            detail = ','.join(detail)
        print(f"{key[0]:<18} {key[1]:>4} {rs[0][1]:>7} "
              f"{np.median(ts):>13.4e} {min(ts):>12.4e}   {detail}")

    print("\n" + "=" * 86)
    print(" DOES COMPOSITION PAY?")
    print("=" * 86)
    for w in args.widths:
        base = float(np.median([r[0] for r in rows[('mlp-best-act', w)]]))
        b = float(np.median([r[0] for r in rows[('mlp+search', w)]]))
        c = float(np.median([r[0] for r in rows[('mlp+search+comp', w)]]))
        print(f"  w{w}: mlp-best-act {base:.4e} | search {b:.4e} "
              f"({base / b:.2f}x) | search+comp {c:.4e} ({base / c:.2f}x)")
        print(f"        composition vs basis-only: {b / c:.2f}x "
              f"({'composition helps' if c < b else 'composition does NOT help'})")

    # Did the search actually reach for multiplicative operators?
    print("\n  multiplicative operators selected (the ones that can express x*y):")
    for w in args.widths:
        picked = [op for r in rows[('mlp+search+comp', w)] for op in r[2]]
        mults = [op for op in picked if '_x_' in op]
        print(f"    w{w}: {len(mults)}/{len(picked)} nodes -- "
              f"{sorted(set(mults)) if mults else 'none'}")


if __name__ == '__main__':
    main()
