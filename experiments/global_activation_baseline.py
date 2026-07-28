"""
The control that decides whether PER-NODE operator search earns its cost.

mlp_comparison.py showed a 7x spread between a relu MLP and a tanh MLP on the
analytic task -- from the activation choice alone. So the honest question is not
"chain+search vs relu MLP", it is:

    per-node operator search   vs   an ordinary MLP using the best SINGLE
                                    activation, picked by trying a handful

The second is trivially cheap: train the same MLP once per activation and keep
the best on validation. If that matches the search, then the expensive part of
the idea -- letting every node choose its own operator -- is buying nothing
that "try 5 activations" would not.

Uses the same tasks, splits, and epoch budget as mlp_comparison.py, so the
numbers are directly comparable to that table.

Run:  python experiments/global_activation_baseline.py --seeds 3
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.mlp_comparison import (TASKS, build_mlp, count, split,
                                        train_plain)
from training.structure_search import seed_everything


class Sin(nn.Module):
    """Matches models.activations 'sin', including the pi scaling."""
    def forward(self, x):
        return torch.sin(x * torch.pi)


class Gaussian(nn.Module):
    def forward(self, x):
        return torch.exp(-x ** 2)


class Square(nn.Module):
    def forward(self, x):
        return x ** 2


ACTS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sin': Sin,
        'gaussian': Gaussian, 'square': Square}
WIDTHS = (4, 8, 16, 32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    args = ap.parse_args()

    for name in args.tasks:
        print(f"\n{'=' * 78}\n TASK: {name}  (MLP, one global activation)\n{'=' * 78}")
        rows = {}
        for seed in range(args.seeds):
            X, y = TASKS[name](seed=seed)
            (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
            T = torch.from_numpy
            tt = (T(Xtr), T(ytr), T(Xva), T(yva), T(Xte), T(yte))

            for act_name, act in ACTS.items():
                for w in WIDTHS:
                    seed_everything(seed)
                    m = build_mlp(X.shape[1], y.shape[1], w, 2, act)
                    v, t = train_plain(m, *tt, epochs=args.epochs)
                    rows.setdefault((act_name, w), []).append((v, t, count(m)))

        print(f"{'activation':<10} " + ' '.join(f"{'w' + str(w):>11}" for w in WIDTHS))
        for act_name in ACTS:
            cells = []
            for w in WIDTHS:
                ts = [t for _, t, _ in rows[(act_name, w)]]
                cells.append(f"{np.median(ts):>11.3e}")
            print(f"{act_name:<10} " + ' '.join(cells))
        print(f"{'params':<10} " +
              ' '.join(f"{rows[('tanh', w)][0][2]:>11}" for w in WIDTHS))

        # "Try 5 activations, keep the best on VAL" -- selection must not peek
        # at test, so pick the config by val and then report its test loss.
        print("\n  best-single-activation MLP, selected on validation:")
        for w in WIDTHS:
            per_seed = []
            for s in range(args.seeds):
                cands = [(rows[(a, w)][s][0], rows[(a, w)][s][1], a) for a in ACTS]
                v, t, a = min(cands)
                per_seed.append((t, a))
            med = float(np.median([t for t, _ in per_seed]))
            picks = ','.join(sorted({a for _, a in per_seed}))
            print(f"    w{w:<3} params={rows[('tanh', w)][0][2]:>5} "
                  f"median test={med:.3e}   picked: {picks}")


if __name__ == '__main__':
    main()
