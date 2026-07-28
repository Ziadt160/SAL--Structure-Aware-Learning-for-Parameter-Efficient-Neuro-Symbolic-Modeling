"""
Does the probe rank operators the same way full training does?

This is the crux. Phase 1 chooses operators using a cheap proxy: reset one
node, train that node plus the read-out for a few dozen epochs, score on
validation. The choice is only as good as the proxy's agreement with the thing
we actually care about -- validation loss after training the whole network to
convergence.

If the two rankings disagree, the search will confidently install operators
that are worse, which is exactly the observed behaviour: on Lorenz the search
picks `gaussian, gaussian, tanh` and lands 4-6x behind an ordinary MLP that
just uses `tanh` everywhere. Enlarging the validation split did not fix it
(tested at 6x), so selection noise is not the explanation.

Reports Spearman rank correlation between proxy score and full-training score,
per node. Near 1.0 means the proxy is trustworthy. Near 0 or negative means the
search is choosing at random or worse, and no amount of search budget helps.

Run:  python experiments/probe_fidelity.py
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.mlp_comparison import task_analytic, task_lorenz
from models.activations import SEARCH_BASIS
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything)


def spearman(a, b):
    """Rank correlation without a scipy dependency."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float('nan')


def split(X, y, val_frac=0.2):
    n = len(X)
    a = int((0.8 - val_frac) * n)
    b = int(0.8 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='lorenz', choices=['lorenz', 'analytic'])
    ap.add_argument('--n', type=int, default=4500)
    ap.add_argument('--hidden', type=int, default=8)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--probe-epochs', nargs='+', type=int, default=[40, 150])
    ap.add_argument('--probe-from', nargs='+', default=['reset'],
                    choices=['reset', 'current', 'both'])
    ap.add_argument('--restarts', nargs='+', type=int, default=[2])
    ap.add_argument('--reduce', nargs='+', default=['median'],
                    choices=['min', 'mean', 'median'],
                    help='how to combine a candidate scores across restarts')
    ap.add_argument('--full-epochs', type=int, default=800)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--quiet', action='store_true',
                    help='summary only -- skip the per-operator tables')
    args = ap.parse_args()

    # Every (probe_epochs, probe_from, restarts) combination gets measured.
    settings = [(pe, pf, r, rd) for pe in args.probe_epochs
                for pf in args.probe_from for r in args.restarts
                for rd in args.reduce]

    X, Y = (task_lorenz(n=args.n) if args.task == 'lorenz'
            else task_analytic(n=args.n))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
    data = (Xtr, ytr, Xva, yva, Xte, yte)

    print("=" * 84)
    print(f" PROBE FIDELITY -- {args.task}, MLP depth={args.depth} "
          f"width={args.hidden}")
    print(f" train={len(Xtr)} val={len(Xva)}  full-training budget="
          f"{args.full_epochs} epochs")
    print("=" * 84)

    all_corr = {s: [] for s in settings}
    top1 = {s: [] for s in settings}
    cost = {s: 0 for s in settings}

    for seed in range(args.seeds):
        # Ground truth: set each operator at each node, train the WHOLE network.
        seed_everything(seed)
        base = MatrixGGLEN(input_dim=X.shape[1], output_dim=Y.shape[1],
                           hidden_dim=args.hidden, num_chains=1,
                           chain_depth=args.depth, rng=random.Random(seed))
        cfg0 = SearchConfig(seed=seed, verbose=False)
        s0 = StructureSearch(base, *data, config=cfg0)
        start_struct = base.get_structure()

        for node_idx in range(args.depth):
            full = []
            for op in SEARCH_BASIS:
                seed_everything(seed)
                m = MatrixGGLEN(input_dim=X.shape[1], output_dim=Y.shape[1],
                                hidden_dim=args.hidden, num_chains=1,
                                chain_depth=args.depth, rng=random.Random(seed))
                struct = [list(start_struct[0])]
                struct[0][node_idx] = op
                m.set_structure(struct)
                s = StructureSearch(m, *data, config=cfg0)
                full.append(s._train(m, args.full_epochs, cfg0.lr))

            best_full = SEARCH_BASIS[int(np.argmin(full))]
            print(f"\n[seed {seed}] node {node_idx} "
                  f"(others fixed at {start_struct[0]}) "
                  f"truth={best_full} spread="
                  f"{100 * (max(full) / min(full) - 1):.0f}%", flush=True)

            probes = {}
            for (pe, pf, nr, rd) in settings:
                seed_everything(seed)
                m = MatrixGGLEN(input_dim=X.shape[1], output_dim=Y.shape[1],
                                hidden_dim=args.hidden, num_chains=1,
                                chain_depth=args.depth, rng=random.Random(seed))
                m.set_structure([list(start_struct[0])])
                cfg = SearchConfig(seed=seed, probe_epochs=pe,
                                   probe_restarts=nr, probe_from=pf,
                                   probe_reduce=rd, verbose=False)
                s = StructureSearch(m, *data, config=cfg)
                s._train(m, cfg.warmup_epochs, cfg.lr)     # same warm start
                node = list(m.iter_nodes())[node_idx][2]
                adapter = s._adapter_params()
                ad0 = [p.detach().clone() for p in adapter]
                current = node.snapshot()
                before = s.trace.train_epochs
                scores = []
                for op in SEARCH_BASIS:
                    sc, _, _ = s._probe_best(node, op, 0, node_idx, 0, ad0,
                                             current)
                    scores.append(sc)
                probes[(pe, pf, nr, rd)] = scores
                cost[(pe, pf, nr, rd)] = s.trace.train_epochs - before

            if not args.quiet:
                print(f"  {'operator':<10} {'full-train':>12} " +
                      ' '.join(f"{'p'+str(pe)+'/'+pf[:3]+'/'+str(nr)+'/'+rd[:3]:>16}"
                               for pe, pf, nr, rd in settings))
                for i, op in enumerate(SEARCH_BASIS):
                    cells = ' '.join(f"{probes[s][i]:>16.4e}" for s in settings)
                    print(f"  {op:<10} {full[i]:>12.4e} {cells}")

            for st in settings:
                r = spearman(probes[st], full)
                all_corr[st].append(r)
                picked = SEARCH_BASIS[int(np.argmin(probes[st]))]
                # What the search would actually lose by trusting this probe.
                regret = full[SEARCH_BASIS.index(picked)] / min(full)
                top1[st].append(regret)
                print(f"    probe@{st[0]:<4} from={st[1]:<7} r={st[2]} "
                      f"{st[3]:<6} spearman={r:+.2f}  picks {picked:<9} "
                      f"regret={regret:.2f}x", flush=True)

    print("\n" + "=" * 84)
    print(" VERDICT -- what does each probe setting buy?")
    print("=" * 84)
    print(f"  {'setting':<32} {'spearman':>9} {'mean regret':>12} "
          f"{'epochs/node':>12}   assessment")
    for st in sorted(settings, key=lambda s: -np.mean(all_corr[s])):
        rs = np.array(all_corr[st], dtype=float)
        reg = float(np.mean(top1[st]))
        m = float(np.mean(rs))
        verdict = ('trustworthy' if m > 0.7 else
                   'weak' if m > 0.3 else 'uninformative (choosing blind)')
        label = f"probe@{st[0]} {st[1]} r={st[2]} {st[3]}"
        print(f"  {label:<32} {m:>+9.2f} {reg:>11.2f}x {cost[st]:>12} "
              f"  {verdict}")
    print("\n  spearman: agreement between probe ranking and full-training ranking.")
    print("  regret:   how much worse the probe's PICK is than the true best.")
    print("            1.00x means the probe picks the genuinely best operator.")
    print("            This is what actually matters -- a probe can rank the")
    print("            middle badly and still be useful if it gets the top right.")


if __name__ == '__main__':
    main()
