"""
Re-run the negative results at settings that are not broken.

Every unfavourable number reported for this method came from greedy search with
probe_epochs=40. Both have since been measured and found wanting:

  probe@40             spearman +0.20 vs full training -- choosing near-blind
  probe@400            spearman +0.81                  -- trustworthy
  greedy   (recovery)  3/8 exact, median test 2.5e-04
  exhaustive           6/8 exact, median test 3.4e-11  -- 7 orders better

So the earlier conclusions describe a configuration, not the idea. This re-runs
the two tasks where the method looked worst -- Lorenz and the MLP-teacher --
with `search_mode='exhaustive'` and the tuned probe, against the same
best-of-4-global-activations MLP baseline.

Arms, all on MLP topology (num_chains=1) since the chain ensemble measured as a
liability at matched parameters:

  mlp-best-act    ordinary MLP, one global activation, best of 4 chosen on val
  search-greedy   per-layer operator search, greedy, tuned probe
  search-exhaust  per-layer operator search, exhaustive over the basis

Run:  python experiments/tuned_rerun.py --seeds 3
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
                                        task_analytic, task_lorenz,
                                        task_teacher, train_plain)
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything, slot_frequencies,
                                       structural_agreement)

ACTS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sin': Sin, 'gaussian': Gaussian}
TASKS = {'lorenz': task_lorenz, 'teacher': task_teacher,
         'analytic': task_analytic}


def split(X, y):
    n = len(X)
    a, b = int(0.6 * n), int(0.8 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--tasks', nargs='+', default=['lorenz', 'teacher'])
    ap.add_argument('--width', type=int, default=8)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--n', type=int, default=4500)
    ap.add_argument('--composites', action='store_true')
    ap.add_argument('--train-score', default='best', choices=['best', 'final'],
                    help="How a candidate is scored inside the search. 'best' "
                         "(original) takes the MINIMUM val loss over the "
                         "training budget, which is a min-over-N statistic that "
                         "flatters high-variance operators -- and on Lorenz the "
                         "correct answer (square) is one of the high-variance "
                         "ones. 'final' scores the end of the budget instead. "
                         "If the win survives 'final', it is not the estimator.")
    ap.add_argument('--restarts', type=int, default=5,
                    help='final-training restarts, applied to EVERY arm')
    args = ap.parse_args()

    for task in args.tasks:
        X, Y = TASKS[task](n=args.n)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
        T = torch.from_numpy
        tt = (T(Xtr), T(ytr), T(Xva), T(yva), T(Xte), T(yte))
        data = (Xtr, ytr, Xva, yva, Xte, yte)

        print("=" * 84)
        print(f" TUNED RE-RUN -- {task} | MLP depth={args.depth} "
              f"width={args.width} | {args.seeds} seeds")
        print(f" train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
        print("=" * 84)

        rows = {}
        structs = {'search-greedy': [], 'search-exhaust': []}

        for seed in range(args.seeds):
            # Baseline: best single global activation, selected on val, with the
            # SAME number of independent retrains the search arms get. Without
            # this the search would be given N draws at the weight lottery while
            # the baseline gets one -- exactly the asymmetry that made the
            # original repo's comparisons unusable.
            cands = []
            for name, act in ACTS.items():
                for r in range(args.restarts):
                    seed_everything(seed * 10007 + r * 101)
                    m = build_mlp(X.shape[1], Y.shape[1], args.width,
                                  args.depth, act)
                    v, t = train_plain(m, *tt, epochs=args.epochs)
                    cands.append((v, t, count(m), name))
            v, t, p, a = min(cands)
            base_epochs = len(ACTS) * args.restarts * args.epochs
            rows.setdefault('mlp-best-act', []).append((t, p, base_epochs))
            print(f"[seed {seed}] mlp-best-act    test={t:.4e} params={p} "
                  f"picked={a} (best of {len(ACTS)}x{args.restarts})",
                  flush=True)

            for label, mode in (('search-greedy', 'greedy'),
                                ('search-exhaust', 'exhaustive')):
                seed_everything(seed)
                cfg = SearchConfig(
                    seed=seed, search_mode=mode,
                    warmup_epochs=300, consolidate_epochs=250,
                    max_op_sweeps=2, use_composites=args.composites,
                    exhaustive_refine_composites=args.composites,
                    exhaustive_screen_epochs=150, exhaustive_verify_top=5,
                    train_score=args.train_score,
                    final_restarts=args.restarts,
                    topology_rounds=0, allow_growth=False, allow_pruning=False,
                    compress=False, verbose=False)
                m = MatrixGGLEN(input_dim=X.shape[1], output_dim=Y.shape[1],
                                hidden_dim=args.width, num_chains=1,
                                chain_depth=args.depth, rng=random.Random(seed))
                tr = StructureSearch(m, *data, config=cfg).run()
                rows.setdefault(label, []).append(
                    (tr.test_loss, tr.params, tr.train_epochs))
                structs[label].append(tr.final_structure)
                print(f"[seed {seed}] {label:<15} test={tr.test_loss:.4e} "
                      f"params={tr.params} epochs={tr.train_epochs} "
                      f"{tr.final_structure[0]}", flush=True)

        print(f"\n{'arm':<16} {'params':>7} {'median test':>13} {'best':>12} "
              f"{'worst':>12} {'epochs':>9}")
        print("-" * 76)
        base = None
        for arm in ('mlp-best-act', 'search-greedy', 'search-exhaust'):
            ts = [r[0] for r in rows[arm]]
            med = float(np.median(ts))
            if arm == 'mlp-best-act':
                base = med
            print(f"{arm:<16} {rows[arm][0][1]:>7} {med:>13.4e} "
                  f"{min(ts):>12.4e} {max(ts):>12.4e} "
                  f"{int(np.mean([r[2] for r in rows[arm]])):>9}")

        print(f"\n  vs the plain-MLP baseline ({base:.4e}):")
        for arm in ('search-greedy', 'search-exhaust'):
            med = float(np.median([r[0] for r in rows[arm]]))
            print(f"    {arm:<16} {base / med:5.2f}x  "
                  f"({'BETTER' if med < base else 'worse'})")

        print("\n  structural agreement across seeds (modulo equivalence):")
        for arm in ('search-greedy', 'search-exhaust'):
            r = structural_agreement(structs[arm], modulo_equivalence=True)
            tag = ('above chance' if r['mean_agreement'] > r['null_agreement'] * 1.5
                   else 'at or near chance')
            print(f"    {arm:<16} agreement={r['mean_agreement']:.2f} "
                  f"chance={r['null_agreement']:.2f}  -- {tag}")
        for arm in ('search-greedy', 'search-exhaust'):
            freq = slot_frequencies(structs[arm], modulo_equivalence=True)
            picks = {k: max(v.items(), key=lambda kv: -kv[1] and kv[1])
                     for k, v in freq.items()}
            summary = ', '.join(f"{k}={v[0]}({v[1]}/{args.seeds})"
                                for k, v in sorted(picks.items()))
            print(f"    {arm:<16} per-slot: {summary}")
        print()


if __name__ == '__main__':
    main()
