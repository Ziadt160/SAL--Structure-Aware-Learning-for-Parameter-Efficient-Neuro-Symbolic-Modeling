"""
When a mutation happens, do we actually reach a BETTER model?

The best-model audit showed GAIOptimizer never loses the best it finds, and that
GAI-A's best arrives at epoch 1353 -- after mutations begin around epoch 850. So
a mutation plausibly caused it. But "the best arrived after a mutation" is not
"the mutation produced it": ordinary training might have reached the same place.

This separates the two with a PAIRED control. For each seed, the identical model
and budget are run twice:

    mutate   GAIOptimizer as configured
    frozen   same everything, patience set beyond the budget so no structural
             move can ever fire -- pure weight training

Same seed means the same initial architecture and the same initial weights, so
the only difference is whether mutation is allowed. That makes the comparison
paired rather than across-population, which matters at these sample sizes.

Reported per config:

  gain            frozen_best / mutate_best, per seed. >1 means mutation helped.
  win rate        fraction of seeds where mutation helped at all
  attributable    how many mutations were followed by a NEW GLOBAL BEST within
                  the grace window -- i.e. mutations that demonstrably paid off
  best-before     the best loss reached BEFORE the first mutation, so we can see
                  whether everything good happened pre-search

Run:  python experiments/mutation_gain.py --seeds 6
"""

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.gai_final import GAI_CONFIGS
from experiments.sota_baselines import TASKS, split
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import seed_everything
from training.trainer import GAIOptimizer


def one_run(cfg_name, seed, epochs, data, mutate=True, sa='fixed'):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    seed_everything(seed)
    cfg = dict(GAI_CONFIGS[cfg_name])
    eff = cfg.pop('optimize_efficiency', False)
    # 'legacy' = original acceptance/stagnation behaviour with a weight-resetting
    # mutation; 'fixed' = rolling stagnation reference, revert cooldown,
    # preserved Adam moments, scale-invariant acceptance, finite tabu, and a
    # homotopy operator swap (which never resets, so the two arms differ in
    # whether reset_weights_near_identity is on the path at all).
    cfg['legacy_sa'] = (sa == 'legacy')
    cfg['mutation_mode'] = 'reset' if sa == 'legacy' else 'homotopy'
    if not mutate:
        cfg['patience'] = epochs * 10        # structural search can never fire
        eff = False
    model = MatrixGGLEN(input_dim=Xtr.shape[1], output_dim=ytr.shape[1],
                        hidden_dim=cfg.pop('hidden_dim'),
                        num_chains=cfg.pop('num_chains'),
                        chain_depth=cfg.pop('chain_depth'),
                        rng=random.Random(seed))
    opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **cfg)
    _, history = opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name=cfg_name,
                         optimize_efficiency=eff)
    curve = -np.asarray(history, dtype=float)
    return curve, opt.model.get_structure(), list(opt.mutation_log)


def mutation_epochs(curve, thresh=3.0):
    """DEPRECATED fallback: infer structural events from a >thresh loss jump.

    Do not trust this. Measured against GAIOptimizer.mutation_log on GAI-A /
    gauss_of_sin, 1500 epochs:

        seed 0:  1 real mutation  vs 16 "detected"
        seed 1:  0 real mutations vs 16 "detected"

    The detected epochs arrive in tight clusters (864, 868, 880, 883, 886, 891,
    894) that are ordinary Adam loss spikes, not structural events -- the
    threshold is measuring TRAINING INSTABILITY and labelling it mutation. Every
    mutation count and payoff rate derived this way is dominated by false
    positives.

    It fails in the other direction too: a homotopy swap starts at t=0 where the
    network is bit-identical to before the swap, so it leaves no discontinuity
    to find at all.

    Kept only so curves recorded before mutation_log existed still parse.
    """
    if len(curve) < 3:
        return []
    ratio = curve[1:] / np.maximum(curve[:-1], 1e-30)
    return (np.nonzero(ratio > thresh)[0] + 1).tolist()


def analyse(curve, grace=60, log=None):
    """best before the first mutation, and how many mutations were followed by
    a new global best within `grace` epochs."""
    # +1 on logged epochs: evolution fires at the END of epoch e, after that
    # epoch's score is appended, so curve[e] is still pre-mutation and
    # curve[e+1] is the first reading that reflects the change. The curve
    # fallback already returns the post-jump index, so it takes no shift.
    muts = ([e['epoch'] + 1 for e in log if e['epoch'] + 1 < len(curve)]
            if log is not None else mutation_epochs(curve))
    running = np.minimum.accumulate(curve)
    best_epoch = int(curve.argmin())
    if not muts:
        return dict(first_mut=None, best_before=float(curve.min()),
                    attributable=0, n_mut=0, best_epoch=best_epoch)
    first = muts[0]
    best_before = float(curve[:first].min()) if first > 0 else float(curve[0])
    attributable = 0
    for m in muts:
        prior = running[m - 1] if m > 0 else np.inf
        window = curve[m:m + grace]
        if len(window) and window.min() < prior * 0.999:
            attributable += 1
    # Accepted vs reverted is only knowable from the log; the curve fallback
    # cannot distinguish them.
    acc = (sum(1 for e in log if e.get('accepted')) if log is not None else None)
    return dict(first_mut=first, best_before=best_before,
                attributable=attributable, n_mut=len(muts),
                accepted=acc, best_epoch=best_epoch)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='gauss_of_sin', choices=list(TASKS))
    ap.add_argument('--configs', nargs='+', default=['GAI-A', 'GAI-C'])
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--n', type=int, default=4500)
    ap.add_argument('--sa', nargs='+', default=['fixed'],
                    choices=['legacy', 'fixed'])
    ap.add_argument('--out', default='results/mutation_gain.png')
    args = ap.parse_args()

    fn, _ = TASKS[args.task]
    X, Y = fn(n=args.n)
    data = split(X, Y)

    print("=" * 100)
    print(f" DOES MUTATION PRODUCE A BETTER MODEL?  {args.task}, "
          f"{args.seeds} paired seeds, {args.epochs} epochs")
    print(" paired: same seed => same initial architecture AND weights;"
          " only mutation differs")
    print("=" * 100)

    store = {}
    for cfg in args.configs:
        for sa in args.sa:
            key = f"{cfg}/{sa}" if len(args.sa) > 1 else cfg
            print(f"\n--- {cfg}  (SA: {sa}) ---")
            print(f"{'seed':>4} {'mutate best':>13} {'frozen best':>13} "
                  f"{'gain':>8} {'best-before-1st-mut':>20} {'muts':>5} "
                  f"{'kept':>5} {'paid off':>9}")
            rows = []
            for s in range(args.seeds):
                cm, _, log = one_run(cfg, s, args.epochs, data, mutate=True,
                                     sa=sa)
                cf, _, _ = one_run(cfg, s, args.epochs, data, mutate=False,
                                   sa=sa)
                a = analyse(cm, log=log)
                mb, fb = float(cm.min()), float(cf.min())
                gain = fb / mb
                rows.append(dict(seed=s, mut=mb, frozen=fb, gain=gain, **a))
                kept = '-' if a['accepted'] is None else a['accepted']
                print(f"{s:>4} {mb:>13.3e} {fb:>13.3e} {gain:>7.2f}x "
                      f"{a['best_before']:>20.3e} {a['n_mut']:>5} "
                      f"{kept:>5} {a['attributable']:>9}", flush=True)
            store[key] = rows
            g = np.array([r['gain'] for r in rows])
            wins = int((g > 1.0).sum())
            tot_m = sum(r['n_mut'] for r in rows)
            tot_a = sum(r['attributable'] for r in rows)
            # How often the mutating run's overall best was already reached
            # BEFORE its first mutation -- i.e. everything after the first
            # structural change was wasted budget.
            pre = sum(1 for r in rows
                      if r['best_before'] <= r['mut'] * (1 + 1e-12))
            print(f"  median gain {np.median(g):.2f}x   "
                  f"mutation helped in {wins}/{args.seeds} seeds   "
                  f"mutations that produced a new best: {tot_a}/{tot_m}"
                  + (f" ({tot_a/tot_m:.0%})" if tot_m else ""))
            print(f"  best was already reached BEFORE the first mutation in "
                  f"{pre}/{args.seeds} seeds")

    # ---- paired plot
    panels = list(store)
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5),
                             squeeze=False)
    for ax, cfg in zip(axes[0], panels):
        rows = store[cfg]
        for r in rows:
            colour = '#2e8b57' if r['gain'] > 1 else '#c1440e'
            ax.plot([0, 1], [r['frozen'], r['mut']], '-o', color=colour,
                    alpha=0.75, ms=5, lw=1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['frozen\n(no mutation)', 'mutate'])
        ax.set_yscale('log'); ax.set_ylabel('best validation MSE')
        g = np.array([r['gain'] for r in rows])
        wins = int((g > 1.0).sum())
        ax.set_title(f"{cfg}\nmutation helped {wins}/{len(rows)} seeds, "
                     f"median gain {np.median(g):.2f}x")
        ax.grid(alpha=0.3, which='both')
        ax.plot([], [], color='#2e8b57', lw=2, label='mutation helped')
        ax.plot([], [], color='#c1440e', lw=2, label='mutation hurt')
        ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"\nplot -> {args.out}")

    print("\n" + "=" * 100)
    print(" READING IT")
    print("=" * 100)
    print("  Each line joins the SAME seed run both ways. Green = mutation")
    print("  reached a better model than pure training; red = it did not.")
    print("  'paid off' counts mutations followed by a new global best -- if")
    print("  that is near zero while gain > 1, the improvement came from the")
    print("  extra training after the mutation, not from the structure change.")


if __name__ == '__main__':
    main()
