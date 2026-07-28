"""
Does resetting the mutated node's weights cost more than it buys?

Two things were established first, because they change the question:

  * mutate() already resets ONLY the node being mutated. Every other weight in
    the network is untouched. The 4-6 order-of-magnitude loss spike seen after
    each mutation comes from re-initialising one node in a 2-chain model:
    that destroys the signal through half the network, and every downstream
    layer is still tuned for the old node's output distribution.

  * mutate(reset_weights=False) -- carry the existing weights straight into the
    new operator -- already existed as a parameter but nothing ever passed it,
    so it was unreachable and had never been measured.

The trade-off has a real argument on each side. Resetting avoids feeding weights
tuned for a bounded operator (tanh) into an unbounded one (square), where they
could explode. Transferring keeps the learned input projection, which is largely
operator-independent, so the model restarts near its current function instead of
near random.

Measured per arm, paired by seed:

  best loss        the best validation loss reached
  shock            median ratio loss_after / loss_before across mutations --
                   how violently a mutation disrupts the model
  recovery         median epochs to get back under the pre-mutation loss
  productive       share of mutations followed by a NEW GLOBAL BEST

TWO THINGS THIS MEASUREMENT CANNOT DO, both of which have already misled once:

  * `productive` is NOT a causal attribution. It uses a `max_recovery`-epoch
    window (default 800). Over that span ordinary training supplies a new best
    on its own, so the figure is an upper bound that mostly reflects window
    length -- the same runs score ~20% here and 0/126 under the 60-epoch window
    in mutation_gain.py. Shrink `max_recovery` before quoting it.

  * The two arms diverge in RNG STREAM, not just in mechanism: `reset` draws
    random numbers from the global generator and `transfer` does not, so from
    the first mutation onward the arms see different noise. Differences in
    `best loss` therefore carry a stream-position confound and should not be
    read as the effect of resetting. `shock` and `recovery` are measured per
    mutation event against that same run's own preceding loss, so they are not
    affected.

Run:  python experiments/mutation_mechanics.py --seeds 5
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

# mutation_mode='reset' is REQUIRED for these arms to differ at all.
#
# Under the default mutation_mode='homotopy', _trigger_evolution restores
# backup_state and backup_chains after evolve_structure runs, which undoes the
# weight reset before it can affect anything -- so mutate_reset_weights becomes
# dead. Two arms set only by that flag then run identical configurations,
# differing solely in how many draws reset_weights_near_identity consumed from
# the global RNG before being discarded. An earlier version of this file did
# exactly that and was measuring stream position, not mechanism.
ARMS = {
    'reset':    dict(mutate_reset_weights=True, mutation_mode='reset'),
    'transfer': dict(mutate_reset_weights=False, mutation_mode='reset'),
}


def run(cfg_name, seed, epochs, data, arm, strategy=None):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    seed_everything(seed)
    cfg = dict(GAI_CONFIGS[cfg_name])
    eff = cfg.pop('optimize_efficiency', False)
    if strategy is not None:
        cfg['mutation_strategy'] = strategy
    model = MatrixGGLEN(input_dim=Xtr.shape[1], output_dim=ytr.shape[1],
                        hidden_dim=cfg.pop('hidden_dim'),
                        num_chains=cfg.pop('num_chains'),
                        chain_depth=cfg.pop('chain_depth'),
                        rng=random.Random(seed))
    opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **ARMS[arm], **cfg)
    _, history = opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name=f"{cfg_name}/{arm}",
                         optimize_efficiency=eff)
    return -np.asarray(history, dtype=float), list(opt.mutation_log)


def mutation_stats(curve, jump=2.0, max_recovery=800, log=None):
    """Characterise each mutation: shock, recovery time, and whether it paid off.

    Mutation epochs come from GAIOptimizer.mutation_log when available. The
    loss-discontinuity fallback below only sees mutations that actually
    perturbed the model, which makes it blind to homotopy swaps (bit-identical
    at t=0) and biased toward the ones that went wrong.
    """
    if log is not None:
        # +1: evolution fires at the END of epoch e, after that epoch's
        # validation score is already appended, so curve[e] is still the
        # PRE-mutation reading and curve[e+1] is the first one that reflects
        # the change. (The curve fallback below returns the post-jump index
        # directly, which is why it needs no shift -- mixing the two
        # conventions silently reports a shock of 1.0x.)
        idx = [e['epoch'] + 1 for e in log if 0 < e['epoch'] + 1 < len(curve)]
    else:
        ratio = curve[1:] / np.maximum(curve[:-1], 1e-30)
        idx = (np.nonzero(ratio > jump)[0] + 1).tolist()
    running = np.minimum.accumulate(curve)
    shocks, recov, productive = [], [], 0
    for m in idx:
        before, after = curve[m - 1], curve[m]
        shocks.append(after / max(before, 1e-30))
        seg = curve[m:m + max_recovery]
        back = np.nonzero(seg <= before)[0]
        recov.append(int(back[0]) if len(back) else max_recovery)
        prior = running[m - 1]
        win = curve[m:m + max_recovery]
        if len(win) and win.min() < prior * 0.999:
            productive += 1
    return dict(n=len(idx),
                shock=float(np.median(shocks)) if shocks else float('nan'),
                recovery=float(np.median(recov)) if recov else float('nan'),
                productive=productive)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='gauss_of_sin', choices=list(TASKS))
    ap.add_argument('--config', default='GAI-A')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--n', type=int, default=4500)
    ap.add_argument('--out', default='results/mutation_mechanics.png')
    args = ap.parse_args()

    fn, _ = TASKS[args.task]
    X, Y = fn(n=args.n)
    data = split(X, Y)

    print("=" * 104)
    print(f" MUTATION MECHANICS -- {args.config} on {args.task}, "
          f"{args.seeds} paired seeds")
    print(" reset   = re-initialise the mutated node (current behaviour)")
    print(" transfer= carry its weights into the new operator (never tested)")
    print("=" * 104)
    print(f"{'arm':<9} {'seed':>4} {'best loss':>12} {'muts':>5} "
          f"{'shock (x worse)':>16} {'recovery (ep)':>14} {'productive':>11}")
    print("-" * 104)

    store = {a: [] for a in ARMS}
    curves = {}
    for arm in ARMS:
        for s in range(args.seeds):
            c, log = run(args.config, s, args.epochs, data, arm)
            st = mutation_stats(c, log=log)
            best = float(c.min())
            store[arm].append(dict(seed=s, best=best, **st))
            curves.setdefault(arm, {})[s] = c
            print(f"{arm:<9} {s:>4} {best:>12.3e} {st['n']:>5} "
                  f"{st['shock']:>15.1f}x {st['recovery']:>14.0f} "
                  f"{st['productive']:>4}/{st['n']:<6}", flush=True)

    print("\n" + "=" * 104)
    print(" SUMMARY")
    print("=" * 104)
    for arm in ARMS:
        rows = store[arm]
        b = np.array([r['best'] for r in rows])
        sh = np.array([r['shock'] for r in rows], dtype=float)
        rc = np.array([r['recovery'] for r in rows], dtype=float)
        tm = sum(r['n'] for r in rows); tp = sum(r['productive'] for r in rows)
        print(f"  {arm:<9} median best {np.median(b):.3e}   "
              f"median shock {np.nanmedian(sh):6.1f}x   "
              f"median recovery {np.nanmedian(rc):5.0f} ep   "
              f"productive {tp}/{tm}"
              + (f" ({tp/tm:.0%})" if tm else ""))

    br = np.array([r['best'] for r in store['reset']])
    bt = np.array([r['best'] for r in store['transfer']])
    wins = int((bt < br).sum())
    print(f"\n  transfer beat reset in {wins}/{args.seeds} paired seeds; "
          f"median ratio reset/transfer = {np.median(br) / np.median(bt):.2f}x")
    print("  >1 means transferring the weights is better.")

    ties = int((br == bt).sum())
    if ties:
        print(f"\n  CAVEAT: {ties}/{args.seeds} seeds TIED exactly on best loss.")
        print("  While no mutation improves on the pre-mutation best, this")
        print("  column is degenerate by construction: both arms share a seed,")
        print("  an initialisation, and therefore an identical trajectory up to")
        print("  the first mutation -- so the best is already fixed before the")
        print("  arms can diverge. An exact tie is the EXPECTED result here, not")
        print("  evidence that the two mechanisms are equivalent. Read 'shock'")
        print("  and 'recovery', which is where they actually separate.")

    # ---- plot: one representative curve per arm + paired bests
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
    for arm, colour in (('reset', '#c1440e'), ('transfer', '#2e8b57')):
        axes[0].semilogy(curves[arm][0], lw=0.9, color=colour, label=arm)
    axes[0].set_title(f'{args.config} seed 0 — validation loss')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('val MSE')
    axes[0].legend(); axes[0].grid(alpha=0.3, which='both')

    for r_r, r_t in zip(store['reset'], store['transfer']):
        colour = '#2e8b57' if r_t['best'] < r_r['best'] else '#c1440e'
        axes[1].plot([0, 1], [r_r['best'], r_t['best']], '-o', color=colour,
                     alpha=0.8, ms=6)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['reset', 'transfer'])
    axes[1].set_yscale('log'); axes[1].set_ylabel('best val MSE')
    axes[1].set_title(f'paired best loss — transfer won {wins}/{args.seeds}')
    axes[1].grid(alpha=0.3, which='both')

    sh_r = [r['shock'] for r in store['reset']]
    sh_t = [r['shock'] for r in store['transfer']]
    axes[2].boxplot([sh_r, sh_t], labels=['reset', 'transfer'])
    axes[2].set_yscale('log')
    axes[2].set_ylabel('loss multiplier at mutation')
    axes[2].set_title('mutation shock — how far the model is thrown')
    axes[2].grid(alpha=0.3, which='both')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"\nplot -> {args.out}")


if __name__ == '__main__':
    main()
