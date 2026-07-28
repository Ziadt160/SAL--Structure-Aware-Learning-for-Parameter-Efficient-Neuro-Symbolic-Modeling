"""
Does GAIOptimizer return the best model it found, or does it lose it?

The concern: the search mutates, reaches a good structure, then mutates again --
and if the good one was never captured, it is gone. Reasoning about the code
only goes so far, so this measures it.

Two quantities per run:

  best seen    min over every epoch of the validation loss -- the best the
               model was, at any point during the search
  final        the validation loss of the model fit() actually returns

If `final > best seen`, the run threw away a better model than it returned.

There are two distinct mechanisms, and they need separating:

  1. MUTATION CHURN. `_update_best` snapshots state_dict AND chains whenever the
     validation score exceeds the all-time best, and fit() restores that
     snapshot at the end. So a good mutation should be recoverable. This checks
     whether it actually is.

  2. THE EFFICIENCY SWEEP. This one is real and by design: after restoring the
     best-ever state, Phase 2 accepts a smaller model when
     `curr_loss <= base_loss * 1.15` -- up to 15% WORSE -- and assigns it with
     no further restoration. `optimize_efficiency=True` therefore returns a
     model that is knowingly worse than the best found, and reports the
     pre-sweep score. GAI-A, the grid-search winner, has this switched on.

Run:  python experiments/best_model_audit.py
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


class MutationLog:
    """Observer that records every structural event with its epoch."""

    def __init__(self):
        self.epoch = 0
        self.events = []

    def __call__(self, kind, details):
        if kind == 'epoch_end':
            self.epoch = details['epoch']
        elif kind == 'mutation':
            self.events.append({
                'epoch': self.epoch,
                'accepted': bool(details.get('accepted')),
                'from': details.get('old_op'),
                'to': details.get('new_op'),
                'delta': float(details.get('score_delta', 0.0)),
            })


def run(cfg_name, seed, epochs, data, force_eff=None):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    seed_everything(seed)
    cfg = dict(GAI_CONFIGS[cfg_name])
    eff = cfg.pop('optimize_efficiency', False)
    if force_eff is not None:
        eff = force_eff
    model = MatrixGGLEN(input_dim=Xtr.shape[1], output_dim=ytr.shape[1],
                        hidden_dim=cfg.pop('hidden_dim'),
                        num_chains=cfg.pop('num_chains'),
                        chain_depth=cfg.pop('chain_depth'),
                        rng=random.Random(seed))
    log = MutationLog()
    opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), observer=log, **cfg)
    final_preds, history = opt.fit(Xtr, ytr, Xva, yva, epochs=epochs,
                                   name=cfg_name, optimize_efficiency=eff)

    val_loss = -np.asarray(history, dtype=float)          # history = -loss
    best_seen = float(val_loss.min())
    best_epoch = int(val_loss.argmin())
    final = float(np.mean((final_preds - yva) ** 2))
    opt.model.eval()
    with torch.no_grad():
        test = float(nn.MSELoss()(opt.model(torch.as_tensor(Xte)),
                                  torch.as_tensor(yte)))
    return dict(curve=val_loss, best_seen=best_seen, best_epoch=best_epoch,
                final=final, test=test, events=log.events,
                params=sum(p.numel() for p in opt.model.parameters()),
                reported=-opt.best_score, eff=eff)


def plot(runs, path):
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.4 * n), squeeze=False)
    for ax, (label, r) in zip(axes[:, 0], runs):
        c = r['curve']
        ax.semilogy(c, lw=1.0, color='#3b6ea5', label='validation loss')
        ax.axhline(r['best_seen'], color='#2e8b57', ls='--', lw=1.4,
                   label=f"best seen {r['best_seen']:.3e} (ep {r['best_epoch']})")
        ax.axhline(r['final'], color='#c1440e', ls='-', lw=1.4,
                   label=f"final returned {r['final']:.3e}")
        if r['final'] > r['best_seen'] * 1.001:
            ax.axhspan(r['best_seen'], r['final'], color='#c1440e', alpha=0.12)
            ax.set_title(f"{label}   —   LOST {r['final']/r['best_seen']:.2f}x "
                         f"(returned worse than best found)", fontsize=11,
                         color='#c1440e')
        else:
            ax.set_title(f"{label}   —   returned the best it found",
                         fontsize=11, color='#2e8b57')
        acc = [e['epoch'] for e in r['events'] if e['accepted']]
        rej = [e['epoch'] for e in r['events'] if not e['accepted']]
        for x in acc:
            ax.axvline(x, color='#2e8b57', alpha=0.30, lw=0.7)
        for x in rej:
            ax.axvline(x, color='#999999', alpha=0.22, lw=0.6)
        ax.plot([], [], color='#2e8b57', alpha=0.6, lw=1,
                label=f'mutation accepted ({len(acc)})')
        ax.plot([], [], color='#999999', alpha=0.6, lw=1,
                label=f'mutation rejected ({len(rej)})')
        ax.axvline(r['best_epoch'], color='#2e8b57', lw=1.8, alpha=0.9)
        ax.set_xlabel('epoch'); ax.set_ylabel('val MSE')
        ax.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.9)
        ax.grid(alpha=0.25, which='both')
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    print(f"\nplot -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='gauss_of_sin', choices=list(TASKS))
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=4500)
    ap.add_argument('--out', default='results/best_model_audit.png')
    args = ap.parse_args()

    fn, _ = TASKS[args.task]
    X, Y = fn(n=args.n)
    data = split(X, Y)

    arms = [
        ('GAI-A  (efficiency sweep ON — as tuned)', 'GAI-A', None),
        ('GAI-A  (efficiency sweep OFF)', 'GAI-A', False),
        ('GAI-B  (patience=150, never mutates)', 'GAI-B', None),
        ('GAI-C  (patience=60, mutates often)', 'GAI-C', None),
    ]

    print("=" * 100)
    print(f" BEST-MODEL AUDIT -- {args.task}, seed {args.seed}, "
          f"{args.epochs} epochs")
    print(" 'lost' = the run returned a model worse than the best it saw")
    print("=" * 100)
    print(f"{'arm':<42} {'best seen':>11} {'final':>11} {'lost':>8} "
          f"{'muts':>6} {'params':>7}")
    print("-" * 100)

    runs = []
    for label, cfg, feff in arms:
        r = run(cfg, args.seed, args.epochs, data, feff)
        lost = r['final'] / r['best_seen']
        print(f"{label:<42} {r['best_seen']:>11.3e} {r['final']:>11.3e} "
              f"{lost:>7.2f}x {len(r['events']):>6} {r['params']:>7}",
              flush=True)
        runs.append((label, r))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plot(runs, args.out)

    print("\n" + "=" * 100)
    print(" READING IT")
    print("=" * 100)
    print("  lost = 1.00x  -> fit() returned the best model it found")
    print("  lost > 1.00x  -> a better model existed during the run and was")
    print("                   discarded. Compare the two GAI-A rows to see how")
    print("                   much of that is the efficiency sweep's 15%")
    print("                   accuracy-for-parameters trade.")


if __name__ == '__main__':
    main()
