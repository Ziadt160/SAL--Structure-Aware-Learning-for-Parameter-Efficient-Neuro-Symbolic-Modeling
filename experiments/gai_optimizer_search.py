"""
Randomised hyperparameter search over GAIOptimizer -- the ORIGINAL algorithm.

This closes a gap I created. I replaced GAIOptimizer with
training/structure_search.py and then benchmarked only my replacement, so the
original algorithm's hyperparameter space was never searched. Everything I said
about "the method" was really about my reimplementation at one setting.

GAIOptimizer has axes that structure_search does not have at all:

  use_annealing      probabilistic acceptance of WORSE structural moves. Neither
                     greedy nor exhaustive search has this, and it is aimed
                     exactly at the failure I diagnosed in greedy -- getting
                     stuck where two nodes must change together. If it escapes
                     those without exhaustive's k**n cost, it also solves the
                     scaling wall.
  initial_temp       how much worse a move may be and still be accepted
  mutation_strategy  'importance' (gradient-norm ranked) or 'random'
  l1_lambda          L1 on first-layer weights -- sparsity pressure, which is a
                     different route to a compact model than operator choice
  patience           epochs of stagnation before a structural move is attempted
  grace_period       epochs after a move before it is judged
  optimize_efficiency  the built-in width-reduction sweep

Search is RANDOMISED over the joint space rather than a full grid (the grid is
648+ points), configs are selected on VALIDATION, and only the winner is scored
on test.

Reference bars:  lorenz SINDy 4.12e-15 | structure_search best 3.72e-05
                 gauss_of_sin SINDy 6.15e-06 | structure_search 2.55e-05

Run:  python experiments/gai_optimizer_search.py --task lorenz --configs 40
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

from experiments.sota_baselines import TASKS, split
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import seed_everything
from training.trainer import GAIOptimizer

BARS = {'lorenz': (4.1187e-15, 3.7179e-05), 'polynomial': (1.0989e-14, 5.7080e-14),
        'gauss_of_sin': (6.145e-06, 2.5504e-05), 'sin_of_square': (4.858e-15, 2.4101e-05),
        'square_of_sin': (3.368e-15, 1.0962e-05)}

SPACE = {
    'lr': [0.001, 0.003, 0.01, 0.03],
    'patience': [10, 30, 60, 150],
    'grace_period': [5, 10, 25, 50],
    'initial_temp': [0.05, 0.2, 0.5, 1.0],
    'use_annealing': [True, False],
    'mutation_strategy': ['importance', 'random'],
    'l1_lambda': [0.0, 1e-5, 1e-4, 1e-3],
    'hidden_dim': [8, 16, 32],
    'num_chains': [1, 2, 3],
    'chain_depth': [2, 3, 4],
    'optimize_efficiency': [False, True],
}


def sample_config(rng):
    return {k: rng.choice(v) for k, v in SPACE.items()}


def evaluate(cfg, data, seed, epochs):
    Xtr, ytr, Xva, yva, Xte, yte = data
    seed_everything(seed)
    model = MatrixGGLEN(
        input_dim=Xtr.shape[1], output_dim=ytr.shape[1],
        hidden_dim=cfg['hidden_dim'], num_chains=cfg['num_chains'],
        chain_depth=cfg['chain_depth'], rng=random.Random(seed))
    opt = GAIOptimizer(
        model, lr=cfg['lr'], patience=cfg['patience'],
        grace_period=cfg['grace_period'], initial_temp=cfg['initial_temp'],
        use_annealing=cfg['use_annealing'],
        mutation_strategy=cfg['mutation_strategy'],
        l1_lambda=cfg['l1_lambda'], loss_fn=nn.MSELoss())
    opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name='cfg',
            optimize_efficiency=cfg['optimize_efficiency'])

    m = opt.model
    m.eval()
    crit = nn.MSELoss()
    with torch.no_grad():
        val = float(crit(m(torch.as_tensor(Xva)), torch.as_tensor(yva)))
        test = float(crit(m(torch.as_tensor(Xte)), torch.as_tensor(yte)))
    return val, test, m.get_structure(), sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='lorenz', choices=list(TASKS))
    ap.add_argument('--configs', type=int, default=40)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--seeds', type=int, default=1,
                    help='seeds per config during the search; the winner is '
                         'then re-run over more seeds')
    ap.add_argument('--final-seeds', type=int, default=3)
    ap.add_argument('--n', type=int, default=4500)
    args = ap.parse_args()

    fn, kind = TASKS[args.task]
    X, Y = fn(n=args.n)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
    data = (Xtr, ytr, Xva, yva, Xte, yte)
    sindy_bar, ss_bar = BARS[args.task]

    print("=" * 96)
    print(f" GAIOptimizer RANDOM SEARCH -- {args.task} [{kind}]")
    print(f" {args.configs} configs x {args.seeds} seed(s) x {args.epochs} epochs")
    print(f" bars:  SINDy {sindy_bar:.3e}   structure_search {ss_bar:.3e}")
    print("=" * 96)

    rng = random.Random(0)
    rows = []
    t_start = time.time()
    for i in range(args.configs):
        cfg = sample_config(rng)
        vals, tests = [], []
        try:
            for s in range(args.seeds):
                v, t, struct, params = evaluate(cfg, data, s, args.epochs)
                vals.append(v); tests.append(t)
        except Exception as exc:
            print(f"[{i:>3}] FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        v, t = float(np.median(vals)), float(np.median(tests))
        rows.append((v, t, cfg, struct, params))
        tag = ('BEATS SINDy' if t < sindy_bar else
               'beats struct_search' if t < ss_bar else '')
        print(f"[{i:>3}] val={v:.3e} test={t:.3e} p={params:<5} "
              f"anneal={str(cfg['use_annealing']):<5} T={cfg['initial_temp']:<4} "
              f"strat={cfg['mutation_strategy']:<10} l1={cfg['l1_lambda']:<6} "
              f"lr={cfg['lr']:<5} H={cfg['hidden_dim']:<2} "
              f"c={cfg['num_chains']} d={cfg['chain_depth']} {tag}", flush=True)

    if not rows:
        print("no config completed")
        return

    rows.sort(key=lambda r: r[0])           # select on VALIDATION
    print("\n" + "=" * 96)
    print(" TOP 5 BY VALIDATION")
    print("=" * 96)
    for v, t, cfg, struct, params in rows[:5]:
        print(f"  val={v:.4e} test={t:.4e} params={params}")
        print(f"    {dict((k, cfg[k]) for k in SPACE)}")
        print(f"    structure: {struct}")

    best_v, best_t, best_cfg, _, _ = rows[0]
    print("\n" + "=" * 96)
    print(" VERDICT (config chosen on validation, then re-run over more seeds)")
    print("=" * 96)
    tests = []
    for s in range(args.final_seeds):
        _, t, struct, params = evaluate(best_cfg, data, s, args.epochs)
        tests.append(t)
        print(f"  seed {s}: test={t:.4e}  {struct}")
    med = float(np.median(tests))
    print(f"\n  median test over {args.final_seeds} seeds: {med:.4e}")
    print(f"  vs SINDy            {sindy_bar:.4e}  -> "
          f"{'WINS by %.3gx' % (sindy_bar / med) if med < sindy_bar else 'loses by %.3gx' % (med / sindy_bar)}")
    print(f"  vs structure_search {ss_bar:.4e}  -> "
          f"{'WINS by %.3gx' % (ss_bar / med) if med < ss_bar else 'loses by %.3gx' % (med / ss_bar)}")
    print(f"\n  search wall time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == '__main__':
    main()
