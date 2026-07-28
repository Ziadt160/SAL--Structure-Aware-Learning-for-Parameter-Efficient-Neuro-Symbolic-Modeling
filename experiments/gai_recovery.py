"""
Ground-truth operator recovery, with GAIOptimizer at its tuned settings.

Sharper question than "does it beat a baseline": on a target whose correct
operator set is KNOWN, does the search find it? The earlier version of this test
used training/structure_search.py, which has since been shown to be the weaker
algorithm, so its 38%/75% recovery rates described my rewrite rather than the
project's optimiser.

Target:  y = sin(pi*x1) + x2**2,  x ~ U(-1,1)^2
Model:   2 chains, depth 1, hidden_dim 1

hidden_dim=1 is required for the question to be answerable. At wider hidden
layers many operator assignments fit the target equally well, so "did it recover
the right operators" stops being identifiable. At width 1 each chain is a single
scalar operator and MatrixGGLEN sums them before a linear read-out, so

    final(sin(pi*x1) + x2**2) = a*(sin(pi*x1) + x2**2) + c

is exact when one chain picks `sin` and the other `square` -- unique up to
swapping the chains, so the SET of operator classes is compared, not positions.

The tuned configurations supply only the OPTIMISER hyperparameters (lr,
patience, grace_period, temperature, annealing, mutation strategy, l1). The
architecture is fixed at the identifiable one, since the tuned H/chains/depth
were selected for a different task and would destroy identifiability here.

Two things are swept:
  * configuration: the three grid-search winners plus GAIOptimizer's DEFAULTS,
    so "did tuning help?" gets an answer rather than an assumption
  * epoch budget: recovery rate against training length, which is the epochs
    question measured on a task with a known right answer

Recovery is scored by TEST LOSS < 1e-8, not by operator names. The basis
contains equivalences beyond the declared classes -- relu(x^2) == x^2 exactly,
since x^2 >= 0 -- so name matching scores genuine exact solutions as misses.

Run:  python experiments/gai_recovery.py --seeds 8
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.gai_final import GAI_CONFIGS
from models.activations import canonical_class
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import seed_everything
from training.trainer import GAIOptimizer

TARGET_CLASSES = {'periodic', 'quadratic'}
EXACT = 1e-8

# GAIOptimizer's own defaults -- the "did tuning actually help?" control.
DEFAULTS = dict(lr=0.001, patience=30, grace_period=10, initial_temp=0.5,
                use_annealing=True, mutation_strategy='importance',
                l1_lambda=0.0)

OPTIM_KEYS = ('lr', 'patience', 'grace_period', 'initial_temp',
              'use_annealing', 'mutation_strategy', 'l1_lambda')


def make_data(n=900, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    y = (np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1).astype(np.float32)
    a, b = int(0.6 * n), int(0.8 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def arm_params(name):
    if name == 'defaults':
        return dict(DEFAULTS)
    return {k: GAI_CONFIGS[name][k] for k in OPTIM_KEYS}


def run(name, seed, epochs, data, sa='fixed', save_dir='results/models'):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    seed_everything(seed)
    model = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=1,
                        num_chains=2, chain_depth=1, rng=random.Random(seed))
    params = arm_params(name)
    # 'legacy'      = the original acceptance/stagnation behaviour + reset
    # 'fixed'       = rolling stagnation reference, revert cooldown, preserved
    #                 Adam moments, scale-invariant acceptance, finite tabu,
    #                 AND a homotopy operator swap
    # 'fixed-reset' = all of the above EXCEPT the homotopy swap
    #
    # The third arm exists to separate two confounded things. Homotopy preserves
    # the mutated node's weights by construction, which removes the fresh
    # near-zero draw that a reset gives it. At width 1 that draw is what lets
    # sin(pi*w*x) grow into the correct frequency instead of starting in a wrong
    # basin -- i.e. every reset-mutation doubles as a RESTART, and restarts are
    # the only lever at this width. If recovery returns under 'fixed-reset',
    # the homotopy swap is what costs it, not the other SA changes.
    params['legacy_sa'] = (sa == 'legacy')
    params['mutation_mode'] = 'homotopy' if sa == 'fixed' else 'reset'
    opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **params)
    opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name=name)
    opt.model.eval()
    with torch.no_grad():
        test = float(nn.MSELoss()(opt.model(torch.as_tensor(Xte)),
                                  torch.as_tensor(yte)))
    # Persist exact recoveries. Until now every model this project produced was
    # discarded at process exit -- including a run that reached 7.65e-16, an
    # exact closed-form recovery of the target, which no longer exists.
    if test < EXACT:
        path = os.path.join(save_dir,
                            f"recovery_{name}_{sa}_e{epochs}_s{seed}_{test:.2e}.pt")
        opt.save_best(path, extra={'task': 'sin(pi*x1)+x2^2', 'test_mse': test,
                                   'epochs': epochs, 'seed': seed, 'arm': name,
                                   'sa': sa,
                                   'structure': opt.model.get_structure()})
    return test, opt.model.get_structure()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--epochs-list', nargs='+', type=int,
                    default=[2000, 6000, 15000])
    ap.add_argument('--arms', nargs='+',
                    default=['defaults', 'GAI-A', 'GAI-B', 'GAI-C'])
    ap.add_argument('--sa', nargs='+', default=['legacy', 'fixed'],
                    choices=['legacy', 'fixed', 'fixed-reset'])
    ap.add_argument('--n', type=int, default=900)
    ap.add_argument('--dump', default=None,
                    help='write per-seed test losses to this JSON path')
    args = ap.parse_args()

    data = make_data(args.n)
    print("=" * 96)
    print(" GAIOptimizer OPERATOR RECOVERY -- target y = sin(pi*x1) + x2^2")
    print(f" architecture fixed at H=1, 2 chains, depth 1 (identifiable);"
          f" {args.seeds} seeds")
    print(f" recovery = test loss < {EXACT:g}   (name matching misses"
          f" relu(x^2) == x^2 and similar)")
    print("=" * 96)
    print(f"{'arm':<10} {'SA':<7} {'epochs':>7} {'recovery':>9} "
          f"{'median test':>13} {'best test':>12}  example structures")
    print("-" * 100)

    grid = {}
    per_seed: dict = {}
    for name in args.arms:
        for sa in args.sa:
            for epochs in args.epochs_list:
                losses, structs, hits = [], [], 0
                for s in range(args.seeds):
                    t, st = run(name, s, epochs, data, sa)
                    losses.append(t); structs.append(st)
                    hits += (t < EXACT)
                rate = hits / args.seeds
                grid[(name, sa, epochs)] = (rate, float(np.median(losses)),
                                            float(min(losses)))
                # Keep the per-seed losses. Recovery COUNTS are a coarse metric
                # -- 3/28 vs 1/28 is p=0.61 -- while the median test loss can
                # move 1.8x across the same arms. Testing that needs the raw
                # per-seed values, which this used to discard, leaving the
                # median differences permanently unfalsifiable.
                per_seed[f"{name}|{sa}|{epochs}"] = losses
                ex = '; '.join(','.join(c[0] for c in st) for st in structs[:3])
                print(f"{name:<10} {sa:<7} {epochs:>7} {hits:>4}/{args.seeds}   "
                      f"{np.median(losses):>13.3e} {min(losses):>12.3e}  {ex}",
                      flush=True)

    print("\n" + "=" * 96)
    print(" DID TUNING HELP?  (recovery rate / median test, vs defaults)")
    print("=" * 96)
    header = ' '.join(f"{e:>13}" for e in args.epochs_list)
    print(f"{'arm':<10} {'SA':<7} {header}")
    for name in args.arms:
        for sa in args.sa:
            cells = []
            for e in args.epochs_list:
                r, med, _ = grid[(name, sa, e)]
                cells.append(f"{r:>6.0%}/{med:<6.0e}")
            print(f"{name:<10} {sa:<7} " + ' '.join(f"{c:>13}" for c in cells))

    print("\n DOES MORE TRAINING HELP?  (per arm, recovery across budgets)")
    for name in args.arms:
        for sa in args.sa:
            rates = [f"{grid[(name, sa, e)][0]:.0%}@{e}"
                     for e in args.epochs_list]
            meds = [grid[(name, sa, e)][1] for e in args.epochs_list]
            trend = ('monotone improvement'
                     if meds == sorted(meds, reverse=True)
                     else 'non-monotone -- more epochs is not reliably better')
            print(f"  {name:<10} {sa:<7} {' -> '.join(rates):<40} {trend}")

    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or '.', exist_ok=True)
        with open(args.dump, 'w') as fh:
            json.dump(per_seed, fh, indent=1)
        print(f"\n  per-seed test losses -> {args.dump}")
        print("  (recovery counts are coarse: 3/28 vs 1/28 is p=0.61. Use these"
              " for a rank test on the medians, which move further than the"
              " counts do.)")


if __name__ == '__main__':
    main()
