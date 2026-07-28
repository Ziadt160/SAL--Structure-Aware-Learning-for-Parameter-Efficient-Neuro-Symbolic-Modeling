"""
Is the MatrixChain worth it, or would a normal deep model do?

The architecture bundles three separable claims. This measures them apart:

  arm 1  MLP                  plain nn.Sequential, tanh or relu
  arm 2  chain, fixed ops     MatrixChain with RANDOM operators, no search
  arm 3  chain + search       MatrixChain with the Phase 1 operator search

  arm 2 vs arm 1  ->  does the chain ARCHITECTURE help, on its own?
  arm 3 vs arm 2  ->  does the operator SEARCH help, on top of it?

Tasks are chosen so the answer can come out either way:

  analytic   y = sin(pi*x1) + x2**2         exactly a composition of the basis
  lorenz     (x,y,z) -> (dx,dy,dz)          polynomial, has xy and xz products
  teacher    y = random tanh-MLP(x)         ground truth IS an MLP -- the
                                            adversarial case for the chain

Arms 1 and 2 get an identical epoch budget. Arm 3 additionally spends epochs
on search, so its total is reported: it is not a compute-matched comparison
and pretending otherwise would flatter it.

Run:  python experiments/mlp_comparison.py --seeds 3
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything)


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
def task_analytic(n=900, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    y = (np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1).astype(np.float32)
    return X, y


def task_lorenz(n=1500, seed=0):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt = 0.01
    s = np.array([1.0, 1.0, 1.0])

    def f(v):
        x, y, z = v
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    states = []
    for _ in range(n):                     # RK4, no scipy dependency
        k1 = f(s); k2 = f(s + dt * k1 / 2)
        k3 = f(s + dt * k2 / 2); k4 = f(s + dt * k3)
        s = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(s.copy())
    X = np.array(states, dtype=np.float32)
    Y = np.array([f(v) for v in X], dtype=np.float32)
    # Standardise BOTH sides. Raw Lorenz states reach |x|~20 and z~50; fed
    # straight into a bounded activation every unit saturates, and into
    # square/gaussian it explodes into the soft-clip. Without this the
    # comparison measures input scaling, not architecture. (The repo's
    # use_cases/physics/gai_lorrenz.py feeds the raw states in.)
    X = (X - X.mean(0)) / X.std(0)
    Y = (Y - Y.mean(0)) / Y.std(0)
    return X.astype(np.float32), Y.astype(np.float32)


def task_teacher(n=900, seed=0):
    g = torch.Generator().manual_seed(1234)      # same teacher for all seeds
    teacher = nn.Sequential(nn.Linear(3, 12), nn.Tanh(),
                            nn.Linear(12, 12), nn.Tanh(), nn.Linear(12, 1))
    with torch.no_grad():
        for p in teacher.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.8)
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32)
    with torch.no_grad():
        y = teacher(torch.from_numpy(X)).numpy().astype(np.float32)
    return X, y


TASKS = {'analytic': task_analytic, 'lorenz': task_lorenz, 'teacher': task_teacher}


def split(X, y):
    n = len(X)
    a, b = int(0.6 * n), int(0.8 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


# --------------------------------------------------------------------------
def count(m):
    return sum(p.numel() for p in m.parameters())


def train_plain(model, Xtr, ytr, Xva, yva, Xte, yte, epochs, lr=0.01):
    """Full-batch training; select on val, report test at the val optimum."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_val, best_test = float('inf'), float('nan')
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        crit(model(Xtr), ytr).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = float(crit(model(Xva), yva))
            if v < best_val:
                best_val = v
                best_test = float(crit(model(Xte), yte))
    return best_val, best_test


class Sin(nn.Module):
    """Matches models.activations 'sin', including the pi scaling."""
    def forward(self, x):
        return torch.sin(x * torch.pi)


class Gaussian(nn.Module):
    def forward(self, x):
        return torch.exp(-x ** 2)


def build_mlp(in_dim, out_dim, width, depth, act):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
def run_task(name, seeds, epochs, verbose=False):
    rows = []
    print(f"[{name}] starting ({seeds} seeds, {epochs} epochs/arm)", flush=True)
    for seed in range(seeds):
        print(f"[{name}] seed {seed}", flush=True)
        X, y = TASKS[name](seed=seed)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
        T = lambda a: torch.from_numpy(a)
        tt = (T(Xtr), T(ytr), T(Xva), T(yva), T(Xte), T(yte))
        in_dim, out_dim = X.shape[1], y.shape[1]

        # arm 1: plain MLPs, one global activation each.
        # sin and gaussian are included because the decisive control is not
        # "chain vs relu MLP" but "per-node search vs the best SINGLE
        # activation" -- on the analytic task relu and tanh already differ 7x,
        # so activation choice alone explains a lot of the gap.
        for act_name, act in (('tanh', nn.Tanh), ('relu', nn.ReLU),
                              ('sin', Sin), ('gaussian', Gaussian)):
            for width in (4, 8, 16, 32):
                seed_everything(seed)
                m = build_mlp(in_dim, out_dim, width, 2, act)
                v, t = train_plain(m, *tt, epochs=epochs)
                rows.append(dict(arm=f'mlp-{act_name}', cfg=f'w{width}',
                                 seed=seed, params=count(m), val=v, test=t,
                                 epochs=epochs))

        # arm 2: chain architecture, random operators, no search
        for h in (2, 4, 8):
            seed_everything(seed)
            m = MatrixGGLEN(input_dim=in_dim, output_dim=out_dim, hidden_dim=h,
                            num_chains=2, chain_depth=2, rng=random.Random(seed))
            v, t = train_plain(m, *tt, epochs=epochs)
            rows.append(dict(arm='chain-fixed', cfg=f'h{h}', seed=seed,
                             params=count(m), val=v, test=t, epochs=epochs))

        # arm 3: chain + operator search
        for h in (4, 8):
            seed_everything(seed)
            cfg = SearchConfig(seed=seed, warmup_epochs=200, probe_epochs=40,
                               probe_restarts=2, consolidate_epochs=200,
                               max_op_sweeps=2, use_composites=False,
                               topology_rounds=0, allow_growth=False,
                               allow_pruning=False, compress=False,
                               verbose=verbose)
            m = MatrixGGLEN(input_dim=in_dim, output_dim=out_dim, hidden_dim=h,
                            num_chains=2, chain_depth=2, rng=random.Random(seed))
            s = StructureSearch(m, Xtr, ytr, Xva, yva, Xte, yte, cfg)
            tr = s.run()
            rows.append(dict(arm='chain-search', cfg=f'h{h}', seed=seed,
                             params=tr.params, val=tr.val_loss, test=tr.test_loss,
                             epochs=tr.train_epochs,
                             struct=str(tr.final_structure)))
            print(f"    search h{h}: test={tr.test_loss:.3e} "
                  f"epochs={tr.train_epochs} {tr.final_structure}", flush=True)
    return rows


def summarise(name, rows):
    print(f"\n{'=' * 84}\n TASK: {name}\n{'=' * 84}")
    print(f"{'arm':<14} {'cfg':>5} {'params':>7} {'epochs':>8} "
          f"{'median test':>13} {'best test':>12}")
    print("-" * 84)
    groups = {}
    for r in rows:
        groups.setdefault((r['arm'], r['cfg']), []).append(r)
    best_by_arm = {}
    for (arm, cfg), rs in sorted(groups.items()):
        med = float(np.median([r['test'] for r in rs]))
        best = float(np.min([r['test'] for r in rs]))
        print(f"{arm:<14} {cfg:>5} {rs[0]['params']:>7} "
              f"{int(np.mean([r['epochs'] for r in rs])):>8} "
              f"{med:>13.3e} {best:>12.3e}")
        if arm not in best_by_arm or med < best_by_arm[arm][0]:
            best_by_arm[arm] = (med, cfg, rs[0]['params'])

    # "Try a handful of activations, keep the best on VAL" -- the cheap
    # alternative to per-node search. Selection uses val, never test.
    print("-" * 84)
    by_seed_width = {}
    for r in rows:
        if r['arm'].startswith('mlp-'):
            by_seed_width.setdefault((r['seed'], r['cfg']), []).append(r)
    for cfg in sorted({c for _, c in by_seed_width}, key=lambda c: int(c[1:])):
        picked = [min((r for r in by_seed_width[(s, cfg)]), key=lambda r: r['val'])
                  for s in {s for s, c in by_seed_width if c == cfg}]
        med = float(np.median([r['test'] for r in picked]))
        acts = ','.join(sorted({r['arm'].split('-')[1] for r in picked}))
        print(f"{'mlp-BEST-act':<14} {cfg:>5} {picked[0]['params']:>7} "
              f"{picked[0]['epochs']:>8} {med:>13.3e} "
              f"{min(r['test'] for r in picked):>12.3e}   picked: {acts}")
        if 'mlp-BEST' not in best_by_arm or med < best_by_arm['mlp-BEST'][0]:
            best_by_arm['mlp-BEST'] = (med, cfg, picked[0]['params'])
    return best_by_arm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=1200)
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    verdicts = {}
    for name in args.tasks:
        rows = run_task(name, args.seeds, args.epochs, args.verbose)
        verdicts[name] = summarise(name, rows)

    print(f"\n{'=' * 84}\n VERDICT: best median test loss per arm (params in brackets)\n{'=' * 84}")
    print(f"{'task':<10} {'mlp-best-act':>20} {'chain-fixed':>20} "
          f"{'chain-search':>20}   winner")
    for name, b in verdicts.items():
        mlp = b.get('mlp-BEST', (float('nan'), '', 0))
        cf = b.get('chain-fixed', (float('nan'), '', 0))
        cs = b.get('chain-search', (float('nan'), '', 0))
        cands = {'mlp-best-act': mlp[0], 'chain-fixed': cf[0],
                 'chain-search': cs[0]}
        winner = min(cands, key=lambda k: cands[k])
        fmt = lambda t: f"{t[0]:.3e} [{t[2]}]"
        print(f"{name:<10} {fmt(mlp):>20} {fmt(cf):>20} {fmt(cs):>20}"
              f"   {winner}")
    print("\n  chain-fixed  vs mlp-best-act -> is the ARCHITECTURE worth anything?")
    print("  chain-search vs chain-fixed   -> is the SEARCH worth anything?")
    print("  chain-search vs mlp-best-act  -> is per-node search worth more than")
    print("                                   just trying 4 global activations?")
    print("  chain-search spends ~5x the epochs; see the epochs column.")


if __name__ == '__main__':
    main()
