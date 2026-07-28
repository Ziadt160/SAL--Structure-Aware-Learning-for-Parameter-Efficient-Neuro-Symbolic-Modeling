"""
The decisive experiment: learnable frequency and direction vs an enumerated library.

Every other niche proposed for this project died on contact with a fairly
configured SINDy. This one did not. On `sin(3*pi*x) + 0.5*sin(11*pi*x)` a rich
SINDy library reached NMSE 0.186 with coefficients like -3,959,827, while the
activation-search methods reached 0.0001 -- a ~1900x gap, because the library
held sin at pi, 2pi, 3pi but not at 11pi. One missing frequency wrecks a
least-squares fit.

The mechanism is structural, and it has two parts:

  FREQUENCY   a searchable `sin` sets its frequency in the WEIGHTS, w in
              sin(pi * w * x). A library holds only frequencies chosen up front.
  DIRECTION   in d dimensions the argument is sin(pi * w . x) for a continuous
              direction w in R^d. A library would have to enumerate directions
              as well as frequencies -- a continuous, d-dimensional grid.

So this measures the honest boundary. SINDy is given three libraries, including a
DENSE frequency grid that contains the integer frequencies exactly. Two frequency
regimes are run:

  integer     f = 3, 11        -- the dense grid contains them; SINDy should win
  irrational  f = 2.7, 7.3     -- the grid misses; the gap should open

and d is swept, because at d=1 the direction is trivial and at d>1 it is not.

If the dense grid wins everywhere, the niche is dead and so is the paper. If it
loses only at irrational frequencies, the claim is narrow but real. If it loses
as d grows even at integer frequencies, the claim is structural.

Run:  python experiments/frequency_niche.py --dims 1 2 4 --regimes integer irrational
"""

import argparse
import itertools
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.mlp_comparison import Gaussian, Sin, build_mlp, train_plain
from experiments.sota_baselines import split, stlsq
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import seed_everything
from training.trainer import GAIOptimizer

ACTS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sin': Sin, 'gaussian': Gaussian}

FREQS = {'integer': (3.0, 11.0), 'irrational': (2.7, 7.3)}

# Best configuration found by experiments/gai_optimizer_search.py on
# gauss_of_sin, selected on validation: test 5.701e-06 at 201 parameters, which
# beat both structure_search (2.550e-05) and SINDy (6.145e-06). Notably small,
# with annealing ON, random mutation, and non-zero L1 -- three things earlier
# single-setting runs had wrong.
# Verbatim from the grid search's top-2 by validation. An earlier version of
# this file guessed patience=30 because it was not in the line being read; the
# real value is 150. The #1 config was also omitted entirely.
GAI_CONFIGS = {
    # #1 by validation: test 2.981e-06 on gauss_of_sin, 4449 params
    'GAI-best': dict(lr=0.01, patience=150, grace_period=25, initial_temp=0.5,
                     use_annealing=False, mutation_strategy='importance',
                     l1_lambda=0.0, hidden_dim=32, num_chains=2, chain_depth=3,
                     optimize_efficiency=True),
    # #2: test 5.701e-06 at only 201 params -- best parameter efficiency
    'GAI-small': dict(lr=0.01, patience=150, grace_period=10, initial_temp=0.5,
                      use_annealing=True, mutation_strategy='random',
                      l1_lambda=1e-05, hidden_dim=8, num_chains=2,
                      chain_depth=2, optimize_efficiency=False),
}


def make_task(d: int, regime: str, n: int, seed: int):
    """y = sin(pi*f1*(a.x)) + 0.5*sin(pi*f2*(b.x)) with random unit directions."""
    f1, f2 = FREQS[regime]
    rng = np.random.RandomState(1000 + seed)
    a = rng.randn(d); a /= np.linalg.norm(a)
    b = rng.randn(d); b /= np.linalg.norm(b)
    X = rng.uniform(-1, 1, (n, d)).astype(np.float32)
    y = (np.sin(np.pi * f1 * (X @ a)) + 0.5 * np.sin(np.pi * f2 * (X @ b)))
    return X, y.reshape(-1, 1).astype(np.float32), (a, b)


# --------------------------------------------------------------------------
def sindy_library(X, kind: str, max_k: int = 15):
    """generic | rich | dense -- increasingly generous candidate term sets."""
    n, d = X.shape
    feats, names = [np.ones(n)], ['1']
    for i in range(d):
        feats.append(X[:, i]); names.append(f'x{i}')
    for i in range(d):
        for j in range(i, d):
            feats.append(X[:, i] * X[:, j]); names.append(f'x{i}x{j}')
    if kind in ('rich', 'dense'):
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    feats.append(X[:, i] * X[:, j] * X[:, k])
                    names.append(f'x{i}x{j}x{k}')
    if kind == 'generic':
        for i in range(d):
            feats.append(np.sin(X[:, i])); names.append(f'sin(x{i})')
            feats.append(np.cos(X[:, i])); names.append(f'cos(x{i})')
    elif kind == 'rich':
        for i in range(d):
            for m in (1, 2, 3):
                feats.append(np.sin(m * np.pi * X[:, i]))
                names.append(f'sin({m}pi x{i})')
                feats.append(np.cos(m * np.pi * X[:, i]))
                names.append(f'cos({m}pi x{i})')
            feats.append(np.exp(-X[:, i] ** 2)); names.append(f'exp(-x{i}^2)')
    elif kind == 'dense':
        # The strongest honest SINDy: every integer frequency up to max_k on
        # every input axis. At d=1 with integer target frequencies this contains
        # the answer EXACTLY, so SINDy should win there -- that is the control
        # that keeps this from being a strawman.
        for i in range(d):
            for m in range(1, max_k + 1):
                feats.append(np.sin(m * np.pi * X[:, i]))
                names.append(f'sin({m}pi x{i})')
                feats.append(np.cos(m * np.pi * X[:, i]))
                names.append(f'cos({m}pi x{i})')
    return np.stack(feats, axis=1), names


def run_sindy(Xtr, ytr, Xva, yva, Xte, yte, kind):
    Th_tr, names = sindy_library(Xtr, kind)
    Th_va, _ = sindy_library(Xva, kind)
    Th_te, _ = sindy_library(Xte, kind)
    best = (np.inf, None)
    for th in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        xi = stlsq(Th_tr, ytr[:, 0], threshold=th)
        v = float(np.mean((Th_va @ xi - yva[:, 0]) ** 2))
        if v < best[0]:
            best = (v, xi)
    xi = best[1]
    mse = float(np.mean((Th_te @ xi - yte[:, 0]) ** 2))
    return mse, int(np.count_nonzero(xi)), Th_tr.shape[1]


def run_mlp(tt, d, seeds, epochs, width):
    """Returns (median-over-seeds, best-of-all, params).

    BOTH statistics are returned because reporting only one is how the previous
    version of this experiment went wrong: the MLP was scored as the minimum of
    12 runs while GAI was scored as the median of 3. A max statistic against a
    central statistic is not a comparison -- it is the same min-over-restarts
    bias diagnosed in the candidate probe, reproduced at the experiment level.
    Per activation we take the median across seeds, then select the ACTIVATION on
    validation; `best` additionally takes the luckiest single run.
    """
    per_act, all_runs, params = {}, [], None
    for name, act in ACTS.items():
        vs, ts = [], []
        for s in range(seeds):
            seed_everything(s)
            m = build_mlp(d, 1, width, 2, act)
            params = sum(p.numel() for p in m.parameters())
            v, t = train_plain(m, *tt, epochs=epochs)
            vs.append(v); ts.append(t); all_runs.append((v, t))
        per_act[name] = (float(np.median(vs)), float(np.median(ts)))
    pick = min(per_act, key=lambda k: per_act[k][0])       # select on val
    return per_act[pick][1], min(t for _, t in all_runs), params, pick


def run_gai(data, seeds, epochs, cfg_name):
    """Returns (median-over-seeds, best-of-seeds, params) -- same statistics as
    the MLP arm, so the two are actually comparable."""
    Xtr, ytr, Xva, yva, Xte, yte = data
    tests, params = [], None
    for s in range(seeds):
        seed_everything(s)
        cfg = dict(GAI_CONFIGS[cfg_name])
        eff = cfg.pop('optimize_efficiency', False)
        model = MatrixGGLEN(input_dim=Xtr.shape[1], output_dim=1,
                            hidden_dim=cfg.pop('hidden_dim'),
                            num_chains=cfg.pop('num_chains'),
                            chain_depth=cfg.pop('chain_depth'),
                            rng=random.Random(s))
        opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **cfg)
        opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name='gai',
                optimize_efficiency=eff)
        opt.model.eval()
        params = sum(p.numel() for p in opt.model.parameters())
        with torch.no_grad():
            tests.append(float(nn.MSELoss()(opt.model(torch.as_tensor(Xte)),
                                            torch.as_tensor(yte))))
    return float(np.median(tests)), min(tests), params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dims', nargs='+', type=int, default=[1, 2, 4])
    ap.add_argument('--regimes', nargs='+', default=['integer', 'irrational'])
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--mlp-epochs', type=int, default=6000,
                    help='raised because NMSE ~0.2 at d>=2 in the earlier run '
                         'indicated both neural arms were undertrained')
    ap.add_argument('--mlp-width', type=int, default=12,
                    help='12 gives ~205 params at d=2, matching GAI-small (201); '
                         'the earlier run used 16 (337 params) against 201')
    args = ap.parse_args()

    print("=" * 108)
    print(" LEARNABLE FREQUENCY/DIRECTION  vs  ENUMERATED LIBRARY")
    print(" target: y = sin(pi*f1*(a.x)) + 0.5*sin(pi*f2*(b.x)), a,b random unit"
          " directions")
    print(" integer f=(3,11): the dense grid contains these."
          "   irrational f=(2.7,7.3): it does not.")
    print(" NMSE (1.0 = predicting the mean). Every neural arm reports"
          " median-over-seeds / best-of-seeds.")
    print("=" * 108)
    print(f"{'d':>3} {'regime':<11} {'terms':>6} {'SINDy-gen':>10} "
          f"{'SINDy-rich':>10} {'SINDy-dense':>11} | "
          f"{'MLP med/best':>19} {'GAI-small med/best':>21} "
          f"{'GAI-best med/best':>21}   winner(median)")
    print("-" * 108)

    for d, regime in itertools.product(args.dims, args.regimes):
        X, Y, _ = make_task(d, regime, args.n, 0)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
        data = (Xtr, ytr, Xva, yva, Xte, yte)
        var = float(Y.var())
        T = torch.from_numpy
        tt = (T(Xtr), T(ytr), T(Xva), T(yva), T(Xte), T(yte))

        d64 = [a.astype(np.float64) for a in (Xtr, ytr, Xva, yva, Xte, yte)]
        sg, _, _ = run_sindy(*d64, 'generic')
        sr, _, _ = run_sindy(*d64, 'rich')
        sd, _, nterms = run_sindy(*d64, 'dense')
        mlp_med, mlp_best, mlp_p, act = run_mlp(tt, d, args.seeds,
                                                args.mlp_epochs, args.mlp_width)
        gs_med, gs_best, gs_p = run_gai(data, args.seeds, args.epochs, 'GAI-small')
        gb_med, gb_best, gb_p = run_gai(data, args.seeds, args.epochs, 'GAI-best')

        # Winner decided on the MEDIAN only -- comparing anyone's best-of-N
        # against anyone else's median is the bias this rewrite exists to remove.
        med = {'SINDy-gen': sg / var, 'SINDy-rich': sr / var,
               'SINDy-dense': sd / var, 'MLP': mlp_med / var,
               'GAI-small': gs_med / var, 'GAI-best': gb_med / var}
        win = min(med, key=lambda k: med[k])
        print(f"{d:>3} {regime:<11} {nterms:>6} {sg/var:>10.4f} {sr/var:>10.4f} "
              f"{sd/var:>11.4f} | "
              f"{mlp_med/var:>9.4f}/{mlp_best/var:<9.4f} "
              f"{gs_med/var:>10.4f}/{gs_best/var:<10.4f} "
              f"{gb_med/var:>10.4f}/{gb_best/var:<10.4f}   {win}", flush=True)
        print(f"      params: MLP({act})={mlp_p}  GAI-small={gs_p}  "
              f"GAI-best={gb_p}  SINDy-dense terms={nterms}", flush=True)

    print("\n" + "=" * 108)
    print(" HOW TO READ THIS")
    print("=" * 108)
    print("  Compare MEDIANS across arms. The med/best pairs are shown so the")
    print("  spread is visible, not so the flattering number can be quoted.")
    print()
    print("  SINDy-dense wins everywhere       -> niche is dead")
    print("  SINDy-dense wins only at d=1      -> the library cannot enumerate a")
    print("                                       continuous PROJECTION DIRECTION")
    print("  GAI ~= MLP at d>=2                -> the win belongs to neural nets,")
    print("                                       not to operator search")
    print("  GAI < MLP at matched params       -> operator search earns its keep")


if __name__ == '__main__':
    main()
