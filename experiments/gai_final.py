"""
The head-to-head comparison, redone with GAIOptimizer instead of my rewrite.

Every earlier baseline comparison in this repo used training/structure_search.py
-- a reimplementation I wrote -- and it turned out to be the WEAKER algorithm:
on gauss_of_sin the tuned GAIOptimizer beat it 4.92x and beat SINDy 1.19x, while
structure_search lost to SINDy by 4.2x. So those comparisons measured my code,
not the project's. This redoes them with the configurations the grid search
actually found.

Three configurations, taken verbatim from experiments/gai_optimizer_search.py.
Two of them appear in the top three of BOTH task families, so they are robust
rather than task-specific:

  GAI-A  H=32 c=2 d=3, opt_eff=ON   4449 params   gauss #1, lorenz #2
  GAI-B  H=8  c=2 d=2               201 params    gauss #2, lorenz #3
  GAI-C  H=32 c=1 d=3               2339 params   lorenz #1

FAIRNESS RULES, because getting these wrong is what invalidated the earlier runs:

  * Every neural arm reports MEDIAN over the same seeds, and `best` alongside it
    for transparency. Comparing anyone's best-of-N against anyone else's median
    is the min-over-restarts bias, and it silently favoured the MLP in three
    previous experiments here.
  * The MLP selects its ACTIVATION on validation (legitimate model selection)
    but does not get to select its seed. Same treatment as the GAI arms.
  * MLP width is chosen to match GAI-B's parameter count, not left at whatever
    happened to be convenient.
  * SINDy gets two libraries, threshold tuned on validation, and is compared on
    the better of the two.

Run:  python experiments/gai_final.py --seeds 3
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.mlp_comparison import Gaussian, Sin, build_mlp, train_plain
from experiments.sota_baselines import TASKS, run_sindy, split
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import seed_everything
from training.trainer import GAIOptimizer

ACTS = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'sin': Sin, 'gaussian': Gaussian}

GAI_CONFIGS = {
    'GAI-A': dict(lr=0.01, patience=150, grace_period=25, initial_temp=0.5,
                  use_annealing=False, mutation_strategy='importance',
                  l1_lambda=0.0, hidden_dim=32, num_chains=2, chain_depth=3,
                  optimize_efficiency=True),
    'GAI-B': dict(lr=0.01, patience=150, grace_period=10, initial_temp=0.5,
                  use_annealing=True, mutation_strategy='random',
                  l1_lambda=1e-05, hidden_dim=8, num_chains=2, chain_depth=2,
                  optimize_efficiency=False),
    'GAI-C': dict(lr=0.001, patience=60, grace_period=5, initial_temp=0.05,
                  use_annealing=True, mutation_strategy='importance',
                  l1_lambda=1e-05, hidden_dim=32, num_chains=1, chain_depth=3,
                  optimize_efficiency=False),
}


def run_gai(data, name, seeds, epochs, task='', save_dir='results/models'):
    Xtr, ytr, Xva, yva, Xte, yte = data
    tests, structs, params = [], [], None
    best_seen, best_opt = float('inf'), None
    for s in range(seeds):
        seed_everything(s)
        cfg = dict(GAI_CONFIGS[name])
        eff = cfg.pop('optimize_efficiency', False)
        model = MatrixGGLEN(input_dim=Xtr.shape[1], output_dim=ytr.shape[1],
                            hidden_dim=cfg.pop('hidden_dim'),
                            num_chains=cfg.pop('num_chains'),
                            chain_depth=cfg.pop('chain_depth'),
                            rng=random.Random(s))
        opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **cfg)
        opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name=name,
                optimize_efficiency=eff)
        opt.model.eval()
        params = sum(p.numel() for p in opt.model.parameters())
        structs.append(opt.model.get_structure())
        with torch.no_grad():
            t = float(nn.MSELoss()(opt.model(torch.as_tensor(Xte)),
                                   torch.as_tensor(yte)))
        tests.append(t)
        if t < best_seen:
            best_seen, best_opt = t, opt
    if best_opt is not None and task:
        best_opt.save_best(
            os.path.join(save_dir, f"{task}_{name}_{best_seen:.3e}.pt"),
            extra={'task': task, 'config': name, 'test_mse': best_seen})
    return float(np.median(tests)), min(tests), params, structs


def run_mlp(tt, in_d, out_d, width, seeds, epochs):
    """Median over seeds per activation; activation selected on validation."""
    per_act, all_t, params = {}, [], None
    for name, act in ACTS.items():
        vs, ts = [], []
        for s in range(seeds):
            seed_everything(s)
            m = build_mlp(in_d, out_d, width, 2, act)
            params = sum(p.numel() for p in m.parameters())
            v, t = train_plain(m, *tt, epochs=epochs)
            vs.append(v); ts.append(t); all_t.append(t)
        per_act[name] = (float(np.median(vs)), float(np.median(ts)))
    pick = min(per_act, key=lambda k: per_act[k][0])
    return per_act[pick][1], min(all_t), params, pick


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--mlp-epochs', type=int, default=4000)
    ap.add_argument('--mlp-width', type=int, default=12,
                    help='~200 params, matching GAI-B')
    ap.add_argument('--n', type=int, default=4500)
    args = ap.parse_args()

    print("=" * 112)
    print(" GAIOptimizer (tuned)  vs  SINDy  vs  MLP     -- medians over "
          f"{args.seeds} seeds, test split read once")
    print(" every neural arm: median / best.  MLP picks its activation on val,"
          " not its seed.")
    print("=" * 112)
    print(f"{'task':<14} {'SINDy':>11} {'MLP med/best':>20} "
          f"{'GAI-A med/best':>20} {'GAI-B med/best':>20} "
          f"{'GAI-C med/best':>20}  winner")
    print("-" * 112)

    rows = []
    for task in args.tasks:
        fn, kind = TASKS[task]
        X, Y = fn(n=args.n)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
        data = (Xtr, ytr, Xva, yva, Xte, yte)
        T = torch.from_numpy
        tt = (T(Xtr), T(ytr), T(Xva), T(yva), T(Xte), T(yte))

        # run_sindy takes (Xtr, ytr, Xte, yte) positionally with the validation
        # split passed by keyword -- splatting all six shifts Xte/yte into
        # degree/include_trig.
        a64 = lambda a: a.astype(np.float64)
        sindy_args = (a64(Xtr), a64(ytr), a64(Xte), a64(yte))
        sindy_kw = dict(Xva=a64(Xva), yva=a64(yva))
        sg, _, _ = run_sindy(*sindy_args, **sindy_kw)
        sr, _, _ = run_sindy(*sindy_args, rich=True, **sindy_kw)
        sindy = min(sg, sr)

        mlp_m, mlp_b, mlp_p, act = run_mlp(tt, X.shape[1], Y.shape[1],
                                           args.mlp_width, args.seeds,
                                           args.mlp_epochs)
        res = {'SINDy': (sindy, sindy, 12), 'MLP': (mlp_m, mlp_b, mlp_p)}
        for name in GAI_CONFIGS:
            m, b, p, st = run_gai(data, name, args.seeds, args.epochs,
                                  task=task)
            res[name] = (m, b, p)

        win = min(res, key=lambda k: res[k][0])
        print(f"{task:<14} {sindy:>11.3e} "
              f"{mlp_m:>10.3e}/{mlp_b:<9.3e} "
              f"{res['GAI-A'][0]:>10.3e}/{res['GAI-A'][1]:<9.3e} "
              f"{res['GAI-B'][0]:>10.3e}/{res['GAI-B'][1]:<9.3e} "
              f"{res['GAI-C'][0]:>10.3e}/{res['GAI-C'][1]:<9.3e}  {win}",
              flush=True)
        print(f"{'':14} params: SINDy~12  MLP({act})={mlp_p}  "
              f"A={res['GAI-A'][2]}  B={res['GAI-B'][2]}  C={res['GAI-C'][2]}",
              flush=True)
        rows.append((task, kind, res))

    print("\n" + "=" * 112)
    print(" SUMMARY -- how does the best GAI config compare, per task?")
    print("=" * 112)
    print(f"{'task':<14} {'regime':<26} {'best GAI':>11} {'vs SINDy':>14} "
          f"{'vs MLP':>14}")
    for task, kind, res in rows:
        gais = {k: v for k, v in res.items() if k.startswith('GAI')}
        bk = min(gais, key=lambda k: gais[k][0])
        g = gais[bk][0]
        s, m = res['SINDy'][0], res['MLP'][0]
        f = lambda ref: (f"{ref / g:.2f}x better" if g < ref
                         else f"{g / ref:.2g}x worse")
        print(f"{task:<14} {kind:<26} {g:>11.3e} {f(s):>14} {f(m):>14}  ({bk})")


if __name__ == '__main__':
    main()
