"""
Is the operator search doing anything that random restarts would not?

This is the control the project has never run, and the evidence says it is the
decisive one:

  * 0 of 126 real mutations ever produced a new global best (GAI-A, legacy SA,
    6 seeds x 4000 epochs). On every seed inspected, the run's best model was
    already reached BEFORE its first mutation fired.
  * Yet exact recovery of `sin(pi*x1) + x2^2` still happens 3/8 of the time at
    hidden_dim=1 -- machine precision, correct structure `sin, square`.
  * Recovery collapses to 0/8 when the mutated node keeps its weights (Xavier
    draw too wide, or homotopy preserving them) and returns when the node gets
    a fresh near-zero draw.

Those three facts fit one explanation: a reset-mutation is not selecting a
better operator, it is performing a RANDOM RESTART of one node, and restarts are
the only lever at width 1 (theory/structural_learning.md). If that is right,
spending the same epoch budget on plain independent restarts should recover at
least as often -- with no search, no annealing, and no importance ranking.

The comparison is budget-matched: GAI gets ONE run of `--epochs`, the baseline
gets K runs of `--epochs // K` each and keeps the best. Both see the same number
of gradient steps, so any difference is attributable to the search.

  ARMS
    gai         GAIOptimizer, legacy SA (the arm that actually recovers)
    restart-K   K independent models, random operator assignment per node,
                plain Adam, no mutation. Best of K by validation.
    restart-T   as above but every node forced to the TRUE operators
                [sin, square] -- an upper bound on what restarts can do once
                operator selection is free.

Run:  python experiments/restart_baseline.py --seeds 8 --epochs 6000 --k 8
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.gai_recovery import EXACT, arm_params, make_data
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import seed_everything
from training.trainer import GAIOptimizer

TRUE_OPS = ['sin', 'square']
BASIS = ['identity', 'tanh', 'relu', 'sin', 'gaussian', 'square']


def _build(seed):
    return MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=1, num_chains=2,
                       chain_depth=1, rng=random.Random(seed))


def _test_loss(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        return float(nn.MSELoss()(model(torch.as_tensor(Xte)),
                                  torch.as_tensor(yte)))


def run_gai(seed, epochs, data, cfg='GAI-A'):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    seed_everything(seed)
    model = _build(seed)
    params = arm_params(cfg)
    params['legacy_sa'] = True
    params['mutation_mode'] = 'reset'
    opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **params)
    opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name=f'gai/s{seed}')
    return _test_loss(opt.model, Xte, yte), len(opt.mutation_log)


def run_restarts(seed, epochs, data, k, forced=False, lr=0.01):
    """K independent trainings, no mutation at all. Best-of-K by VALIDATION.

    Selecting on validation (not test) matters: picking best-of-K on the test
    split is the leakage that made an earlier result in this project look far
    better than it was.
    """
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    per = max(1, epochs // k)
    Xtr_t, ytr_t = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    Xva_t, yva_t = torch.as_tensor(Xva), torch.as_tensor(yva)
    crit = nn.MSELoss()

    best_val, best_test = float('inf'), float('nan')
    for r in range(k):
        seed_everything(seed * 1000 + r)
        model = _build(seed * 1000 + r)
        for ci, li, node in model.iter_nodes():
            node.set_op(TRUE_OPS[ci % len(TRUE_OPS)] if forced
                        else random.choice(BASIS))
            node.reset_weights_near_identity()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(per):
            model.train()
            opt.zero_grad()
            crit(model(Xtr_t), ytr_t).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(crit(model(Xva_t), yva_t))
        if v < best_val:
            best_val, best_test = v, _test_loss(model, Xte, yte)
    return best_test


def run_search_restarts(seed, epochs, data, k, cfg='GAI-A'):
    """K independent GAIOptimizer searches on a K-way split budget, best-of-K.

    The measured result this exists to act on: at a fixed budget, restarts are
    worth 1/8 -> 8/8 once operators are correct, and the mutation loop forfeits
    that entirely by spending everything on ONE trajectory. This is the cheapest
    possible way to give it back -- no algorithmic change, just don't put all
    the epochs in one run.

    Selection is on VALIDATION, never test.
    """
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    per = max(1, epochs // k)
    best_val, best_test, total_muts = float('inf'), float('nan'), 0
    for r in range(k):
        s = seed * 1000 + r
        seed_everything(s)
        model = _build(s)
        params = arm_params(cfg)
        params['legacy_sa'] = True
        params['mutation_mode'] = 'reset'
        opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **params)
        opt.fit(Xtr, ytr, Xva, yva, epochs=per, name=f'searchR/s{seed}r{r}')
        total_muts += len(opt.mutation_log)
        opt.model.eval()
        with torch.no_grad():
            v = float(nn.MSELoss()(opt.model(torch.as_tensor(Xva)),
                                   torch.as_tensor(yva)))
        if v < best_val:
            best_val, best_test = v, _test_loss(opt.model, Xte, yte)
    return best_test, total_muts


def run_search_then_restart(seed, epochs, data, k, cfg='GAI-A'):
    """Let the search pick the operators, then restart the WEIGHTS on them.

    This is the high-power version of "does the search work?".

    End-to-end exact recovery is a conjunction -- correct operators AND a lucky
    weight basin -- so it is a rare event (11% vs 0%), and separating those at
    n=28 gives p=0.24. But the second half is already known to be solved: given
    correct operators, restarts recover 28/28. So freeze whatever operators the
    search actually selected and run restarts on top. Every seed whose operators
    are right becomes a recovery, which converts the rare event into a
    near-deterministic readout of operator quality.

        result ~ 28/28  ->  the search picks correct operators; the 11%
                            end-to-end figure was weight-optimisation luck, and
                            appending restarts fixes it outright.
        result ~ 0/28   ->  the search picks WRONG operators; selection needs
                            replacing, not tuning.

    NOT budget-matched by design: it spends `epochs` on the search and another
    `epochs` on restarts. It is a diagnostic of operator quality, not a proposed
    configuration -- the budget-matched version is `search-x`.

    Returns (test_loss, ops_selected).
    """
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    seed_everything(seed)
    model = _build(seed)
    params = arm_params(cfg)
    params['legacy_sa'] = True
    params['mutation_mode'] = 'reset'
    opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **params)
    opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name=f'searchThen/s{seed}')

    # Whatever the search settled on, as a flat list of operator names.
    ops = [n.op_name for _, _, n in opt.model.iter_nodes()]

    # Now redo the weights from scratch on exactly those operators.
    per = max(1, epochs // k)
    Xtr_t, ytr_t = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    Xva_t, yva_t = torch.as_tensor(Xva), torch.as_tensor(yva)
    crit = nn.MSELoss()
    best_val, best_test = float('inf'), float('nan')
    for r in range(k):
        s = seed * 1000 + r
        seed_everything(s)
        m2 = _build(s)
        for i, (_, _, node) in enumerate(m2.iter_nodes()):
            node.set_op(ops[i])
            node.reset_weights_near_identity()
        o2 = torch.optim.Adam(m2.parameters(), lr=params['lr'])
        for _ in range(per):
            m2.train(); o2.zero_grad()
            crit(m2(Xtr_t), ytr_t).backward(); o2.step()
        m2.eval()
        with torch.no_grad():
            v = float(crit(m2(Xva_t), yva_t))
        if v < best_val:
            best_val, best_test = v, _test_loss(m2, Xte, yte)
    return best_test, ops


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=6000)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--n', type=int, default=900)
    ap.add_argument('--arms', nargs='+', default=None,
                    choices=['gai', 'search-x', 'search-then', 'restart',
                             'restart-true'],
                    help='subset to run; default is all five')
    args = ap.parse_args()

    data = make_data(args.n)
    print("=" * 92)
    print(" IS THE SEARCH BEATING RANDOM RESTARTS?  y = sin(pi*x1) + x2^2")
    print(f" budget-matched: gai = 1 x {args.epochs} epochs;"
          f" restart = {args.k} x {args.epochs // args.k} epochs, best-of-K")
    print(f" recovery = test loss < {EXACT:g};  {args.seeds} seeds")
    print("=" * 92)
    print(f"{'arm':<14} {'recovery':>9} {'median test':>14} {'best test':>13}"
          f"  {'note':<28}")
    print("-" * 92)

    all_arms = {'gai': 'gai',
                'search-x': f'search-x{args.k}',
                'search-then': f'search-then-restart{args.k}',
                'restart': f'restart-{args.k}',
                'restart-true': f'restart-{args.k}-true'}
    selected = [all_arms[a] for a in (args.arms or list(all_arms))]

    rows = {}
    selections: list = []
    for label in selected:
        losses, muts, picked = [], [], []
        for s in range(args.seeds):
            if label == 'gai':
                t, m = run_gai(s, args.epochs, data)
                muts.append(m)
            elif label.startswith('search-then'):
                t, ops = run_search_then_restart(s, args.epochs, data, args.k)
                picked.append(ops)
            elif label.startswith('search-x'):
                t, m = run_search_restarts(s, args.epochs, data, args.k)
                muts.append(m)
            else:
                t = run_restarts(s, args.epochs, data, args.k,
                                 forced=label.endswith('true'))
            losses.append(t)
        hits = sum(1 for t in losses if t < EXACT)
        rows[label] = (hits, float(np.median(losses)), float(min(losses)))
        note = (f"{sum(muts)} real mutations total" if muts
                else ('operators forced to truth' if label.endswith('true')
                      else 'random operators, no search'))
        if label.startswith('search-x'):
            note = f"search x{args.k} restarts, {sum(muts)} muts"
        if label.startswith('search-then'):
            # How often did the search land on the ground-truth operator set?
            # Order across chains is irrelevant -- the chains are summed.
            hits = sum(1 for o in picked
                       if sorted(o) == sorted(TRUE_OPS))
            note = f"exact op-set {hits}/{args.seeds}, then restarts"
            selections.extend(picked)
        print(f"{label:<14} {hits:>4}/{args.seeds}   {np.median(losses):>14.3e} "
              f"{min(losses):>13.3e}  {note:<28}", flush=True)

    if selections:
        print("\n" + "=" * 92)
        print(" WHAT THE SEARCH ACTUALLY SELECTED")
        print("=" * 92)
        from collections import Counter
        c = Counter(tuple(sorted(o)) for o in selections)
        truth = tuple(sorted(TRUE_OPS))
        for ops, n in c.most_common(8):
            mark = '  <-- GROUND TRUTH' if ops == truth else ''
            print(f"  {n:>3}/{len(selections)}  {', '.join(ops)}{mark}")
        # Chance rate for drawing the true unordered pair from the basis.
        import itertools
        space = [tuple(sorted(p)) for p in
                 itertools.product(BASIS, repeat=len(selections[0]))]
        chance = space.count(truth) / len(space)
        got = c.get(truth, 0) / len(selections)
        print(f"\n  ground-truth op-set: {got:.0%} of seeds"
              f"   vs {chance:.0%} by chance")

    print("\n" + "=" * 92)
    print(" READING IT")
    print("=" * 92)
    if 'gai' not in rows or f'restart-{args.k}' not in rows:
        print("  (run both the 'gai' and 'restart' arms for the comparison)")
        return
    g = rows['gai'][0]
    r = rows[f'restart-{args.k}'][0]
    if r >= g:
        print(f"  Restarts match or beat the search ({r}/{args.seeds} vs"
              f" {g}/{args.seeds}) at equal budget. The mutation loop is then")
        print("  best understood as a restart mechanism, not as operator")
        print("  selection -- consistent with 0/126 mutations ever producing a")
        print("  new global best.")
    else:
        print(f"  The search recovers more often than budget-matched restarts"
              f" ({g}/{args.seeds} vs {r}/{args.seeds}), so it is contributing")
        print("  something beyond re-drawing weights.")
    print("  restart-*-true is the ceiling for restarts once operator choice is")
    print("  free; the gap between it and plain restart is what operator")
    print("  selection would be worth if it were solved perfectly.")


if __name__ == '__main__':
    main()
