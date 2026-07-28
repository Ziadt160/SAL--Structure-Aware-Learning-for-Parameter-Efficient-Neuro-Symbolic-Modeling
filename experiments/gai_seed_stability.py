"""
Seed stability, redone with GAIOptimizer instead of my rewrite.

The original question for this project was: does the discovered architecture
come out the same across random seeds? The earlier answer -- agreement 0.16
against chance 0.17 -- was measured with training/structure_search.py, which has
since been shown to be the weaker algorithm (tuned GAIOptimizer beats it 4.92x
on gauss_of_sin). So that answer described my reimplementation.

This redoes it with the three configurations the grid search actually found,
against two controls:

  fixed   the same model, NO structural search, matched epoch budget
  random  best-of-N random architectures at a matched number of trainings

Three separate notions of stability are reported, because they have different
answers and conflating them is how "the model discovered X" claims get made:

  performance   spread of test loss across seeds
  structural    pairwise agreement between discovered architectures, compared
                MODULO operator equivalence classes (sin/cos and tanh/sigmoid
                are exactly interchangeable given the surrounding affine maps,
                so raw string comparison overstates instability) and against
                the agreement expected from independent uniform draws
  marginal      per-slot operator frequency -- "does position (0,1) prefer a
                periodic operator?" can hold even when no two seeds agree on the
                whole architecture, and it is the reportable version

Run:  python experiments/gai_seed_stability.py --task gauss_of_sin --seeds 6
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.gai_final import GAI_CONFIGS
from experiments.sota_baselines import TASKS, split
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (seed_everything, slot_frequencies,
                                       structural_agreement)
from training.trainer import GAIOptimizer


def build(cfg, in_d, out_d, seed, importance_mode='taylor'):
    c = dict(cfg)
    c.pop('optimize_efficiency', None)
    return MatrixGGLEN(input_dim=in_d, output_dim=out_d,
                       hidden_dim=c.pop('hidden_dim'),
                       num_chains=c.pop('num_chains'),
                       chain_depth=c.pop('chain_depth'),
                       rng=random.Random(seed),
                       importance_mode=importance_mode), c


def evaluate(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        return float(nn.MSELoss()(model(torch.as_tensor(Xte)),
                                  torch.as_tensor(yte)))


def run_arm(data, cfg_name, seeds, epochs, mode='search',
            importance_mode='taylor'):
    """mode: 'search' = normal GAIOptimizer; 'fixed' = patience beyond the
    budget so no structural move ever fires.

    importance_mode resolves a confound I introduced: the original code ranked
    nodes by mean ||dL/dz|| (gradnorm) and I replaced the default with the
    first-order Taylor criterion |dL/dz * z|. The two rank differently -- on a
    trained model taylor picks node (0,1) as least important while gradnorm
    picks (0,0) -- so the grid search's winning 'importance' configs were using
    MY metric, not the project's original one. Running both settles it."""
    Xtr, ytr, Xva, yva, Xte, yte = data
    tests, structs, inits = [], [], []
    for s in range(seeds):
        seed_everything(s)
        model, c = build(GAI_CONFIGS[cfg_name], Xtr.shape[1], ytr.shape[1], s,
                         importance_mode)
        inits.append(model.get_structure())
        eff = GAI_CONFIGS[cfg_name].get('optimize_efficiency', False)
        if mode == 'fixed':
            c['patience'] = epochs * 10          # never triggers evolution
            eff = False
        opt = GAIOptimizer(model, loss_fn=nn.MSELoss(), **c)
        opt.fit(Xtr, ytr, Xva, yva, epochs=epochs, name=cfg_name,
                optimize_efficiency=eff)
        tests.append(evaluate(opt.model, Xte, yte))
        structs.append(opt.model.get_structure())
    return tests, structs, inits


def held_init(inits, finals):
    """How many nodes still carry the operator the initialiser gave them."""
    out = []
    for i, f in zip(inits, finals):
        n = sum(1 for ci, cf in zip(i, f) for a, b in zip(ci, cf) if a == b)
        out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='gauss_of_sin', choices=list(TASKS))
    ap.add_argument('--configs', nargs='+', default=['GAI-A', 'GAI-B'])
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--n', type=int, default=4500)
    ap.add_argument('--out', default='results/gai_seed_stability.json')
    args = ap.parse_args()

    fn, kind = TASKS[args.task]
    X, Y = fn(n=args.n)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
    data = (Xtr, ytr, Xva, yva, Xte, yte)

    print("=" * 92)
    print(f" GAIOptimizer SEED STABILITY -- {args.task} [{kind}], "
          f"{args.seeds} seeds")
    print("=" * 92)

    # (config, mode, importance_metric)
    arms = []
    for cfg_name in args.configs:
        uses_importance = (GAI_CONFIGS[cfg_name]['mutation_strategy']
                           == 'importance')
        arms.append((cfg_name, 'search', 'taylor'))
        if uses_importance:
            arms.append((cfg_name, 'search', 'gradnorm'))
        arms.append((cfg_name, 'fixed', 'taylor'))

    store = {}
    for cfg_name, mode, imode in arms:
        if True:
            label = (f"{cfg_name}/{mode}" if mode == 'fixed'
                     else f"{cfg_name}/{mode}/{imode}")
            tests, structs, inits = run_arm(data, cfg_name, args.seeds,
                                            args.epochs, mode, imode)
            store[label] = {'tests': tests, 'structs': structs,
                            'inits': inits}
            L = np.array(tests)
            q1, q3 = np.percentile(L, [25, 75])
            print(f"\n--- {label} ---")
            print(f"  test: median {np.median(L):.4e}  IQR {q3-q1:.4e}  "
                  f"min {L.min():.4e}  max {L.max():.4e}")
            if mode == 'search':
                n_nodes = sum(len(c) for c in structs[0])
                hi = held_init(inits, structs)
                print(f"  nodes still holding their init operator: "
                      f"{np.mean(hi):.1f}/{n_nodes}")
                for mod, tag in ((True, 'modulo equivalence'),
                                 (False, 'raw op names')):
                    r = structural_agreement(structs, modulo_equivalence=mod)
                    verdict = ('above chance'
                               if r['mean_agreement'] > r['null_agreement'] * 1.5
                               else 'at or near chance')
                    print(f"  agreement ({tag:<18}): {r['mean_agreement']:.2f}"
                          f" +/-{r['std_agreement']:.2f}  "
                          f"chance {r['null_agreement']:.2f}  -- {verdict}")
                freq = slot_frequencies(structs, modulo_equivalence=True)
                for slot in sorted(freq):
                    items = sorted(freq[slot].items(), key=lambda kv: -kv[1])
                    print(f"    slot {slot:<8} {items[0][0]:<22} "
                          f"{items[0][1]}/{args.seeds}"
                          + (f"   others {dict(items[1:])}" if len(items) > 1
                             else "   (unanimous)"))

    print("\n" + "=" * 92)
    print(" DOES THE SEARCH BEAT NOT SEARCHING?")
    print("=" * 92)
    for cfg_name in args.configs:
        f = float(np.median(store[f"{cfg_name}/fixed"]['tests']))
        for key in [k for k in store if k.startswith(f"{cfg_name}/search")]:
            s = float(np.median(store[key]['tests']))
            verdict = ('search better by %.2fx' % (f / s) if s < f
                       else 'FIXED better by %.2fx' % (s / f))
            print(f"  {key:<28} {s:.4e}  vs  fixed {f:.4e}  -> {verdict}")

    print("\n  taylor vs gradnorm (the metric I swapped in):")
    for cfg_name in args.configs:
        t = store.get(f"{cfg_name}/search/taylor")
        g = store.get(f"{cfg_name}/search/gradnorm")
        if t and g:
            tm, gm = float(np.median(t['tests'])), float(np.median(g['tests']))
            better = 'taylor' if tm < gm else 'gradnorm (the ORIGINAL metric)'
            print(f"    {cfg_name}: taylor {tm:.4e}  gradnorm {gm:.4e}  -> "
                  f"{better} better by {max(tm,gm)/min(tm,gm):.2f}x")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(store, fh, indent=2)
    print(f"\nraw -> {args.out}")


if __name__ == '__main__':
    main()
