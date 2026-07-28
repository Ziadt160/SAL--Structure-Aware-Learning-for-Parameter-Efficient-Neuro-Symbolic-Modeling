"""
Does the discovered architecture agree across random seeds?

Three arms, N seeds each, on Lorenz derivative prediction:

  search   the full operator search
  fixed    same model, no search, matched epoch budget
  random   best-of-N random architectures at a matched evaluation budget

and three separate notions of stability, because they have different answers:

  performance   spread of test loss across seeds
  structural    pairwise agreement between discovered architectures,
                compared modulo operator equivalence classes and against the
                agreement expected from independent uniform draws
  marginal      per-slot operator frequency -- "does position (0,1) prefer a
                periodic operator?" can hold even when no two seeds agree on
                the whole architecture, and it is the reportable version of
                the claim

`held_init` counts nodes still carrying the operator the initialiser gave them.
If that number is large, the reported architecture is largely a report of
`random.choice` and cannot agree across seeds regardless of the search rule.

Run:  python experiments/seed_stability.py --seeds 6
"""

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       fixed_architecture_control,
                                       random_search_control, seed_everything,
                                       slot_frequencies, structural_agreement)


def lorenz_data(n=1500):
    """Lorenz states -> derivatives. RK4, no scipy dependency."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt = 0.01
    s = np.array([1.0, 1.0, 1.0])

    def f(v):
        x, y, z = v
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    states = []
    for _ in range(n):
        k1 = f(s); k2 = f(s + dt * k1 / 2)
        k3 = f(s + dt * k2 / 2); k4 = f(s + dt * k3)
        s = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(s.copy())
    X = np.array(states, dtype=np.float32)
    Y = np.array([f(v) for v in X], dtype=np.float32)
    # Standardise both sides: raw states reach |x|~20, z~50, which saturates
    # every bounded activation and blows square/gaussian into the soft-clip.
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)
    Y = ((Y - Y.mean(0)) / Y.std(0)).astype(np.float32)
    return X, Y


def split(X, y):
    """Contiguous split. A random split would leak: consecutive trajectory
    points are nearly identical, so neighbours of every test point would be
    in train."""
    n = len(X)
    a, b = int(0.6 * n), int(0.8 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


def make_config(seed, args):
    return SearchConfig(
        seed=seed,
        warmup_epochs=300,
        probe_epochs=40,
        probe_restarts=args.restarts,
        consolidate_epochs=250,
        max_op_sweeps=args.sweeps,
        use_composites=args.composites,
        topology_rounds=0,
        allow_growth=False,
        allow_pruning=False,
        compress=False,
        verbose=False,
    )


def build(seed, args):
    return MatrixGGLEN(input_dim=3, output_dim=3, hidden_dim=args.hidden,
                       num_chains=args.chains, chain_depth=args.depth,
                       rng=random.Random(seed), readout=args.readout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--hidden', type=int, default=8)
    ap.add_argument('--chains', type=int, default=2)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--sweeps', type=int, default=2)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--readout', default='sum', choices=['sum', 'concat'])
    ap.add_argument('--composites', action='store_true')
    ap.add_argument('--out', default='results/seed_stability.json')
    args = ap.parse_args()

    X, Y = lorenz_data()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
    data = (Xtr, ytr, Xva, yva, Xte, yte)

    print("=" * 80)
    print(" SEED STABILITY -- Lorenz derivatives")
    print(f" {args.seeds} seeds | H={args.hidden} chains={args.chains} "
          f"depth={args.depth} readout={args.readout} "
          f"composites={args.composites}")
    print("=" * 80)

    arms = {'search': [], 'fixed': [], 'random': []}

    for seed in range(args.seeds):
        cfg = make_config(seed, args)

        seed_everything(seed)
        tr = StructureSearch(build(seed, args), *data, config=cfg).run()
        arms['search'].append(tr)
        print(f"[seed {seed}] search  test={tr.test_loss:.4e} "
              f"epochs={tr.train_epochs} held_init={tr.nodes_holding_init_op()}"
              f"/{sum(len(o) for o in tr.final_structure)} "
              f"{tr.final_structure}", flush=True)

        seed_everything(seed)
        fx = fixed_architecture_control(build(seed, args), *data, config=cfg)
        arms['fixed'].append(fx)
        print(f"[seed {seed}] fixed   test={fx.test_loss:.4e} "
              f"epochs={fx.train_epochs} {fx.final_structure}", flush=True)

        # Compute-matched: give random search the same total epochs.
        n_cand = max(2, round(tr.train_epochs / cfg.consolidate_epochs))
        seed_everything(seed)
        rd = random_search_control(lambda r: build(r.randint(0, 10 ** 6), args),
                                   *data, config=cfg, n_candidates=n_cand)
        arms['random'].append(rd)
        print(f"[seed {seed}] random  test={rd.test_loss:.4e} "
              f"epochs={rd.train_epochs} (best of {n_cand}) "
              f"{rd.final_structure}", flush=True)

    # ---- performance stability
    print("\n" + "=" * 80)
    print(" 1. PERFORMANCE STABILITY (test loss across seeds)")
    print("=" * 80)
    print(f"{'arm':<10} {'median':>12} {'IQR':>12} {'min':>12} {'max':>12} "
          f"{'epochs':>9}")
    for name, traces in arms.items():
        L = np.array([t.test_loss for t in traces], dtype=float)
        q1, q3 = np.percentile(L, [25, 75])
        print(f"{name:<10} {np.median(L):>12.4e} {q3 - q1:>12.4e} "
              f"{L.min():>12.4e} {L.max():>12.4e} "
              f"{int(np.mean([t.train_epochs for t in traces])):>9}")

    s_med = float(np.median([t.test_loss for t in arms['search']]))
    f_med = float(np.median([t.test_loss for t in arms['fixed']]))
    r_med = float(np.median([t.test_loss for t in arms['random']]))
    print(f"\n  search vs fixed : {f_med / s_med:6.2f}x "
          f"({'search better' if s_med < f_med else 'FIXED BETTER'})")
    print(f"  search vs random: {r_med / s_med:6.2f}x "
          f"({'search better' if s_med < r_med else 'RANDOM BETTER'})")

    # ---- structural stability
    print("\n" + "=" * 80)
    print(" 2. STRUCTURAL STABILITY (searched arm)")
    print("=" * 80)
    structs = [t.final_structure for t in arms['search']]
    for label, mod in (('modulo equivalence classes', True), ('raw op names', False)):
        r = structural_agreement(structs, modulo_equivalence=mod)
        verdict = ('above chance' if r['mean_agreement'] > r['null_agreement'] * 1.5
                   else 'AT OR NEAR CHANCE')
        print(f"  {label:<28} agreement={r['mean_agreement']:.2f} "
              f"+/-{r['std_agreement']:.2f}  chance={r['null_agreement']:.2f}  "
              f"-- {verdict}")

    held = [t.nodes_holding_init_op() for t in arms['search']]
    total = sum(len(o) for o in structs[0])
    print(f"\n  nodes still holding their init operator: "
          f"{np.mean(held):.1f}/{total} on average")
    print(f"  search coverage (nodes probed): "
          f"{np.mean([t.coverage() for t in arms['search']]):.0%}")

    # ---- marginal stability
    print("\n" + "=" * 80)
    print(" 3. MARGINAL STABILITY (per-slot operator class, searched arm)")
    print("=" * 80)
    freq = slot_frequencies(structs, modulo_equivalence=True)
    for slot in sorted(freq):
        items = sorted(freq[slot].items(), key=lambda kv: -kv[1])
        top, cnt = items[0]
        print(f"  slot {slot:<8} {top:<24} {cnt}/{args.seeds}"
              + (f"   others: {dict(items[1:])}" if len(items) > 1 else "  (unanimous)"))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({name: [{'seed': t.seed, 'test': t.test_loss,
                           'val': t.val_loss, 'params': t.params,
                           'epochs': t.train_epochs,
                           'structure': t.final_structure,
                           'init_structure': t.init_structure}
                          for t in traces]
                   for name, traces in arms.items()}, fh, indent=2)
    print(f"\nRaw results -> {args.out}")


if __name__ == '__main__':
    main()
