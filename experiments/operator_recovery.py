"""
Ground-truth operator recovery.

The claim under test is the core of the idea: "the algorithm will find the
right activation functions through the loss metric." That is only checkable
against a target whose right answer is known, so this builds one.

Target:  y = sin(pi * x1) + x2**2,  x ~ U(-1, 1)^2

Model:   two chains, depth 1, hidden_dim 1.

hidden_dim=1 is deliberate. With a wider hidden layer many operator
assignments fit equally well -- eight units per chain can approximate the
target whatever operator they use -- so recovery becomes unidentifiable and
the test proves nothing. At width 1 each chain is a single scalar operator,
MatrixGGLEN sums the chains and applies a linear read-out, so

    final(sin(pi*x1) + x2**2) = a*(sin(pi*x1) + x2**2) + c

reproduces the target exactly when one chain picks `sin` and the other picks
`square`. The right answer is in the hypothesis space and it is unique up to
swapping the two chains -- so we compare the SET of recovered operator
classes, not their positions.

Run:  python experiments/operator_recovery.py --seeds 8
"""

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.activations import canonical_class
from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything)

TARGET_CLASSES = {'periodic', 'quadratic'}


def make_data(n: int = 900, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = (np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1).astype(np.float32)
    tr, va = int(0.6 * n), int(0.8 * n)
    return (X[:tr], y[:tr]), (X[tr:va], y[tr:va]), (X[va:], y[va:])


def recovered_classes(structure):
    """Operator classes present anywhere in the architecture."""
    return {canonical_class(op) for ops in structure for op in ops}


# Reaching this loss means the closed form was found: the target is exactly
# representable, so anything near float64 noise IS the target.
EXACT = 1e-8


def recovered_exactly(test_loss) -> bool:
    """Loss-based recovery -- the criterion that actually means something.

    Class matching is too strict once composites are in play, because the
    operator set contains equivalences beyond the declared classes. `relu`
    is the identity on non-negative inputs, so `relu_of_square` computes
    exactly `square`; likewise `relu_of_gaussian` computes `gaussian`. Runs
    that found `relu_of_square` hit test losses of 1.06e-15 and 3.10e-14 --
    exact solutions that the class check scored as misses because
    `rectifier_of_quadratic` is not the string `quadratic`.
    """
    return bool(test_loss < EXACT)


def run_seed(seed: int, use_composites: bool, readout: str = 'sum',
             restarts: int = 3, verbose: bool = False,
             search_mode: str = 'greedy'):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = make_data(seed=seed)
    seed_everything(seed)

    cfg = SearchConfig(
        seed=seed,
        search_mode=search_mode,
        warmup_epochs=300,
        probe_epochs=120,
        probe_restarts=restarts,
        consolidate_epochs=300,
        max_op_sweeps=3,
        use_composites=use_composites,
        exhaustive_refine_composites=use_composites,
        exhaustive_screen_epochs=120,
        topology_rounds=0,      # isolate Phase 1
        allow_growth=False,
        allow_pruning=False,
        compress=False,
        verbose=verbose,
    )
    model = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=1,
                        num_chains=2, chain_depth=1, rng=random.Random(seed),
                        readout=readout)
    search = StructureSearch(model, Xtr, ytr, Xva, yva, Xte, yte, cfg)
    return search.run()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--restarts', type=int, default=3)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print("=" * 78)
    print(" OPERATOR RECOVERY  --  target: y = sin(pi*x1) + x2^2")
    print(f" required operator classes: {sorted(TARGET_CLASSES)}")
    print(f" probe restarts: {args.restarts}")
    print("=" * 78)

    arms = [
        ('greedy     / basis', False, 'sum', 'greedy'),
        ('greedy     / comp',  True,  'sum', 'greedy'),
        ('exhaustive / basis', False, 'sum', 'exhaustive'),
        ('exhaustive / comp',  True,  'sum', 'exhaustive'),
    ]

    overall = {}
    for label, use_comp, readout, mode in arms:
        print(f"\n--- {mode} search, "
              f"{'basis+composites' if use_comp else 'basis only'}, "
              f"readout={readout} ---")
        print(f"{'seed':>4} | {'init':<20} | {'found':<40} | "
              f"{'test loss':>10} | exact | class")
        print("-" * 96)

        exact_hits, class_hits, losses = 0, 0, []
        for seed in range(args.seeds):
            tr = run_seed(seed, use_comp, readout, args.restarts,
                          args.verbose, mode)
            by_class = TARGET_CLASSES.issubset(
                recovered_classes(tr.final_structure))
            by_loss = recovered_exactly(tr.test_loss)
            exact_hits += by_loss
            class_hits += by_class
            init = ','.join(o for ops in tr.init_structure for o in ops)
            fin = ','.join(o for ops in tr.final_structure for o in ops)
            print(f"{seed:>4} | {init:<20} | {fin:<40} | "
                  f"{tr.test_loss:>10.3e} | {'YES  ' if by_loss else 'no   '} "
                  f"| {'YES' if by_class else 'no'}")
            losses.append(tr.test_loss)

        rate = exact_hits / args.seeds
        print("-" * 96)
        print(f"exact recovery (test < {EXACT:g}): {exact_hits}/{args.seeds} "
              f"= {rate:.0%}   |   class match: {class_hits}/{args.seeds}")
        print(f"median test {np.median(losses):.3e}   best {min(losses):.3e}")
        overall[label] = (rate, float(np.median(losses)), float(min(losses)))

    print("\n" + "=" * 78)
    print(" VERDICT")
    print("=" * 78)
    print(f"  {'arm':<20} {'recovery':>9} {'median test':>13} {'best test':>13}")
    for label, (rate, med, best) in overall.items():
        print(f"  {label:<20} {rate:>8.0%} {med:>13.3e} {best:>13.3e}")
    print("\n  Recovery = did Phase 1 select the operators the target actually")
    print("  needs, rather than keeping whatever the initialiser picked?")
    print("  greedy stalls when two nodes must change together; exhaustive")
    print("  evaluates every assignment and cannot.")


if __name__ == '__main__':
    main()
