"""
Does the search machinery transfer to quantum circuits unmodified?

training/structure_search.py is NOT changed for this. The only new code is a
gate basis (models/quantum_ops.py) and a model that exposes the same handful of
methods (models/quantum_model.py). If the search protocol is general, that
should be sufficient.

Task: learn a target unitary from its action on random states. The target is
built from a KNOWN gate sequence, so an exact solution is guaranteed to exist at
the search depth -- the direct analogue of the sin(pi*x^2) + x^2 recovery test.
We can therefore ask whether the search finds an answer known to be reachable,
not merely whether it beats a baseline.

Reported per run:
  test MSE           on held-out states
  process fidelity   |Tr(U_target^dag U_found)| / dim, 1.0 = equal up to phase
  gate count         non-identity slots
  two-qubit count    entangling gates, the expensive hardware resource

Recovery means fidelity > 0.99: the circuit implements the same operation, which
may be reached by a DIFFERENT gate sequence than the target -- gate identities
make several sequences equivalent, exactly as sin/cos were interchangeable in
the neural version. Fidelity is the honest criterion, not string matching.

Run:  python experiments/quantum_recovery.py --depth 4 --seeds 4
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.quantum_model import (QuantumCircuitModel, process_fidelity,
                                  unitary_learning_task)
from models.quantum_ops import SEARCH_GATES
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything)

# Target sequences of increasing difficulty. Each is expressible at its own
# length, so depth == len(target) always admits an exact solution.
TARGETS = {
    3: ['h', 'cnot', 'rz'],
    4: ['h', 'cnot', 'rz', 'cnot'],
    5: ['ry', 'cnot', 'rz', 'cnot', 'h'],
    6: ['h', 'cnot', 'rz', 'ry', 'cnot', 'h'],
}


def split(X, Y):
    n = len(X)
    a, b = int(0.6 * n), int(0.8 * n)
    return (X[:a], Y[:a]), (X[a:b], Y[a:b]), (X[b:], Y[b:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qubits', type=int, default=3)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--states', type=int, default=240)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--mode', nargs='+', default=['exhaustive', 'greedy'])
    ap.add_argument('--screen-epochs', type=int, default=40)
    args = ap.parse_args()

    target_ops = TARGETS[args.depth]
    X, Y, u_target, _ = unitary_learning_task(args.qubits, target_ops,
                                              args.states, seed=0)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
    data = (Xtr, ytr, Xva, yva, Xte, yte)
    space = len(SEARCH_GATES) ** args.depth

    print("=" * 92)
    print(f" QUANTUM CIRCUIT RECOVERY -- {args.qubits} qubits, depth {args.depth}")
    print(f" target sequence: {target_ops}")
    print(f" gate basis: {SEARCH_GATES}  ->  search space {len(SEARCH_GATES)}"
          f"**{args.depth} = {space:,}")
    print(f" train/val/test states: {len(Xtr)}/{len(Xva)}/{len(Xte)}")
    print(" structure_search.py is used UNMODIFIED")
    print("=" * 92)

    summary = {}
    for mode in args.mode:
        print(f"\n--- {mode} ---")
        print(f"{'seed':>4} | {'found circuit':<34} | {'test MSE':>10} | "
              f"{'fidelity':>8} | gates | 2q | rec")
        print("-" * 92)
        fids, mses, recs = [], [], []
        for seed in range(args.seeds):
            seed_everything(seed)
            cfg = SearchConfig(
                seed=seed, search_mode=mode,
                basis=tuple(SEARCH_GATES),          # gates, not activations
                use_composites=False,               # composition = depth here
                exhaustive_refine_composites=False,
                exhaustive_max_configs=space + 1,   # allow full enumeration
                exhaustive_screen_epochs=args.screen_epochs,
                exhaustive_verify_top=5,
                warmup_epochs=200, probe_epochs=120, consolidate_epochs=300,
                max_op_sweeps=3, topology_rounds=0, allow_growth=False,
                allow_pruning=False, final_restarts=4, compress=False,
                verbose=False)
            model = QuantumCircuitModel(
                input_dim=X.shape[1], hidden_dim=args.qubits, num_chains=1,
                chain_depth=args.depth, rng=random.Random(seed))
            search = StructureSearch(model, *data, config=cfg)
            tr = search.run()

            found = search.model
            fid = process_fidelity(u_target, found.unitary())
            rec = fid > 0.99
            fids.append(fid); mses.append(tr.test_loss); recs.append(rec)
            print(f"{seed:>4} | {','.join(tr.final_structure[0]):<34} | "
                  f"{tr.test_loss:>10.3e} | {fid:>8.4f} | "
                  f"{found.gate_count():>5} | {found.two_qubit_count():>2} | "
                  f"{'YES' if rec else 'no'}")
        print("-" * 92)
        print(f"recovery {sum(recs)}/{args.seeds}   median fidelity "
              f"{np.median(fids):.4f}   median test {np.median(mses):.3e}")
        summary[mode] = (sum(recs) / args.seeds, float(np.median(fids)),
                         float(np.median(mses)))

    print("\n" + "=" * 92)
    print(" VERDICT")
    print("=" * 92)
    print(f"  {'mode':<12} {'recovery':>9} {'median fidelity':>17} "
          f"{'median test MSE':>17}")
    for mode, (r, f, m) in summary.items():
        print(f"  {mode:<12} {r:>8.0%} {f:>17.4f} {m:>17.3e}")
    print("\n  What this does and does not show. It shows whether the search")
    print("  protocol transfers to a gate basis at all. It does NOT show novelty:")
    print("  differentiable and RL-based quantum architecture search already do")
    print("  gate selection, and gadget-mining already does composite reuse.")
    print("  Treat a good result here as 'the machinery is general', not as a")
    print("  publishable method.")


if __name__ == '__main__':
    main()
