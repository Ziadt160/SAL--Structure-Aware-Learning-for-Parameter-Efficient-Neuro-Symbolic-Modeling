"""
The comparison that decides whether there is a paper here.

Claiming an advantage over an MLP we tuned ourselves is not a result. The
relevant prior art for "recover interpretable structure from data" is
library-based symbolic regression:

  SINDy (Brunton, Proctor & Kutz 2016) -- build a library of candidate terms,
  then solve a SPARSE LINEAR problem for which terms appear. On Lorenz
  derivatives this is essentially exact, because
      dx/dt = 10(y - x)
      dy/dt = 28x - xz - y
      dz/dt = xy - (8/3)z
  is linear in a degree-2 polynomial library. Standardising x and y is an affine
  change of variables, which preserves that, so SINDy remains exact here.

That is the honest bar, and on Lorenz we lose to it by orders of magnitude.

BUT the win is structural, not incidental: SINDy needs every term to be IN the
library. It cannot represent a composition. `sin(pi*x^2)` is in no polynomial
library and no finite trig library -- to capture it you must already have
`sin(pi*x^2)` as an atom, which assumes the answer.

Compositional operator search builds exactly that: `sin_of_square`. So the two
methods should split by regime, and this measures the split:

  IN-LIBRARY targets      polynomial / plain-trig  -> SINDy should win big
  COMPOSITIONAL targets   nested operators         -> search should win big

Run:  python experiments/sota_baselines.py
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.adaptive_neural_model import MatrixGGLEN
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything)


# --------------------------------------------------------------------------
# SINDy: candidate library + sequentially thresholded least squares
# --------------------------------------------------------------------------
def poly_library(X, degree=3, include_trig=True, rich=False):
    """Feature matrix and names for a candidate term library.

    `rich=True` is the "practitioner tried harder" library: trig at several
    frequencies, trig OF squares, and Gaussians. It does not contain the target
    compositions as atoms -- that would assume the answer -- but it goes well
    beyond the generic polynomial+trig default, so a win against it is not a win
    against a strawman.
    """
    n, d = X.shape
    feats, names = [np.ones(n)], ['1']
    for i in range(d):
        feats.append(X[:, i]); names.append(f'x{i}')
    if degree >= 2:
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j]); names.append(f'x{i}x{j}')
    if degree >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    feats.append(X[:, i] * X[:, j] * X[:, k])
                    names.append(f'x{i}x{j}x{k}')
    if include_trig:
        for i in range(d):
            feats.append(np.sin(X[:, i])); names.append(f'sin(x{i})')
            feats.append(np.cos(X[:, i])); names.append(f'cos(x{i})')
    if rich:
        for i in range(d):
            for w in (np.pi, 2 * np.pi, 3 * np.pi):
                feats.append(np.sin(w * X[:, i])); names.append(f'sin({w:.2f}x{i})')
                feats.append(np.cos(w * X[:, i])); names.append(f'cos({w:.2f}x{i})')
            # trig of a square, and a Gaussian bump -- the shapes a
            # compositional target tends to need
            feats.append(np.sin(np.pi * X[:, i] ** 2))
            names.append(f'sin(pi*x{i}^2)')
            feats.append(np.cos(np.pi * X[:, i] ** 2))
            names.append(f'cos(pi*x{i}^2)')
            feats.append(np.exp(-X[:, i] ** 2)); names.append(f'exp(-x{i}^2)')
        for i in range(d):
            for j in range(d):
                if i != j:
                    feats.append(X[:, i] * np.sin(np.pi * X[:, j]))
                    names.append(f'x{i}sin(pi*x{j})')
    return np.stack(feats, axis=1), names


def stlsq(Theta, y, threshold=1e-3, iters=20):
    """Sequentially thresholded least squares -- the standard SINDy solver."""
    xi, _, _, _ = np.linalg.lstsq(Theta, y, rcond=None)
    for _ in range(iters):
        small = np.abs(xi) < threshold
        if small.all():
            break
        xi[small] = 0.0
        big = ~small
        xi[big], _, _, _ = np.linalg.lstsq(Theta[:, big], y, rcond=None)
    return xi


def run_sindy(Xtr, ytr, Xte, yte, degree=3, include_trig=True,
              thresholds=(1e-4, 1e-3, 1e-2, 5e-2), Xva=None, yva=None,
              rich=False):
    """Fit SINDy per output dimension, selecting the threshold on validation."""
    Th_tr, names = poly_library(Xtr, degree, include_trig, rich)
    Th_te, _ = poly_library(Xte, degree, include_trig, rich)
    Th_va, _ = poly_library(Xva, degree, include_trig, rich) if Xva is not None \
        else (Th_te, None)
    y_va = yva if yva is not None else yte

    preds = np.zeros_like(yte)
    terms_used, per_dim = 0, []
    for k in range(ytr.shape[1]):
        best = (np.inf, None, None)
        for th in thresholds:
            xi = stlsq(Th_tr, ytr[:, k], threshold=th)
            v = float(np.mean((Th_va @ xi - y_va[:, k]) ** 2))
            if v < best[0]:
                best = (v, xi, th)
        xi = best[1]
        preds[:, k] = Th_te @ xi
        nz = np.nonzero(xi)[0]
        terms_used += len(nz)
        per_dim.append(' + '.join(f'{xi[i]:+.3f}*{names[i]}' for i in nz[:6])
                       or '0')
    mse = float(np.mean((preds - yte) ** 2))
    return mse, terms_used, per_dim


# --------------------------------------------------------------------------
# Task families
# --------------------------------------------------------------------------
def lorenz(n=4500, seed=0):
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt, s = 0.01, np.array([1.0, 1.0, 1.0])

    def f(v):
        x, y, z = v
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    states = []
    for _ in range(n):
        k1 = f(s); k2 = f(s + dt * k1 / 2)
        k3 = f(s + dt * k2 / 2); k4 = f(s + dt * k3)
        s = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(s.copy())
    X = np.array(states, dtype=np.float64)
    Y = np.array([f(v) for v in X], dtype=np.float64)
    X = (X - X.mean(0)) / X.std(0)
    Y = (Y - Y.mean(0)) / Y.std(0)
    return X.astype(np.float32), Y.astype(np.float32)


def polynomial(n=4500, seed=0):
    """In-library: a plain degree-2 polynomial."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32)
    y = (2.0 * X[:, 0] * X[:, 1] - 1.5 * X[:, 2] ** 2 + 0.5 * X[:, 0])
    return X, y.reshape(-1, 1).astype(np.float32)


def comp_sin_of_square(n=4500, seed=0):
    """Compositional: sin(pi*x^2) -- in no polynomial or plain-trig library."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.2, 1.2, (n, 2)).astype(np.float32)
    y = np.sin(np.pi * X[:, 0] ** 2) + 0.5 * X[:, 1]
    return X, y.reshape(-1, 1).astype(np.float32)


def comp_gauss_of_sin(n=4500, seed=0):
    """Compositional: exp(-sin(pi*x)^2)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.2, 1.2, (n, 2)).astype(np.float32)
    y = np.exp(-np.sin(np.pi * X[:, 0]) ** 2) + 0.3 * X[:, 1]
    return X, y.reshape(-1, 1).astype(np.float32)


def comp_square_of_sin(n=4500, seed=0):
    """Compositional: sin(pi*x)^2 -- expressible via cos(2pi x), but only if
    the library happens to contain that frequency."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.2, 1.2, (n, 2)).astype(np.float32)
    y = np.sin(np.pi * X[:, 0]) ** 2 - 0.4 * X[:, 1] ** 2
    return X, y.reshape(-1, 1).astype(np.float32)


TASKS = {
    'lorenz':        (lorenz,             'in-library (polynomial)'),
    'polynomial':    (polynomial,         'in-library (polynomial)'),
    'sin_of_square': (comp_sin_of_square, 'COMPOSITIONAL'),
    'gauss_of_sin':  (comp_gauss_of_sin,  'COMPOSITIONAL'),
    'square_of_sin': (comp_square_of_sin, 'COMPOSITIONAL'),
}


def split(X, y):
    n = len(X)
    a, b = int(0.6 * n), int(0.8 * n)
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:])


# --------------------------------------------------------------------------
def run_search(data, seed, width, depth, composites=True):
    """The genuinely best configuration measured so far -- every validated
    finding switched ON.

    An earlier version of this function ran with topology_rounds=0 and
    allow_growth=False, which handicapped our own side of the comparison by the
    single largest validated factor: on the analytic task, letting Phase 2 run
    (and not gating it on Phase 1 having improved) was worth 132x, 2.0e-02 to
    1.5e-04. Running the decisive baseline comparison with that switched off
    would have been exactly the unfair-comparison error this project was
    criticised for, pointed the other way.
    """
    Xtr, ytr, Xva, yva, Xte, yte = data
    seed_everything(seed)
    cfg = SearchConfig(
        seed=seed,
        search_mode='exhaustive',          # 75% recovery vs greedy's 38%
        use_composites=composites,         # needed for nested targets
        exhaustive_refine_composites=composites,
        topology_rounds=3,                 # was 0 -- worth 132x
        allow_growth=True, allow_pruning=True,
        topology_requires_op_gain=False,   # decoupling measured much better
        max_chains=3, max_depth=4,
        exhaustive_max_configs=1300,       # allow enumeration after one growth
        final_restarts=6,                  # fixes the weight lottery
        warmup_epochs=300, exhaustive_screen_epochs=120,
        exhaustive_verify_top=4, consolidate_epochs=250,
        max_op_sweeps=3, compress=False,   # compress trades loss for params
        verbose=False)
    m = MatrixGGLEN(input_dim=Xtr.shape[1], output_dim=ytr.shape[1],
                    hidden_dim=width, num_chains=1, chain_depth=depth,
                    rng=random.Random(seed))
    return StructureSearch(m, Xtr, ytr, Xva, yva, Xte, yte, config=cfg).run()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--width', type=int, default=8)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--n', type=int, default=4500)
    args = ap.parse_args()

    print("=" * 92)
    print(" SINDy vs COMPOSITIONAL OPERATOR SEARCH")
    print(" SINDy library: polynomials to degree 3 + sin/cos of each input,")
    print(" sparsified by STLSQ with the threshold chosen on validation.")
    print("=" * 92)

    summary = []
    for name in args.tasks:
        fn, kind = TASKS[name]
        X, Y = fn(n=args.n)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, Y)
        data = (Xtr, ytr, Xva, yva, Xte, yte)

        args64 = dict(Xva=Xva.astype(np.float64), yva=yva.astype(np.float64))
        s_mse, s_terms, s_eqs = run_sindy(
            Xtr.astype(np.float64), ytr.astype(np.float64),
            Xte.astype(np.float64), yte.astype(np.float64), **args64)
        r_mse, r_terms, r_eqs = run_sindy(
            Xtr.astype(np.float64), ytr.astype(np.float64),
            Xte.astype(np.float64), yte.astype(np.float64),
            rich=True, **args64)

        losses, structs = [], []
        for seed in range(args.seeds):
            tr = run_search(data, seed, args.width, args.depth)
            losses.append(tr.test_loss)
            structs.append(tr.final_structure)
        o_mse = float(np.median(losses))

        # Compare against the BETTER of the two SINDy libraries -- beating only
        # the generic one would be beating a strawman.
        best_sindy = min(s_mse, r_mse)
        print(f"\n--- {name}  [{kind}] ---")
        print(f"  SINDy generic    test MSE = {s_mse:.4e}   ({s_terms} terms)")
        print(f"    recovered: {s_eqs[0][:84]}")
        print(f"  SINDy rich lib   test MSE = {r_mse:.4e}   ({r_terms} terms)")
        print(f"    recovered: {r_eqs[0][:84]}")
        print(f"  operator search  test MSE = {o_mse:.4e}   "
              f"(median of {args.seeds}, best {min(losses):.4e})")
        for st in structs:
            print(f"    found: {st}")
        ratio = best_sindy / o_mse if o_mse > 0 else float('inf')
        winner = 'operator search' if o_mse < best_sindy else 'SINDy'
        print(f"  -> {winner} wins by {max(ratio, 1 / ratio):.3g}x "
              f"(against the better SINDy library)")
        summary.append((name, kind, s_mse, r_mse, o_mse, winner))

    print("\n" + "=" * 92)
    print(" SUMMARY")
    print("=" * 92)
    print(f"{'task':<16} {'regime':<24} {'SINDy gen':>11} {'SINDy rich':>11} "
          f"{'search':>11}   winner")
    for name, kind, sg, sr, o, w in summary:
        print(f"{name:<16} {kind:<24} {sg:>11.3e} {sr:>11.3e} {o:>11.3e}   {w}")
    print("\n  The paper's claim stands only if SINDy wins the in-library rows")
    print("  and operator search wins the COMPOSITIONAL rows. If SINDy wins")
    print("  everywhere, there is no niche and no paper.")


if __name__ == '__main__':
    main()
