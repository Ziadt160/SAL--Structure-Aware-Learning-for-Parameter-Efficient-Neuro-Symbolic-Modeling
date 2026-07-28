# Structure-Aware Learning

**Is it worth searching over per-layer activation functions? A controlled study
that answers "sometimes" — and says precisely when.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![Tests](https://img.shields.io/badge/invariants-62%20passing-brightgreen.svg)](tests/test_invariants.py)

---

> ### In one screen
>
> Instead of fixing `relu` everywhere, let **every layer choose its own operator**
> from `{identity, tanh, relu, sin, gaussian, square}` and their compositions.
> The motivation is real: on a smooth periodic target, the same MLP reaches
> `4.93e-05` with `tanh` and `3.48e-04` with `relu` — a **7x** swing from the
> activation alone.
>
> Two search algorithms were built and measured against matched-budget controls.
>
> | | verdict |
> |---|---|
> | **Exhaustive per-layer search** | **Works, narrowly.** 2.02x over a parameter-matched MLP on Lorenz derivatives, with complete seed separation, recovering the mathematically correct structure. Loses ~3x where no operator structure exists. |
> | **Simulated-annealing mutation loop** | **Does not work.** Operator selection is statistically indistinguishable from random assignment — five measurements across three independent experiments, every one a null. |
> | **The ceiling** | **100%.** Given the correct operators, the same budget recovers the exact symbolic structure at machine precision on 28/28 seeds. Selection is the entire bottleneck. |
>
> Three bugs were found **in the measuring apparatus itself**, each of which had
> inflated the evidence that the search was working — including a mutation
> counter that was detecting Adam loss spikes (16 false events per 1 real one).
> Two changes the author had previously presented as fixes were withdrawn after
> measurement. See [CHANGES.md](CHANGES.md).

## Quickstart

```bash
pip install -r requirements.txt
python tests/test_invariants.py
```

62 invariants covering operator semantics, symbolic-export fidelity
(torch vs SymPy agree to 1e-9), function-preserving topology growth, and the
optimiser's accept/revert contract.

Reproduce the headline control — search vs restarts vs an oracle that is handed
the correct operators, all on an identical 6000-epoch budget:

```bash
python experiments/restart_baseline.py --seeds 28 --epochs 6000 --k 8
```

## What this is

A neural model whose **per-layer activation operator is a searchable variable**,
optimised alongside the weights. Each layer selects an operator from a basis and
their compositions, chosen by validation loss.

The premise is sound — see the 7x swing above. Whether *this* code exploits it
is a separate question, and the answer is mixed. Read the summary before the
details.

## Summary

There are two distinct algorithms in this repository and they do not perform
alike.

**`training/structure_search.py` — exhaustive per-layer operator search. Works,
with a scoped claim.** It beats a tuned, parameter-matched MLP by **2.02x** on
Lorenz derivatives with complete seed separation, and recovers
`identity -> quadratic -> identity`, which is the mathematically correct answer
for a polynomial system. It loses by ~3x where the target has no operator
structure to find. *Caveat: 3 seeds. See "Known costs".*

**`training/trainer.py` — the `GAIOptimizer` mutation loop (simulated annealing,
importance-guided). Does not work.** Its operator selection is statistically
indistinguishable from picking operators at random. Five measurements across
three independent experiments, every one a null:

| test | search | control | p |
|---|---|---|---|
| exact recovery, 28 seeds | 3/28 | 0/28 random | 0.24 |
| correct operator set chosen | 10.7% | 5.6% chance | 0.20 |
| search-then-restart, 28 seeds | 3/28 | 3/28 search alone | — |
| recovery at 2 nodes, 16 seeds | 2/16 | 1/16 random | 1.0 |
| **correct op-set at 4 nodes (1,296 assignments)** | **0/16** | oracle 16/16 | — |

**And the ceiling is 100%.** Handed the correct operators, the same budget
recovers the exact structure at machine precision on **28 of 28 seeds** — and
**16 of 16** even in the 1,296-assignment space. Operator selection is the
entire bottleneck; everything else already works.

Two further scoping results:

- **Not mutating beats mutating** on the median in every version of the paired
  control (best case 0.86x).
- **Nesting, not depth, destroys exact recovery.** Identity-padded to depth 2:
  4/4 at `3.1e-15`. One nested nonlinearity (`gaussian(sin(.))`): 0/4, still 0/4
  at 4x budget — *with the correct operators supplied*. Since composition is the
  method's premise, this is the tightest limit in the project.

Full derivations, controls and the bugs found along the way are in
[CHANGES.md](CHANGES.md).

## What the measurements show

Median test loss over 3 seeds on a held-out split read exactly once, at
identical parameter counts, with **every arm given the same restart budget**
(4 activations x 5 restarts for the baseline, 5 final-training restarts for the
search). Reproduce with `python experiments/tuned_rerun.py --seeds 3 --restarts 5`.

| task | plain MLP, best global activation | search, greedy | search, exhaustive |
|---|---|---|---|
| Lorenz derivatives [203] | 1.19e-03 | 2.68e-03 (0.44x) | **5.90e-04 (2.02x better)** |
| random tanh-MLP teacher [185] | **2.71e-03** | 8.47e-03 (0.32x) | 7.24e-03 (0.37x) |

On Lorenz, exhaustive search's **worst** seed (8.44e-04) beats the baseline's
**best** (1.12e-03) -- complete separation across seeds -- and its best seed
reaches 1.33e-06. It finds `identity -> quadratic -> identity` at 2 of 3 seeds
in every position, which is the mathematically right answer for a polynomial
system: `square` generates the `xy` and `xz` products via `(a+b)^2 = a^2+2ab+b^2`,
which no single global activation can express.

On the teacher task, where the ground truth *is* an ordinary tanh MLP and there
is no operator structure to find, the search is **2.7-3.1x worse**. That gap
widened when restarts were added, because the baseline benefited more.

So the scope of the claim is:

> Exhaustive per-layer operator search, with final-training restarts, beats a
> tuned MLP by ~2x on systems whose structure matches the operator basis, and
> loses by ~3x where no such structure exists.

### Three findings that decide whether it works

- **Enumerate, do not hill-climb.** Greedy coordinate descent loses on *both*
  tasks. On ground-truth recovery it reaches the right answer 38% of the time
  against exhaustive's 75%, with a median test loss seven orders of magnitude
  worse (2.5e-04 against 3.4e-11). Greedy measures a node's value given the
  others, so it cannot reach an optimum that requires two nodes to move
  together. Enumeration is `len(basis) ** n_nodes`, so it applies to
  MLP-depth models and falls back to greedy past `exhaustive_max_configs`.
- **Restarts belong after selection, not inside it.** The architecture search
  and the weight optimisation fail independently: exhaustive finds the right
  Lorenz structure 2/3 of the time but the weights exploit it only 1/3, giving
  a bimodal result. `final_restarts` fixes that (Lorenz median improved 2.9x)
  while leaving the baseline almost unchanged (1.2x) -- it is repairing
  optimisation of `square`-type operators, not adding generic compute. Inside
  the probe, restarts actively *hurt*, because min-over-restarts is a biased
  ranking statistic that flatters high-variance operators.
- **The chain ensemble contributes nothing.** At matched parameters a plain MLP
  beats the chain with fixed random operators on every task. `num_chains=1`
  makes this literally an MLP with searchable activations, and that is the
  recommended configuration.

### What `GAIOptimizer`'s mutation loop actually contributes

Measured separately from the phase search above, on `GAIOptimizer` itself with
the best configuration a grid search found (`GAI-A`).

- **Not mutating usually does better.** Against a frozen-structure control at the
  same seed and initialisation, median gain is **0.66x** and mutation helped on
  1 of 6 seeds (`experiments/mutation_gain.py`, GAI-A, 4000 epochs). On 4 of 6
  seeds the mutating run's best was already reached *before its first mutation
  fired*. No mutation was followed by a new global best within 60 epochs
  (0/126) — but that is a statement about the attribution window, not a claim
  that mutation never helps: seed 4 gained 4.11x and finished 12x below its own
  pre-mutation best while still scoring 0 attributable. Widen the window to 800
  epochs and the same class of run scores 26%, by which point ordinary training
  supplies a new best anyway. Attribution does not settle this; the
  budget-matched control below does.
- **Mutation counts published before this were mostly artifacts.** They were
  recovered by scanning the loss curve for a >3x jump; against a real event log
  that reads 16 "mutations" where 1 occurred, because it was detecting Adam loss
  spikes. `GAIOptimizer.mutation_log` now records events directly.
- **Exact recovery is real, but rarer than 8 seeds suggest, and
  reset-dependent.** On `y = sin(pi*x1) + x2^2` at `hidden_dim=1`, the original
  SA recovers the exact structure `sin, square` at machine precision in
  **3/28 (11%)** of seeds — the same measurement reads 3/8 = 38% if you stop at
  8 seeds. Recovery collapses to 0 if the mutated node's fresh weights are drawn
  Xavier-wide instead of `U(-0.05, 0.05)` (6/16 vs 0/16, Fisher exact
  **p = 0.018**, a paired comparison so the small sample is not a problem here):
  a near-zero start lets `sin(pi*w*x)` grow into the correct frequency, a wide
  one starts it in a basin gradient descent cannot leave.
- **The tuned optimum barely searches.** `GAI-A` has `use_annealing=False` (so
  it is hill-climbing, not annealing) and fires 0-1 mutations per 1500 epochs.
- **Exact recovery and typical-case accuracy are separate objectives that respond
  differently to the same change** — worth knowing before tuning anything here.
  Splitting the budget into 8 restarts improves the median 4.7x and *cuts* exact
  recoveries (3/8 -> 1/8). The modified SA improves the median 1.81x
  (8.39e-02 -> 4.63e-02, 28 paired seeds) and leaves recovery statistically
  unchanged (11% vs 4%, p = 0.61). The only intervention that significantly moved
  recovery was reverting the reset draw. A change that improves the median is
  not thereby an improvement to the symbolic-recovery claim; check both.

Those facts suggested that a reset-mutation might just be *re-drawing* one node
rather than selecting a better operator — restarts wearing a search's clothes.
`experiments/restart_baseline.py` tests that against a budget-matched control.
Every arm gets exactly 6000 gradient steps, the same learning rate, optimiser
and reset draw, and selects on validation (8 seeds):

At 28 seeds, every arm on the same 6000-epoch budget:

| arm | recovery | median test |
|---|---|---|
| **oracle operators `sin, square` + 8 restarts** | **28/28 (100%)** | **7.66e-15** |
| the search (1 trajectory + mutations) | 3/28 (11%) | 8.39e-02 |
| random operators + 8 restarts | 0/28 (0%) | 8.12e-02 |

**Given the right operators, recovery is a solved problem — 100% of seeds, at
the same cost.** The search captures 11% of that (**p = 1.2e-12** against the
ceiling), and its margin over random operator assignment is **not statistically
established even at n=28** (3/28 vs 0/28, **p = 0.24**). Nothing measured here
demonstrates that the mutation loop selects operators better than chance.

Both ingredients are needed and neither alone suffices (8-seed cross):

| operators | 1 run x 6000 ep | 8 restarts x 750 ep |
|---|---|---|
| random | 0/8 | 0/8 |
| oracle | 1/8 | **8/8** |

Restarts given correct operators are worth 1/8 -> 8/8 (**p = 0.0014**); correct
operators given restarts, 0/8 -> 8/8 (**p = 0.00016**); correct operators
*without* restarts, nothing (**p = 1.0**). The forced-operator arm also kills the
obvious confound — 750 epochs per restart is not too short, since with the right
operators it hits machine precision every time. The random arm's 0/28 is an
operator-selection failure, not a convergence failure.

**Splitting the budget does not fix it.** Running the search itself as 8 x 750-ep
restarts gives 1/8 exact recoveries — *worse* than 3/8 — while improving the
median 4.7x (1.80e-02 against 8.39e-02). Restarts and operator search compete
for the same budget: 750 epochs is plenty to fit weights once operators are
known, and far too few to discover them.

**And the search's operator selection is indistinguishable from chance.** Taking
the operators the search settles on and restarting the weights on top of them —
which turns every correctly-selected seed into a recovery, since correct
operators recover 28/28 — leaves the result completely unchanged at 3/28. The
search lands on the ground-truth set in 10.7% of seeds against a 5.6% chance
rate (**p = 0.20**), and its two most frequent picks (`gaussian, sin` and
`relu, tanh`, 4/28 each) are each *more* common than the truth `sin, square`
(3/28). The 3 recoveries are exactly the 3 seeds where selection was right.
So "append restarts to the search" — which looks like a large one-line win — is
worth nothing, because the premise that the search supplies good operators
does not hold.

**A caution about sample size, learned here the hard way.** This same
configuration measured 3/8 = 38% on seeds 0-7 and 3/28 = 11% on seeds 0-27 —
seeds 8-27 contributed no recoveries at all. Absolute recovery rates quoted from
8 seeds in this repo are optimistic. Paired *comparisons* at 8 seeds are not
affected, because both arms share the seeds.

### Known costs

- **Enumeration is not free.** It maximises exposure to validation-set
  selection overfitting -- best-of-216 against best-of-6-per-node -- which is
  why it is marginally *worse* than greedy on the teacher task. It wins where
  real signal exists and loses harder where it does not.
- **`topology_requires_op_gain=True` is expensive.** Gating Phase 2 on Phase 1
  having improved matches the algorithm as originally specified, but on the
  analytic task a run that skipped Phase 2 finished at test 2.0e-02 where
  letting topology run reached 1.5e-04 -- 132x better, and it recovered the
  exact ground-truth structure `[[sin, identity], [square]]`. Set it False
  unless the coupling is wanted.
- **Exact symbolic recovery is width-1 only.** The 1e-14 results occur at
  `hidden_dim=1` and about a third of the time. At practical widths the model
  spreads a good approximation across units instead of finding a compact
  formula, so the interpretability claim should be scoped accordingly.

## The algorithm

`training/structure_search.py`, four phases:

| Phase | What it does |
|---|---|
| 0 warm up | Train weights on the initial structure, so structural comparisons face a roughly stationary objective |
| 1 operators | Every node, every sweep. Each candidate probed from identical weights with an identical budget. `search_mode='exhaustive'` evaluates every assignment when the space is small enough |
| 2 topology | Entered only if Phase 1 improved something. Grow/shrink moves, function-preserving on growth |
| 3 stability | Structure is stable when a sweep and a topology round both accept nothing |
| 4 compress | Reduce `hidden_dim` under an equal-budget ranking, then verify the winner at full budget against the incumbent |

Design choices that matter:

- **Every node is visited every sweep.** Ranking nodes by gradient norm — the
  original approach — has a systematic depth bias that concentrates nearly all
  proposals on the earliest layer, leaving most of the network never searched.
- **Candidates are enumerated deterministically**, not discovered with 5%
  probability, so a run repeats **at a fixed thread count**. Seeding alone is
  not enough: multithreaded BLAS reductions are order-dependent, and the same
  seed gives different results under different `OMP_NUM_THREADS`. The difference
  is 6th-significant-figure per epoch but compounds — `GAIOptimizer` compares
  against a stagnation threshold, so it shifts *when* evolution fires and the
  runs diverge into different mutation counts. Set `OMP_NUM_THREADS` identically
  for any runs you intend to compare, or pass
  `seed_everything(seed, threads=N)`.
- **The read-out is trainable during a probe.** A frozen read-out is still
  scaled for the operator being replaced, which penalises any candidate that
  changes the output's scale.
- **A sweep is reverted** if consolidation cannot recover the loss it started
  from. Probe scores are a proxy; this stops a proxy from degrading the model.
- **Test data is read once**, in `finalise`.

## Operator basis

| name | formula | class | in search? |
|---|---|---|---|
| `identity` | `x` | identity | yes |
| `tanh` | `tanh(x)` | saturating | yes |
| `relu` | `max(0, x)` | rectifier | yes |
| `sigmoid` | `1/(1+e^-x)` | saturating | no — exactly redundant with `tanh` |
| `sin` | `sin(pi*x)` | periodic | yes |
| `cos` | `cos(x)` | periodic | no — exactly redundant with `sin` |
| `gaussian` | `e^(-x^2)` | bump | yes |
| `square` | `x^2` | quadratic | yes |
| `leaky_relu` | `x if x>=0 else 0.01x` | rectifier | no — near-redundant with `relu` |

Every operator sits between two trainable affine maps, so some are *exactly*
interchangeable: `cos(Wx+b) = sin(pi(W'x+b'))` with `W'=W/pi, b'=b/pi+1/2`, and
`sigmoid(u) = 1/2 + 1/2*tanh(u/2)` with the following layer absorbing the scale.
Dropping the aliases shrinks the space without losing expressivity, and stops
seed-to-seed differences that carry no functional content.

Composition — `a_of_b` = `a(b(x))`, `a_x_b` = `a(x)*b(x)`, `a_plus_b` =
`a(x)+b(x)` — each wrapped in a soft clip `1e6*tanh(t/1e6)`. Per node per sweep:
6 basis candidates plus 19 composites anchored on the basis winner.

Operator names are parsed **once** into an expression tree, and both the torch
and SymPy interpreters walk that same tree, so `export_formula` reproduces
`forward` to within 1e-8.

## Install

```bash
pip install -r requirements.txt
```

## Run

Correctness invariants (fast, no training):

```bash
python tests/test_invariants.py
```

Is the architecture worth anything? Is the search?

```bash
python experiments/mlp_comparison.py --seeds 3
```

Does the search recover a known ground-truth operator set?

```bash
python experiments/operator_recovery.py --seeds 8 --restarts 4
```

Do different seeds agree on the discovered architecture?

```bash
python experiments/seed_stability.py --seeds 6
```

Does composition pay off where the target contains products?

```bash
python experiments/lorenz_composites.py --seeds 3
```

## How the measurements are designed

The negative results here are only worth anything if the controls are right, so
the design is stated explicitly rather than left implicit.

- **Matched budgets.** Every arm in a comparison gets the same number of
  gradient steps. The restart study gives all three arms exactly 6000 — one long
  run for the search, `8 x 750` for the restart arms.
- **An oracle arm establishes the ceiling.** Handing the method the correct
  operators separates "this problem is hard" from "this search is bad". It is
  also a feasibility gate: when the oracle fails, the other arms at that setting
  are uninformative and are reported as such rather than read as evidence.
- **Chance baselines, computed not assumed.** Random-operator arms are run, and
  the expected hit rate is derived beforehand — e.g. `2/36` unordered at 2 nodes.
  Null predictions were registered *before* the cells finished.
- **Selection on validation, never test.** The test split is read once, for the
  model already chosen. Best-of-K on test is the leakage that made an earlier
  result in this project look far better than it was.
- **Paired seeds.** Arms share seeds and initial weights, so a comparison is not
  confounded by initialisation luck. Absolute rates need far more seeds than
  paired comparisons do — the same quantity read 38% at 8 seeds and 11% at 28.
- **Significance stated, including when it fails.** Fisher exact / binomial
  tests accompany the claims. Several differences that look decisive are not:
  `3/8 vs 0/8` is p = 0.20, and it is reported that way.
- **Instrumented, not inferred.** Structural events are read from an event log
  in the optimiser. Recovering them from the loss curve — the original approach
  — counted Adam spikes as mutations at 16 false positives per real one.

## Repository layout

```tree
models/            operator library, searchable node, chain ensembles
training/          structure_search.py (exhaustive search), trainer.py (GAIOptimizer)
experiments/       every measurement quoted above
results/logs/      raw stdout of every run behind those numbers
theory/            what the method does and where it breaks
tests/             62 correctness invariants
legacy/            written against a deleted API; does not run (see legacy/README.md)
```

Every figure quoted here traces to a log in `results/logs/` — they are tracked
deliberately, because several runs take hours and the numbers are otherwise
unverifiable.

## Reproducibility: set `OMP_NUM_THREADS`

Seeding is **not sufficient**. Multithreaded BLAS reductions sum in a
nondeterministic order, so an identical seed gives different results at
different thread counts:

| `OMP_NUM_THREADS` | test loss, identical seed |
|---|---|
| 1 | 1.179647445679e-01 |
| 3 | 1.179661080241e-01 |
| 4 | 1.179633960128e-01 |

A 6th-significant-figure difference is not harmless here: `GAIOptimizer`
compares scores against a stagnation threshold, so a tie-break landing the other
way changes *when* evolution fires, and the runs diverge into different mutation
counts (126 vs 137 on the same 6 seeds). **Fix the thread count across any runs
you intend to compare**, or pass `seed_everything(seed, threads=N)`.

## Status

Working and measured: `models/`, `training/`, `tests/`, and the `experiments/`
scripts referenced above. `python tests/test_invariants.py` — **62 invariants**,
all passing.

**One experiment is incomplete.** `experiments/scaling_regime.py` finished 5 of
6 cells; the 4-node `random` control was not run. See [RESUME.md](RESUME.md) for
the single command that completes it, and for the pre-registered null prediction
that cell is testing.

**Quarantined:** 13 scripts written against an API that no longer exists have
been moved to [`legacy/`](legacy/README.md). They raise `ImportError` and are
kept only for provenance — nothing in this README depends on them. Every
remaining `.py` file parses and imports cleanly.

**A prior claim was withdrawn, not restated.** An earlier version of this README
carried a results table (Lorenz 4.5e-4 MSE, 256-bit parity solved, double
pendulum 1300x better than RNNs, E. coli 96.5% vs 88.2%) that the committed
artifacts do not support and in one case directly contradict:
`results/comparative_benchmark.png` shows XGBoost at ~95% accuracy against this
model at ~87%, with zero precision and zero recall — a collapse to
majority-class prediction. Those numbers are gone. Everything quoted now comes
from a held-out split read exactly once, with the raw log committed under
`results/logs/`.

## Details

- [CHANGES.md](CHANGES.md) — every fix, every measurement, and the bugs found in
  the measuring apparatus itself (three of them inflated or invented evidence
  that the search was working)
- [RESUME.md](RESUME.md) — current state, what is unfinished, exact commands
- [theory/structural_learning.md](theory/structural_learning.md) — method and measured limits
- [theory/importance_metric.md](theory/importance_metric.md) — why node ranking was dropped
- `results/logs/` — raw logs behind every number quoted here

### Open questions worth more seeds

- The **Lorenz 2.02x** win rests on 3 seeds. That is the same sample size that
  reported 38% for a recovery rate whose true value at 28 seeds is 11%. Re-run
  at >=16 before building on it.
- The **SINDy head-to-head** was run this session (SINDy won 4/5, three by 9-10
  orders of magnitude) but never written to disk, so those figures are recalled
  rather than sourced. Re-run and record.
- Whether the search beats chance **at all** is formally unresolved — every test
  is a null, but separating an ~11% rate from 0% needs far more seeds than any
  run here used.

## License

MIT — see [LICENSE](LICENSE).
