# Theory: Structure-Aware Learning

Structure-Aware Learning (SAL) treats the choice of per-layer activation
operator as a searchable variable alongside the weights.

## The core hypothesis

Traditional deep learning optimises weights $W$ inside a fixed topology
$\mathcal{G}$ with a fixed activation. SAL proposes that the choice of local
operator is also worth optimising, because a model that can select
$\sin(\cdot)$ directly should approximate periodic structure with far fewer
parameters than one composing ReLUs.

The premise holds. On a smooth periodic target, a two-hidden-layer MLP of
identical size reaches `4.93e-05` with `tanh` and `3.48e-04` with `relu` — a
**7x** difference from the activation alone
(`experiments/mlp_comparison.py`).

## What the operator space actually is

Per **layer**, not per neuron: `MatrixSymbolicNode` applies one operator
elementwise to a whole `[batch, hidden_dim]` pre-activation, so all units in a
layer share it. For an MLP of depth $d$ over a basis of size $b$ the space is
$b^d$ — 216 for six operators and three layers, small enough to enumerate
exhaustively (`search_mode='exhaustive'`).

The basis is six operators after removing exact redundancies: `identity`,
`tanh`, `relu`, `sin`, `gaussian`, `square`. Composition (`_of_`, `_x_`,
`_plus_`) extends it. See `models/activations.py`.

## The evolutionary learning cycle

1. **Warm up** weights on the initial structure, so structural comparisons are
   made against a roughly stationary objective.
2. **Operator search** — every node, every sweep, every candidate probed from
   identical weights with an identical budget.
3. **Topology search** — grow/shrink moves, function-preserving on growth.
4. **Compression** — reduce `hidden_dim` under an equal-budget comparison.

Full description in `training/structure_search.py`.

## Measured limits

### The chain ensemble does not help

At matched parameters, `MatrixGGLEN` with random fixed operators loses to a
plain MLP on all three tasks measured. Summing parallel chains before a single
shared read-out is a constraint, not a feature: one chain's error cannot be
corrected by another, which also degrades the search's credit assignment.
`num_chains=1` — literally an MLP with searchable activations — is the better
configuration. `readout='concat'` is available if multiple chains are wanted.

### The search is task-dependent, not a general win

Improvement over the same architecture with fixed random operators:

| task | effect of search |
|---|---|
| `sin(pi*x1) + x2**2` | **3.5x better** |
| Lorenz derivatives | **2.4x better** |
| random tanh-MLP teacher | **19% worse** |

It helps when the target's structure is a short composition of the basis, and
hurts when it is not. The regression on the teacher task is selection
overfitting: choosing the best of ~25 candidates per node on a small validation
split fits the split. Budget a larger validation set, or penalise selection.

### Weight optimisation, not operator selection, is the binding constraint

With the operators **forced** to the known-correct `[sin, square]` for
`sin(pi*x1) + x2**2`, only 2 of 12 weight initialisations reach the exact
solution; the rest plateau near `9.5e-02`. That plateau is the same value the
best of all 36 operator assignments reaches. So a single-initialisation search
is measuring initialisation luck, not operators.

`sin(\pi(w \cdot x + b))` is periodic in $w$, so gradient descent from a random
$w$ settles into a wrong frequency basin. The search therefore systematically
**undervalues** `sin` — penalising it for having the hardest loss landscape
rather than the worst fit.

**Initialisation is not the fix.** Xavier, wide `U(-3,3)`, narrow `U(-0.3,0.3)`,
and a frequency-diverse scheme that gives each unit its own log-spaced frequency
scale were compared over 16 seeds with the operators forced correct:

| hidden_dim | exact hits (of 16), any scheme | median loss |
|---|---|---|
| 1 | 2–3 regardless of scheme | ~9.5e-02 |
| 4 | 0 | ~4e-05 |
| 8 | 0 | ~1e-05 |

Two conclusions. Restarts are the only lever at width 1 (`probe_restarts`), and
**width trades exactness for reliability**: at `hidden_dim=1` you occasionally
reach machine precision and usually plateau; at practical widths you never reach
machine precision but reliably land near `1e-05`.

**But the *reset* width is a different quantity, and it does matter.** The table
above varies the *initial* initialisation with the operators held correct. It
says nothing about the draw used to re-initialise a node *after a mutation*,
inside a search that is still choosing operators. Those come apart, because the
reset width does not only change how well a given operator fits — it changes
**which operator the search selects**. Measured on `sin(pi*x1) + x2^2` at
`hidden_dim=1`, 8 seeds, 6000 epochs, fully crossed, all four cells inside the
arm where a reset actually happens (`legacy_sa=True, mutation_mode='reset'`;
under `mutation_mode='homotopy'` the reset is undone before it can act, so that
arm cannot speak to this):

| non-square reset draw | stationary-point offset | recovery | best test |
|---|---|---|---|
| `U(-0.05, 0.05)` | off | **3/8** | **7.647e-16** |
| `U(-0.05, 0.05)` | on | 2/8 | 7.857e-16 |
| Xavier `U(-1.414, 1.414)` | off | 0/8 | 2.653e-04 |
| Xavier `U(-1.414, 1.414)` | on | 0/8 | 2.653e-04 |

The draw width accounts for the whole effect — 6/16 against 0/16 pooled over the
offset, Fisher exact p ~ 0.02 — while nudging `square`/`gaussian` off their
stationary point is worth at most one seed. The mechanism is the one
identified two paragraphs above, applied per-proposal: a node mutated to `sin`
and reset wide starts at a random frequency **every time `sin` is proposed**, so
the search scores `sin` on a wrong-basin fit and rejects it. A narrow reset
starts the node almost linear, letting $w$ grow continuously into the correct
frequency. The narrow draw is therefore not an under-scaled initialisation to be
corrected — it is a start-simple-and-grow prior that partially offsets the
search's structural bias against periodic operators.

This is why `reset_scale` defaults to `'small'` rather than `'xavier'`, and why
`tests/test_invariants.py` pins that default: the Xavier draw is what
first-principles reasoning recommends, and on this task it costs every exact
recovery the method is capable of.

### What the search is worth, against a budget-matched control

The section above says weight optimisation is the binding constraint. That is
true *within* a probe, and it turns out to be false for the method as a whole.
Measured with every arm given exactly 6000 gradient steps, the same learning
rate, optimiser and reset draw, all selecting on validation
(`experiments/restart_baseline.py`), at 28 seeds:

| arm | recovery | median test |
|---|---|---|
| forced to `sin, square` + 8 restarts | **28/28 (100%)** | **7.66e-15** |
| GAIOptimizer search, 1 x 6000 ep | 3/28 (11%) | 8.39e-02 |
| random operators + 8 restarts | 0/28 (0%) | 8.12e-02 |

`p = 1.2e-12` for the ceiling against the search; `p = 0.24` for the search
against random operators, i.e. **the search's margin over chance operator
assignment is not statistically established even at n=28**.

The 8-seed cross that separates the two factors:

| operators | 1 run x 6000 ep | 8 restarts x 750 ep |
|---|---|---|
| random | 0/8 | 0/8 |
| forced to `sin, square` | 1/8 | **8/8** |
| GAIOptimizer search | 3/8 (see caution below) | 1/8 |

The forced-operator arm settles a confound before it can be raised: 750 epochs
per restart is not too short. Given the right operators, that budget reaches
machine precision on every seed. So the 0/8 in the random-operator arm is not a
convergence failure — it is an operator-selection failure, full stop.

The crossed design shows the two factors **interact**, and that neither is
sufficient alone:

- restarts given correct operators: 1/8 -> 8/8, **p = 0.0014**
- correct operators given restarts: 0/8 -> 8/8, **p = 0.00016**
- correct operators without restarts: 1/8 vs 0/8, **p = 1.0**

Exact recovery at width 1 is therefore a conjunction: the right operators AND
enough independent draws to land in the right frequency basin. This is the same
periodicity argument as above, now quantified — one trajectory with the correct
operators still fails 7 times in 8, because `w` starts in the wrong basin and
cannot leave it.

Three conclusions, in descending order of how well they are supported:

1. **Operator selection is the whole bottleneck, and it is otherwise a solved
   problem.** Given the correct operators and restarts, recovery is 28/28 —
   every seed, median 7.66e-15. The search reaches 3/28 (**p = 1.2e-12** against
   that ceiling). Weight optimisation, initialisation, annealing schedule and
   node ranking are all downstream of that one number.
2. **Whether the mutation loop beats chance operator assignment is STILL
   unresolved, now at n=28.** It was reasonable to suspect the loop is only a
   restart mechanism — no mutation is followed by a new global best within 60
   epochs, the best model usually predates the first mutation, and recovery
   collapses whenever the mutated node keeps its weights. The search's 3/28
   against random operators' 0/28 points the other way but is **p = 0.24**.
   Separating an 11% rate from a 0% rate needs far more seeds than this; the
   honest statement is that no measurement in this repo demonstrates the search
   selects operators better than chance.
3. **Restarts do not rescue it, because they compete with it for budget.**
   Running the search itself as 8 x 750-epoch restarts gives 1/8 exact
   recoveries against 3/8 for one long run, while improving the median 4.7x.
   750 epochs is ample to fit weights once operators are known (28/28 in the
   oracle arm) and far too few to discover them. Exact recovery wants one long
   trajectory; typical-case accuracy wants many short ones.

**A sample-size caution, learned here by getting it wrong.** The search arm
measured 3/8 = 38% on seeds 0-7 and 3/28 = 11% on seeds 0-27; seeds 8-27
contributed no recoveries. Absolute recovery rates from 8 seeds in this
repository are optimistic and should be re-measured before being quoted. Paired
comparisons at 8 seeds are unaffected, since both arms see the same seeds.

This also reframes the width-1 restart advice above: restarts are a *solved and
cheap* lever, worth 8/8 once operators are right, and the mutation loop does not
use them at all — it spends its entire budget on one trajectory. Search combined
with restarts is the obvious untried configuration.

### Nesting, not depth, is what destroys exact recovery

The width result above ("width trades exactness for reliability") has a depth
analogue, and it is not the obvious one. Measured with the operators FORCED to
the ground truth, so operator selection cannot be blamed
(`experiments/scaling_regime.py`, teacher-generated targets, 8 restarts):

| teacher | nodes | oracle recovery | median test |
|---|---|---|---|
| `sin + square` | 2 | **4/4** | 1.3e-15 |
| `sin,identity + square,identity` (same function, padded to depth 2) | 4 | **4/4** | 3.1e-15 |
| `gaussian(sin(.)) + square` (nested) | 4 | **0/4** | 4.9e-03 |

The nested cell stays 0/4 at four times the budget (24,000 epochs, 3,000 per
restart), so it is not a convergence-time issue.

**Depth costs nothing. Nesting costs everything.** Padding a chain with identity
nodes leaves exact recovery untouched at machine precision; replacing one
identity with a genuine second nonlinearity destroys it, even when every
operator is already correct and the target is exactly representable by
construction.

This is the most restrictive scoping result in this file, because **composition
is the method's premise**. The `a_of_b` / `a_x_b` / `a_plus_b` composite
operators exist precisely to express nested targets, and nested targets are the
ones whose weights cannot be optimised to machine precision. The regime where
exact symbolic recovery works is: a SUM of un-nested basis operators, at width
1. Outside it, the model returns a good approximation rather than a formula.

A useful side effect: the identity-padded teacher keeps the function easy while
making the SEARCH hard — 1,296 assignments at 4 nodes against 36 at 2 — which
is the only configuration in which "does the search beat chance at scale?" can
be asked at all, since every harder function fails the feasibility gate.

### Consequence for the interpretability claim

The `1e-14` results — an exactly recovered closed form — occur only at
`hidden_dim=1`, and only about a third of the time. At usable widths the model
finds a good approximation distributed across many units, not a compact formula.
Symbolic export remains faithful (`export_formula` matches `forward` to 1e-8),
but the exported expression is a large sum of products rather than a readable
law. The interpretability pitch should be scoped to the width-1 regime.

## What the evidence supports

*Per-layer operator search on an MLP buys roughly 1.6-1.8x parameter efficiency
on targets whose structure is a short composition of the operator basis, at
about 1.4x the training compute.*

That is narrower than "structural expressivity compensates for parameter
sparsity", and it is what the measurements show.
