# What changed

Two kinds of change: bug fixes to existing code, and a new implementation of
the four-phase search the project describes. Public class names and
constructor signatures are unchanged, so existing scripts still import.

Verify with:

```bash
python tests/test_invariants.py
```

## The algorithm, as now implemented

`training/structure_search.py`. The phases follow the stated idea:

| Phase | What it does |
|---|---|
| 0 warm up | Train weights on the initial structure, so structural comparisons are made against a roughly stationary objective |
| 1 operators | Greedy coordinate descent: every node, every sweep, all candidate operators probed from identical weights with identical budget |
| 2 topology | Entered only when Phase 1 improved something. Grow/shrink moves, function-preserving on growth |
| 3 stability | Structure is stable when a sweep and a topology round both accept nothing |
| 4 compress | Shrink `hidden_dim` under an equal-budget comparison, then verify the winner at full budget against the incumbent |

Operators are still composed from basis functions (`a_of_b`, `a_x_b`,
`a_plus_b`), but the candidate set is **enumerated deterministically** instead
of discovered with 5% probability. That removes the *algorithmic* source of
irreproducibility. A numerical one remains and is not fixed by seeding: results
differ across `OMP_NUM_THREADS` values (see "Reproducibility" below), so a run
is repeatable at a fixed thread count, not across arbitrary ones.

Scoring uses a three-way split. The test split is read exactly once, in
`finalise`.

## Bug fixes

### models/activations.py — rewritten

- **Operator semantics had two disagreeing implementations.** `forward` and
  `export_formula` each parsed operator names separately and reached opposite
  conclusions: `sin_of_square` executed as `sin(x)**2` but exported as
  `sin(x**2)`. Names are now parsed once into an expression tree, and both the
  torch and SymPy interpreters walk that same tree. `tests/test_invariants.py`
  checks they agree to 1e-9 over 60+ operators.
- **Unknown operators silently became the identity**, so a typo or an
  unrecognised composite produced a plausible-looking wrong formula. Now raises.
- **Nested names lost terms**: `'a_of_b_of_c'.split('_of_')` yields three parts
  and the old code used only two, dropping `c`. The parser has explicit
  precedence and handles nesting.
- Commutative composites are canonically ordered, so `sin_x_cos` and
  `cos_x_sin` no longer occupy the space as if they were different operators.
- Added `EQUIVALENCE_CLASSES`. Because every operator sits between two
  trainable affine maps, some are *exactly* interchangeable —
  `cos(Wx+b) == sin(pi(W'x+b'))` and `sigmoid(u) == 1/2 + 1/2 tanh(u/2)` with
  the following layer absorbing the scale. `SEARCH_BASIS` keeps one
  representative per class (6 operators, down from 9).

### models/symbolic_neuron.py — rewritten

- **The operator registry was a shared global.** Every node held a reference to
  one module-level dict, so a composite discovered by one node appeared in every
  other node in the process — including baselines — and results depended on
  execution order. Each node now gets its own copy.
- **The 25-operator cap was dead code.** It looked for keys "not in
  `ACTIVATIONS`", but discoveries were written *into* `ACTIVATIONS`, so the list
  was always empty and the library grew without bound.
- Importance is now `mean |dL/dz * z|` (first-order Taylor) rather than
  `mean ||dL/dz||`; see `theory/importance_metric.md` for the depth-bias
  argument. `importance_mode='gradnorm'` keeps the original for comparison.
- Dropped a redundant `retain_grad()` that held a `.grad` on every intermediate
  activation. Added `set_tracking()` so the hooks can be switched off during the
  thousands of probe steps that do not need them.
- `op_name` is a property whose setter recompiles the operator, so existing code
  that assigns to it directly stays correct.

### models/adaptive_neural_model.py

- `export_formula` on the ensemble exported `chains[0]` only, dropping both the
  cross-chain sum and the final linear layer — the printed formula was not the
  model's function. It now composes the whole model, and refuses with a clear
  error when the expansion is intractable (the 784-input MNIST case used to hang).
- Added function-preserving `add_chain` / `deepen_chain`, plus `prune_chain` /
  `shrink_chain`. Growth leaves the output bit-identical, so a topology move is
  judged on what training does with the capacity rather than on reinitialisation.
- **Pruning a middle expert scrambled the gate.** The gated model copied the
  first N gate rows, so pruning chain 1 of 3 kept rows 0 and 1 instead of 0 and 2.
- Added `checkpoint()` / `save()` / `load()`. `op_name` is a plain string and so
  absent from `state_dict()`; saving weights alone loses the discovered
  architecture, which is the entire output of the search. There was no
  `torch.save` anywhere in the repo.
- Added `readout='sum' | 'concat'`. `'sum'` is the original behaviour and stays
  the default; `'concat'` gives each chain its own read-out coefficient. See
  `experiments/mlp_comparison.py` for why this matters.

### training/trainer.py

- **`lr` was discarded after the first mutation.** Three rebind sites hardcoded
  `lr=0.001`, so a caller asking for 0.01 silently got 0.001 as soon as
  evolution triggered.
- **`surgical_evolve` could never succeed on a classification task.** It cast
  targets with `torch.tensor(y_val).float()` and no unsqueeze. Under
  `CrossEntropyLoss` that raises, and a bare `except` turned the failure into a
  sentinel `999.0` for both the baseline and the post-op score — so `delta` was
  always exactly `0.0`, `if delta > 0` was always False, and surgery always
  reverted. Under `MSELoss` it broadcast `[N]` against `[N,1]` into an `[N,N]`
  comparison and produced a meaningless number. Targets are now shaped to match
  the criterion, and the bare excepts are gone.

### use_cases/physics/gai_lorrenz.py

- `c_id, l_id = model.evolve_structure(...)` unpacked a 3-tuple as 2, raising
  `ValueError` at the first stagnation event — i.e. the moment evolution began.
  This is one of the two commands the README documents.
- Seeded `random`, which drives every mutation and the annealing coin flip.
  Seeding only `torch` and `numpy` left the architecture search unseeded.

### Project metadata

`requirements.txt` (14 previously-undeclared dependencies), `.gitignore`
(`__pycache__` and the ~190 MB of committed datasets), `LICENSE` (the README
badge had no file behind it).

## New experiments

| Script | Question |
|---|---|
| `experiments/operator_recovery.py` | On a target whose right answer is known, does Phase 1 find it? |
| `experiments/mlp_comparison.py` | Is the chain architecture worth anything over an MLP? Is the search worth anything over the architecture? |
| `experiments/global_activation_baseline.py` | Does per-node search beat just trying a few global activations? |
| `experiments/seed_stability.py` | Do different seeds agree — on performance, on structure, on per-slot operator choice? |
| `experiments/mutation_gain.py` | Paired control: same seed, same initial weights, mutation on vs off. Does mutating reach a better model than just training? `--sa legacy fixed` compares the original SA against the fixed one. |
| `experiments/mutation_mechanics.py` | Reset the mutated node's weights, or carry them into the new operator? Also measures shock, recovery time, and what share of mutations produce a new global best. |
| `experiments/gai_recovery.py` | Does `GAIOptimizer` recover a known closed form, and does tuning or the SA fix change the rate? |
| `experiments/best_model_audit.py` | Does the run keep the best model it finds, or lose it to later mutations? |
| `experiments/restart_baseline.py` | The control this project never ran: at an equal epoch budget, does the search recover more often than plain independent restarts with random operators? |

## Mutation mechanics (after an AutoML literature review)

A measured audit found that `GAIOptimizer` never loses the best model it finds
-- `_update_best` snapshots weights AND structure, and `fit()` restores them --
but that a run typically finds its best early and then degrades for the rest of
the budget. A paired control (same seed, same initial weights, mutation on vs
off) put the median gain at 0.43x: **not mutating reached a better model.** The
changes below address the mechanisms behind that.

All are behind `legacy_sa=True`, which restores the original behaviour exactly.

### models/symbolic_neuron.py

- **Homotopy operator swap** (`begin_swap` / `set_swap_t` / `cancel_swap`).
  `g_t = (1-t)*g_old + t*g_new`, annealed over the grace period. At `t=0` the
  network is bit-identical to before the mutation, so a swap costs nothing and
  the accept/reject test measures the OPERATOR instead of the damage from
  re-initialising the node. Generalises the P-activation family of Network
  Morphism (Wei et al., ICML 2016, arXiv:1603.01670) from `phi -> identity` to
  an arbitrary operator pair. This also makes the reset-vs-transfer question
  moot: at `t=0` the existing weights are trivially correct.
- **Two apparent initialisation defects in `reset_weights_near_identity` — both
  reverted after measurement.** This entry previously claimed them as fixes.
  They are not, and the record is corrected here rather than quietly deleted.

  The identity component applies only when `in_features == out_features`, so a
  non-square node gets a bare `U(-0.05, 0.05)` draw -- for
  `hidden_dim=32, input_dim=3` roughly 8x below Xavier (std 0.029 against
  0.239). Replacing it with a correctly scaled Xavier draw is the textbook
  correction, and it **measured worse** on the one metric where this project has
  its strongest result:

  Fully crossed against the stationary-point offset below, 8 seeds, 6000 epochs,
  all four cells inside the same arm (`legacy_sa=True, mutation_mode='reset'`,
  the only arm where the reset is actually on the path):

  | non-square reset | offset | recovery | best test |
  |---|---|---|---|
  | `U(-0.05, 0.05)` (original) | off | **3/8** | **7.647e-16** |
  | `U(-0.05, 0.05)` | on | 2/8 | 7.857e-16 |
  | Xavier `U(-1.414, 1.414)` | off | 0/8 | 2.653e-04 |
  | Xavier `U(-1.414, 1.414)` | on | 0/8 | 2.653e-04 |

  Pooled over the offset that is 6/16 against 0/16 (Fisher exact p ~ 0.02).

  At `hidden_dim=1` every node is `Linear(2->1)`, where Xavier's bound is
  `sqrt(6/3)` = 1.414 -- a 28x wider draw. For `sin(pi*w*x)` a near-zero `w`
  starts almost linear and can grow smoothly into the correct frequency, while a
  wide random `w` starts at ~1.4 oscillations across the domain, in a wrong
  frequency basin that gradient descent cannot leave. The narrow draw was not an
  oversight; it is load-bearing as a start-simple-and-grow prior. `small` is the
  default; `reset_scale='xavier'` remains available for architectures where
  under-scaling actually costs something.

  Separately, `square`/`gaussian` were placed at `z ~ 0`, a stationary point of
  their own derivative. Offsetting the bias so they wake on a sloped part of the
  curve measured **neutral** (rows 1-2 against 3-4 above): one seed either way,
  with an effectively identical best loss — inside the noise at n=8. Off by
  default; `reset_offset_stationary=True` enables it. The mechanism is real but
  the benefit is unmeasured.

  Scope: under `mutation_mode='homotopy'` the reset is undone before it can act
  (see the dead-flag entry above), so `reset_scale` cannot affect that arm
  mechanistically — it only shifts the global RNG stream, since the discarded
  reset still consumes draws. The homotopy arm did move (3/8 -> 1/8) when the
  scale changed, and that is a seed perturbation, not an effect. It is reported
  separately for exactly that reason and should not be read as evidence either
  way.

  `tests/test_invariants.py` now pins both defaults, so the same
  reasoning-from-first-principles cannot silently re-break them.

  Two lessons, both mine: a change justified by general theory still has to be
  measured in the specific regime it will run in, and it has to be measured on
  the metric the project is actually judged by. Recovery at `hidden_dim=1` is
  not the regime Xavier scaling was derived for.

  An open question this raises, deliberately NOT claimed as a finding: the
  homotopy swap also preserves the mutated node's weights, so it should remove
  the same fresh draw. GAI-A recovery is 5/16 under `legacy` against 1/16 under
  `fixed` (homotopy) — but that is **p = 0.17**, and 3/8 vs 1/8 at 6000 epochs
  alone is p = 0.57. Underpowered, so it stays a hypothesis. The `fixed-reset`
  arm added to `gai_recovery.py` (all SA changes EXCEPT homotopy) tests it
  directly, and needs more seeds than 8 to resolve.

### training/trainer.py

- **Stagnation measured against a rolling reference, not the all-time best.**
  Previously `epochs_no_improve` reset only when the score beat the global best,
  so once a run peaked the counter incremented essentially every epoch and
  evolution fired every `patience` epochs for the remainder of training. This is
  the mechanism behind the observed "finds its best at epoch 1353, then thrashes
  for 2,647 epochs" behaviour.

  **This change was INERT until it was fixed twice more.** As first written it
  did nothing at all, for two independent reasons:

  1. `rolling_reference` started at `-inf`, making the update threshold
     `-inf + delta*abs(-inf)` = `-inf + inf` = `nan`. `x > nan` is False for
     every `x` and every `min_rel_delta`, so the reference could never leave
     `-inf` and the else-branch incremented unconditionally. It is now seeded
     from the first observed score.
  2. `_update_best` set `epochs_no_improve = 0` internally, overriding the
     rolling block on any epoch that set a new global best. The counter could
     therefore never exceed 1 while a run was still improving — measured
     directly: 200 epochs of training ended with `epochs_no_improve == 1` and
     **zero** mutations. The reset now lives at the call site and applies only
     on the `legacy_sa` path, so the rolling block owns the counter.

  Consequence for results already reported: every `fixed`/non-legacy arm run
  before this point used a stagnation signal that was still effectively
  "epochs since the last global best" — i.e. the legacy signal. Differences
  measured between the `legacy` and `fixed` arms are attributable to the OTHER
  changes (homotopy swap, revert cooldown, preserved Adam moments,
  scale-invariant acceptance, finite tabu), not to this one. The `legacy` arm is
  unaffected throughout, since it never enters this branch.
- **Cooldown after a rejection.** `_revert` set `epochs_no_improve = patience+1`,
  re-triggering on the very next epoch with zero intervening training -- against
  a measured 4-6 epoch recovery horizon.
- **Adam moments preserved** across mutations and reverts. Both paths rebound
  the optimiser, discarding first and second moments every time.
- **Scale-invariant acceptance.** `exp(delta/temp)` used `delta` in absolute
  loss units, so one `initial_temp` accepted nearly everything where MSE ~ 1e-5
  and nearly nothing where the loss was ~1. Now normalised by the incumbent.
- **Finite tabu tenure, and exhaustion no longer ends training.** The forbidden
  set was append-only; after ~5 rejections per node `evolve_structure` returned
  None and `fit()` did `break`. In any paired comparison that silently gave the
  no-mutation arm more epochs than the mutating arm. (Verified it did not fire
  in the runs reported here -- both arms completed all 4000 epochs -- but it is
  reachable at higher rejection rates.)
- **Mutations judged on the best score across the grace window**, not a single
  epoch's reading.
- `save_best(path)` writes weights AND structure together. Nothing in the
  project previously persisted a model; every discovered architecture, including
  an exact recovery at test 7.65e-16, was discarded at process exit.

### Mutations were being counted by looking for loss spikes

`mutation_gain.py` and `mutation_mechanics.py` located structural events by
scanning the validation curve for a >3x single-step jump, on the stated
assumption that a mutation always leaves a visible discontinuity. That
assumption is wrong in both directions, and the error is large.

Checked against a real event log on GAI-A / `gauss_of_sin`, 1500 epochs:

| seed | mutations actually fired | "detected" from the curve |
|---|---|---|
| 0 | **1** (epoch 1481) | 16 |
| 1 | **0** | 16 |

The detected epochs arrive in tight clusters — `864, 868, 880, 883, 886, 891,
894`, seven inside thirty epochs — which is one unstable stretch of Adam, not
seven structural events. The detector was measuring **training instability** and
labelling it mutation. (It fails the other way too: a homotopy swap at `t=0` is
bit-identical to no swap, so it leaves nothing to detect.)

Every mutation count and payoff rate derived this way is dominated by false
positives, including the "66 mutations, 0 of them productive" style rows and any
"share of mutations that produced a new global best" figure. The `gain` column —
best-of-mutating against best-of-frozen — is unaffected, since it only reads the
minimum of each curve.

`GAIOptimizer.mutation_log` now records `{epoch, chain, layer, old_op, new_op,
accepted, delta}` per event at the point the mutation is applied and judged, and
both experiments read it. A second, independent finding falls straight out of
the real counts: **GAI-A fires 0-1 mutations in 1500 epochs**, so at this budget
its search is close to inert — the same pathology already noted for GAI-B.

### `mutate_reset_weights` is dead under `mutation_mode='homotopy'`

The homotopy path restores `backup_state` and `backup_chains` after
`evolve_structure` returns, deliberately undoing the operator install *and* the
weight reset so the swap can be re-entered as a function-preserving morph. A
consequence that was not noticed: `mutate_reset_weights` then has no observable
effect at all in that mode.

`experiments/mutation_mechanics.py` built its whole reset-vs-transfer comparison
out of that one flag and never set `mutation_mode`. Both arms therefore ran
*identical* configurations, differing only in how many draws the discarded reset
took from the global RNG before being thrown away — so the two arms are two
random seeds wearing different labels, and any gap between them is noise. Fixed
by pinning `mutation_mode='reset'` on both arms, which is what makes the
distinction live. `tests/test_invariants.py` now asserts the interaction
(`max |dW| = 0` under homotopy for both flag values; non-zero only for
`mode='reset', reset=True`), so the next comparison built this way fails loudly.

### Measured effect

Paired control, 6 seeds, `gauss_of_sin`, GAI-A, 4000 epochs — same seed means the
same initial architecture AND the same initial weights, so the arms differ only
in whether mutation is allowed to fire. Mutation counts are now read from
`mutation_log`, not inferred from the curve.

| | median gain | mutation helped | mutations fired | producing a new best |
|---|---|---|---|---|
| legacy SA | 0.76x | 1/6 seeds | 137 | **0 (0%)** |
| fixed SA | **0.86x** | **3/6 seeds** | **52** | 2 (4%) |

Both arms are below 1.0x, so **not mutating still reaches a better model on the
median** — that conclusion has survived every version of this measurement. On 4
of 6 legacy seeds the mutating run's overall best was already reached *before
its first mutation fired*.

The `fixed` row is the first measured benefit from any of the SA changes. Note
especially the mutation COUNT: the repaired rolling reference fires **62% fewer**
mutations (52 against 137), which is the opposite of what I expected. The
mechanism is that the rolling reference counts any >0.1% relative improvement as
progress and resets the stagnation counter, and that happens constantly during
training; the legacy signal resets only on a new *global* best, which stops
occurring after the run peaks, so its counter climbs unchecked and evolution
fires every `patience` epochs for the remainder of the budget. Firing less often
and at better times is precisely what that change was meant to do, and with it
working the median gain moves 0.76x -> 0.86x with 2 productive mutations against
0.

Read the `0 / 126` carefully: it means no mutation was followed by a new global
best **within 60 epochs**, which is not the same as "mutation never helps".
Seed 4 is the counterexample and it should not be buried — it gained 4.11x over
its own frozen control and finished 12x below its pre-mutation best (2.18e-06
against 2.59e-05), while still scoring 0 attributable. Its payoff simply arrived
more than 60 epochs after any individual mutation.

Attribution here is window-sensitive and no window is obviously right: the same
class of run scores 0% at a 60-epoch window and 26% at 800, where ordinary
training would have supplied a new best regardless. That is why
`experiments/restart_baseline.py` compares END RESULTS at a matched epoch
budget instead — it needs no attribution window at all.

Two earlier versions of this table are withdrawn. The first reported 0.55x
legacy against 0.80x fixed with mutation counts recovered from the loss curve —
those counts were dominated by false positives. The second reported 0.66x legacy
with 126 mutations, measured while the rolling stagnation reference was still
inert, so its `fixed` arm did not measure the arm it named. The table above is
from the repaired code with counts read from `mutation_log`.

The legacy row also shifted between those runs (0.66x/126 mutations against
0.76x/137) on identical seeds and configuration. That is not a code difference —
it is the thread-count nondeterminism documented below. Both are valid runs of
the same experiment.

**Reset or transfer the mutated node's weights?** 5 paired seeds, GAI-A,
`mutation_mode='reset'` on both arms so the flag is actually live:

| arm | median best | median shock | median recovery | "productive" |
|---|---|---|---|---|
| reset | 5.37e-06 | **4321x** | 297 ep | 9/35 (26%) |
| transfer | 3.99e-06 | **545x** | 148 ep | 5/36 (14%) |

Transferring the weights costs 8x less shock and recovers twice as fast, yet
yields *fewer* productive mutations — the disruption of resetting appears to be
inseparable from whatever benefit it has, which is what the restart reading
predicts. Two caveats travel with this table and are repeated in the script:
`productive` uses an 800-epoch attribution window over which ordinary training
supplies a new best anyway (the same runs score 0/126 under a 60-epoch window),
and the two arms diverge in RNG stream because `reset` draws from the global
generator and `transfer` does not — so `best loss` carries a stream confound.
`shock` and `recovery` are measured per event against the same run's own
preceding loss and are clean.

## What the search is actually worth

`experiments/restart_baseline.py`, 28 seeds, every arm given exactly 6000
gradient steps with the same lr, optimiser and reset draw, all selecting on
validation:

| arm | recovery | median test |
|---|---|---|
| **oracle operators `sin, square` + 8 restarts** | **28/28 (100%)** | **7.66e-15** |
| GAIOptimizer search, 1 trajectory + mutations | 3/28 (11%) | 8.39e-02 |
| random operators + 8 restarts | 0/28 (0%) | 8.12e-02 |

Given the correct operators, exact recovery is a solved problem at this budget —
every seed, machine precision. The search captures 11% of it (p = 1.2e-12
against the ceiling), and **its margin over chance operator assignment is not
statistically established even at n=28** (3/28 vs 0/28, p = 0.24).

The 8-seed cross shows both factors are needed and neither suffices alone:
restarts given correct operators are worth 1/8 -> 8/8 (p = 0.0014), correct
operators given restarts 0/8 -> 8/8 (p = 0.00016), correct operators without
restarts nothing (1/8 vs 0/8, p = 1.0). The forced-operator arm also disposes of
the obvious confound: 750 epochs per restart is not too short, so the random
arm's 0/28 is an operator-selection failure and not a convergence failure.

Splitting the search's own budget does not help: 8 x 750-epoch searches give 1/8
exact recoveries against 3/8 for one long run, while improving the median 4.7x
(1.80e-02 against 8.39e-02). Restarts and operator search compete for the same
epochs — enough to fit weights once operators are known, far too few to find
them.

**Sample-size caution.** The same search configuration measures 3/8 = 38% on
seeds 0-7 and 3/28 = 11% on seeds 0-27. Absolute recovery rates quoted from 8
seeds anywhere in this repository are optimistic; paired comparisons at 8 seeds
are unaffected, because both arms see the same seeds.

## Did the SA changes help? Resolved at 28 seeds

The 8-seed grid below suggested the modified SA hurt recovery in every cell. Run
at 28 seeds, with the rolling stagnation reference finally functioning, and with
a third arm that removes only the homotopy swap:

| GAI-A @6000, 28 seeds | recovery | median test | best |
|---|---|---|---|
| legacy (original) | 3/28 (11%) | 8.387e-02 | 7.647e-16 |
| fixed (homotopy) | 1/28 (4%) | 6.640e-02 | 1.439e-14 |
| fixed-reset (no homotopy) | 1/28 (4%) | **4.629e-02** | 7.963e-16 |

**The homotopy swap is exonerated.** `fixed` and `fixed-reset` are identical on
recovery (1/28 vs 1/28, p = 1.0), so removing the swap restores nothing — the
hypothesis that it deletes the fresh draw recovery depends on is wrong, and the
earlier 8-seed signal that pointed at it was noise. Whatever costs recovery is
in the remaining bundle (revert cooldown, preserved Adam moments,
scale-invariant acceptance, finite tabu), and none of those has been isolated.

**No recovery difference here is significant** (legacy vs either fixed arm,
p = 0.61). The 8-seed grid's apparent "original beats modified in all four
cells" does not survive proper power.

**What the changes DO buy is typical-case accuracy.** Median test loss improves
monotonically, 8.39e-02 -> 6.64e-02 -> 4.63e-02, i.e. **1.81x better** for
`fixed-reset` — measured on the same 28 seeds, so this is a paired improvement,
not a sampling artifact.

That splits the objective in two, and it is worth stating plainly because the
project has been treating them as one thing: **exact symbolic recovery and
typical-case accuracy respond differently to the same interventions.** Splitting
the budget into restarts improves the median 4.7x and *reduces* exact recoveries
(3/8 -> 1/8). The SA changes improve the median 1.81x and leave recovery
statistically unchanged. The only intervention that significantly moved recovery
was reverting the reset draw. Optimise for one and check the other; a change
that improves the median is not thereby an improvement to the method's
headline claim.

## The earlier 8-seed grid (superseded above, kept for the SA-arm comparison)

`experiments/gai_recovery.py --seeds 8 --epochs-list 2000 6000 --arms defaults
GAI-A GAI-B --sa legacy fixed`, on `y = sin(pi*x1) + x2^2` at H=1, run under the
corrected (narrow) reset default. Recovery rate, 8 seeds per cell:

| arm | SA | @2000 | @6000 |
|---|---|---|---|
| defaults | legacy | 0% | 0% |
| defaults | fixed | 0% | 0% |
| GAI-A | **legacy** | **25%** | **38%** |
| GAI-A | fixed | 0% | 12% |
| GAI-B | **legacy** | **25%** | 12% |
| GAI-B | fixed | 0% | 0% |

**The original SA beats my modified SA in all four cells where anything recovers
at all.** That is the honest headline, and it should be read with its statistics:
pooling everything gives 8/32 against 1/32 at p = 0.027, but the two budgets
reuse the same seeds, so that pooling is pseudo-replicated and the p is
inflated. Tested per budget, without pseudo-replication, it is p = 0.10 (@2000)
and p = 0.33 (@6000) — neither significant. A sign test on the consistent
direction across all four cells gives p = 0.0625. So: **the direction is
consistent and the magnitude is not established.**

The `fixed` arm here bundles the homotopy swap, revert cooldown, preserved Adam
moments, scale-invariant acceptance and finite tabu — and, in these runs, a
rolling stagnation reference that was still inert. Which component carries the
effect is untested; the `fixed-reset` arm added to the script isolates homotopy
and needs more than 8 seeds.

Also worth noting: the untuned `defaults` arm recovers 0/8 everywhere, and its
`legacy` and `fixed` cells at 2000 epochs are bit-identical (same median, same
best, same structures) — consistent with no mutation there ever producing a new
best, so both arms restore the same early checkpoint.

## Does the search earn its keep as the space grows? No

`experiments/scaling_regime.py`, teacher-generated targets, 16 seeds, identical
6000-epoch budgets. Targets are identity-padded so the FUNCTION stays easy while
the SEARCH gets hard — 36 assignments at 2 nodes against 1,296 at 4.

| nodes | space | arm | recovery | median test | exact op-set |
|---|---|---|---|---|---|
| 2 | 36 | oracle | 16/16 | 2.262e-15 | 16/16 |
| 2 | 36 | search | 2/16 | 7.509e-03 | 1/16 |
| 2 | 36 | random | 1/16 | 7.397e-03 | 1/16 |
| 4 | 1,296 | oracle | **16/16** | 2.209e-15 | 16/16 |
| 4 | 1,296 | **search** | **0/16** | 3.659e-03 | **0/16** |
| 4 | 1,296 | random | not run (session ended) | | |

At 2 nodes the search beats random by exactly one seed (2/16 vs 1/16,
**p = 1.0**) with a *worse* median. At 4 nodes it finds the correct operator set
**zero times out of sixteen**, in a space where the oracle scores 16/16 — so
this is not a hard-target effect, it is a selection failure. Random landed on
1/16 at 2 nodes against a pre-registered null of 0.9/16, i.e. exactly chance.

The gap between search and chance does not widen as the space grows, which is
the specific thing the method needs to be true. This is the fourth independent
null on operator selection (see the section below for the other three).

Caveat on the `op-set` column in `results/logs/scaling.log`: it was computed
order-sensitively, so `[square, sin]` did not count as correct even though the
chains are summed and it is the same model. Fixed in the script; the stored log
undercounts.

## Nesting, not depth, is what destroys exact recovery

With operators FORCED to ground truth, so selection cannot be blamed
(`experiments/scaling_regime.py`, teacher-generated targets, 8 restarts):

| teacher | nodes | oracle recovery | median test |
|---|---|---|---|
| `sin + square` | 2 | **4/4** | 1.3e-15 |
| identity-padded to depth 2 (same function) | 4 | **4/4** | 3.1e-15 |
| `gaussian(sin(.)) + square` (nested) | 4 | **0/4** | 4.9e-03 |

The nested cell is still 0/4 at 4x the budget, so it is not convergence time.
**Depth costs nothing; nesting costs everything.** I expected depth to be the
culprit and the identity-padded control refuted that — padding a chain to depth
2 leaves recovery at machine precision, while one genuine nested nonlinearity
destroys it even with every operator already correct.

This is the tightest scoping result in the project, because composition is the
method's premise: the `a_of_b` composite operators exist for nested targets, and
nested targets are exactly the ones whose weights cannot be optimised to machine
precision. Exact symbolic recovery lives in one place — a SUM of un-nested basis
operators at width 1.

## Does the search select operators at all? No detectable ability

End-to-end recovery is a conjunction — right operators AND a lucky weight basin
— so it is a rare event and 3/28 vs 0/28 sits at p = 0.24. But the second half
is known to be solved (oracle operators + restarts = 28/28), so the question can
be asked with far more power: take whatever operators the search selects, freeze
them, restart the weights on top. Every correctly-selected seed then *becomes* a
recovery. `experiments/restart_baseline.py --arms search-then`, 28 seeds:

| | result |
|---|---|
| search picks the ground-truth operator set | **3/28 (10.7%)** |
| chance rate for that set (unordered, 2/36) | **5.6%** |
| binomial test against chance | **p = 0.202** |
| search alone | 3/28 |
| **search-then-restart** | **3/28** |
| oracle-then-restart | 28/28 |

**The search's operator selection is not distinguishable from a random draw.**
Adding restarts on top of it changed the result by exactly nothing, because the
operators handed over are usually wrong — the 3 recoveries are precisely the 3
seeds where selection happened to be correct. The distribution of what it picks
makes the point sharper still: its two most frequent choices, `gaussian, sin`
(4/28) and `relu, tanh` (4/28), are each *more* common than the ground truth
`sin, square` (3/28).

This kills the obvious fix. "Append restarts to the search" looked like a large,
one-line win from the restart result (1/8 -> 8/8 given correct operators), and
it does nothing here, because the premise — that the search supplies correct
operators — is false. Operator selection needs replacing, not tuning.

## Code review findings

A read-through of `GAIOptimizer` after the experiments were finished, looking
specifically for the failure mode this session kept hitting: flags and code
paths that look active and are not.

- **Validation was scored in `train()` mode — FIXED.** `fit` called
  `self.model.train()` each epoch and never switched to `eval()` before
  computing the validation score. `torch.no_grad()` disables gradients but not
  dropout, so with dropout enabled the score was noisy AND biased: at
  `dropout=0.3`, five readings of a frozen model spanned 1.2327-1.3092 against a
  true 1.1158, a 13% upward bias. That score drives best-model tracking,
  stagnation detection and mutation accept/reject, so the noise fed straight
  into every structural decision. **No result in this repo is affected** —
  `MatrixGGLEN` defaults to `dropout=0.0` and every experiment used the default,
  where train and eval modes are bit-identical (verified, max difference
  0.000e+00). It was reachable via `use_cases/control/run_ctz_baseline.py`
  (0.01) and `baseline_comparison.py --dropout`. Pinned by
  `test_validation_is_measured_in_eval_mode`.

- **`mutation_mode='transfer'` was documented but never implemented — FIXED.**
  Only `'homotopy'` is branched on, so every other string silently behaved as
  `'reset'`; a caller asking for `'transfer'` got reset semantics and no error.
  The constructor now rejects anything but `'homotopy'`/`'reset'` and points at
  `mutate_reset_weights` for the transfer case. This is the same silent-
  fallthrough class that made both arms of `mutation_mechanics.py` identical.

- **`l1_lambda` penalises `layers[0]` of each chain only — documented, NOT
  changed.** The docstring calls it the "L1 regularization coefficient", which
  implies all weights; at `chain_depth=3` two thirds of the network is
  unregularised. Read as *input-feature* sparsity the implementation is
  defensible. Left alone deliberately because GAI-B and GAI-C both set
  `l1_lambda=1e-05`, so widening it would silently change every result those
  configs produced — measure first.

- **Checked and clean:** every constructor parameter is actually read (`loss_fn`
  is stored as `self.criterion`); the compression phase trains on train and
  selects on validation; `restart_baseline.py` records the test loss only for
  the model already chosen on validation, so the headline comparison has no test
  leakage; the other three `eval()` sites in `trainer.py` were already correct.

### Second pass: the search path

- **The exhaustive search ranks candidates by a MIN-OVER-EPOCHS statistic.**
  `_train` returns the *minimum* validation loss seen across its budget, and
  that value is what `exhaustive_operator_search` sorts `6**n` assignments by.
  This is the same estimator family the project already documents as harmful on
  the restart axis — `probe_reduce`'s own comment says min "systematically
  flatters high-variance operators". The epoch axis was never considered.

  It matters for the headline positive result, not just in principle: on Lorenz
  the correct answer is `square`, and `square`/`gaussian` are the operators
  whose loss oscillates most. A bias toward high-variance candidates points at
  the right answer there **for the wrong reason**, so part of the 2.02x win may
  be the estimator rather than the method.

  Now selectable via `SearchConfig.train_score` (`'best'` = original, `'final'`
  = end-of-budget, unbiased w.r.t. variance).

  **RESOLVED — the win is not the estimator.** Re-running the full Lorenz
  comparison at 5 seeds under both settings gives byte-identical results on
  every arm:

  | arm | `train_score='best'` | `train_score='final'` |
  |---|---|---|
  | mlp-best-act | 1.1925e-03 | 1.1925e-03 |
  | search-greedy | 2.2621e-03 | 2.2621e-03 |
  | search-exhaust | 5.8972e-04 | 5.8972e-04 |

  A null result is only interpretable if the knob works, so that was checked
  separately: on an oscillating run `'best'` returns 4.350743e-01 where
  `'final'` returns 4.812836e-01, i.e. the flag genuinely changes what `_train`
  reports (pinned by `test_train_score_selects_the_ranking_statistic`). So the
  bias is real but inert here — **`square` wins its ranking by a margin wide
  enough that the statistic does not change the pick.** The 2.02x stands.

  The run also reproduces the original numbers independently at 5 seeds rather
  than 3: median 5.8972e-04 against the MLP's 1.1925e-03, seed separation
  intact, and seed 1 recovering `identity -> square -> identity` at 1.3302e-06 —
  the same structure and value the README already quoted.

  **What does NOT survive scrutiny is the budget matching.** The arms are
  matched on restarts, not epochs: exhaustive spends 36,950 training epochs
  against the baseline's 16,000. "2x better for 2.3x the compute" is the honest
  phrasing, and an epoch-matched rerun has not been done.

- **Importance-guided selection searches half the network.** Measured over 36
  evolution events on a 6-node model: importance is not degenerate (max/min
  ratio ~4.1, never all-zero), but proposals concentrate — **78% went to one
  node and only 3 of 6 nodes were ever targeted.** Random selection at least
  covers every node, which is consistent with the measured finding that
  importance ranking performs no better than random.

- **`_neutral_state` uses a Xavier draw** for probe initialisation, while the
  mutation path uses `U(-0.05, 0.05)` — and the narrow draw measured
  dramatically better for recovering `sin` (6/16 against 0/16, p = 0.018). The
  two code paths disagree about something that was measured. Untested in the
  probe; a plausible and cheap improvement to `structure_search` recovery.

- **`evolve_structure(rng=...)` is never passed by the trainer**, so the
  explicit-generator plumbing is dead and mutation draws come from module-level
  `random`. Not a correctness bug (that generator is seeded) but the parameter
  does not do what its presence implies.

- **Verified correct, now pinned by tests:** after a mutation or revert, the
  optimiser owns *exactly* the model's live tensors and a step still moves them.
  This was the highest-risk path in the file — `_revert` and the homotopy branch
  both replace `model.chains` with a `deepcopy`, creating new parameter tensors,
  and a stale optimiser would have turned training into a silent no-op with no
  error and no NaN. Checked by tensor identity, both mutation modes.

- **`structure_search` touches the test split in exactly two places**:
  construction and `finalise`. No leakage anywhere in the phase search.

- **Operator parsing is sound.** Separator precedence is correctly ordered
  (`_plus_` before `_x_` before `_of_`); associativity does not matter because
  all three operations are associative; `X_x_X` is skipped only because
  `square_of_X` already covers it; `to_torch` and `to_sympy` apply soft-clipping
  at identical positions.

## Still to do

Code hygiene:

- `benchmarks/run_benchmark.py` and 12 other files import modules absent from
  the repo (`gai_unified`, `modules.core`, `legacy.gai_final_boss`, `gai_moe`,
  `gai`). They need restoring or deleting.

Results measured but never written down:

- **The SINDy head-to-head is not recorded anywhere in this repo.** It was run
  (`experiments/sota_baselines.py`, 5 tasks) and the outcome was that SINDy won
  4 of 5, three of them by 9-10 orders of magnitude, with the single win for
  this method on `gauss_of_sin` at ~1.1x. Those figures are deliberately NOT
  stated as fact in README.md or the theory notes, because they are being
  recalled rather than read from a surviving log. Re-run and record before
  anyone cites them — and note that SINDy only applies where a sparse
  polynomial/library basis fits, which is a real scope limit on the comparison,
  not a fine-print excuse.
- **The Lorenz 2.02x win over the MLP is a 3-SEED result.** It is the strongest
  positive result in the project — complete seed separation, and it recovers
  `identity -> quadratic -> identity`, which is the mathematically right answer
  because `square` generates the `xy`/`xz` cross terms no global activation can
  express. But 3 seeds is exactly the sample size that produced the 3/8 = 38%
  recovery figure later corrected to 3/28 = 11%. Re-run at >=16 seeds before
  building on it.

Measurements that would change conclusions:

- **Re-measure every absolute rate at >=28 seeds.** The 8-seed figures
  throughout this file and the README are optimistic — the search arm reads 38%
  at 8 seeds and 11% at 28. Paired comparisons are safe; standalone rates are
  not.
- **Does the search beat chance operator assignment at all?** 3/28 against 0/28
  is p = 0.24, still unresolved, and it is the question that decides whether the
  mutation loop is a search. Separating an ~11% rate from ~0% needs on the order
  of 100 seeds per arm, or a task with a higher base recovery rate.
- **Which `fixed` component costs recovery.** The `fixed` arm bundles homotopy,
  revert cooldown, preserved Adam moments, scale-invariant acceptance and finite
  tabu. `--sa fixed-reset` isolates homotopy; the remaining four are untested
  individually, and none has a demonstrated benefit.
- **`budget_scaling.py` has still not been redone against `GAIOptimizer`** — it
  measures the `structure_search` path only.

### Reproducibility: seeding is not sufficient, thread count must match

`seed_everything` seeds `random`, numpy and torch, and that is still not enough
to reproduce a run. Multithreaded BLAS reductions sum in a nondeterministic
order, so the same seed and config give different numbers at different thread
counts — measured on GAI-A recovery, 1500 epochs:

| `OMP_NUM_THREADS` | test loss |
|---|---|
| 1 | 1.179647445679e-01 |
| 3 | 1.179661080241e-01 |
| 4 | 1.179633960128e-01 |

A 6th-significant-figure difference sounds harmless and is not. `GAIOptimizer`
compares scores against a stagnation threshold, so a tie-break landing the other
way shifts *when* evolution fires, and over a few thousand epochs the runs
diverge into different mutation counts outright — the same paired control over
the same 6 seeds recorded 126 mutations at one thread count and 137 at another,
with median gain moving 0.66x to 0.76x. Neither number is wrong; they are
different runs.

Consequences: comparisons made inside one process at one thread count are sound,
and every paired comparison reported here is of that kind. Comparisons quoted
ACROSS separately-launched runs are only valid if `OMP_NUM_THREADS` matched.
`seed_everything(seed, threads=N)` now pins it.

Known-inert or unverified claims that should not be repeated without a re-run:

- Anything quoting mutation counts or payoff rates recovered from loss curves
  rather than `GAIOptimizer.mutation_log`.
- Any result from a non-legacy arm produced before the rolling-reference repair,
  since that arm was running the legacy stagnation signal regardless of setting.
