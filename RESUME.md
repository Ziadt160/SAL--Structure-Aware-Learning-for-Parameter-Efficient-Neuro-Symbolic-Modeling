# Checkpoint — 2026-07-28 ~01:40 EDT

Written before a laptop shutdown. Everything below is either already on disk or
tells you exactly how to regenerate it.

## Read this first

**Raw logs for every experiment run in this session are preserved in
`results/logs/`.** They were in `/tmp`, which on this machine resolves to
`C:/Users/20102/AppData/Local/Temp` and is cleared by Windows Disk Cleanup and
some updates. If you ever need to re-check a number against its source, it is in
that directory, not in `/tmp`.

| log | what it is |
|---|---|
| `recov2.log` | recovery grid, 3 arms x 2 SA x 2 budgets x 8 seeds |
| `restart28.log` | **the headline**: search vs restarts vs oracle, 28 seeds |
| `restart.log`, `restart_k1.log` | K=8 and K=1 restart cross |
| `searchthen.log` | search-then-restart, operator-selection quality, 28 seeds |
| `searchx.log` | search run as 8 restarts |
| `homotopy28.log` | legacy vs fixed vs fixed-reset, 28 seeds |
| `mutgain3.log`, `mutgain4.log` | paired mutate/frozen control |
| `mech3.log` | reset vs transfer mechanics |
| `ab_xav.log`, `gate4_big.log` | reset-scale ablation, 4-node budget probe |
| `scaling.log` | scaling regime — **INCOMPLETE, see below** |

## State of the code

All committed changes are in the working tree (nothing is staged or committed —
`git status` will show them). **62 invariants pass**:

```bash
python tests/test_invariants.py
```

Run that first after restarting. If it passes, the code is in the state this
session left it.

## The one job that did NOT finish

`experiments/scaling_regime.py` — 4 of 6 cells complete. Results so far, from
`results/logs/scaling.log`:

| nodes | space | arm | recovery | median test | op-set |
|---|---|---|---|---|---|
| 2 | 36 | oracle | 16/16 | 2.262e-15 | 16/16 |
| 2 | 36 | search | 2/16 | 7.509e-03 | 1/16 |
| 2 | 36 | random | 1/16 | 7.397e-03 | 1/16 |
| 4 | 1,296 | oracle | **16/16** | 2.209e-15 | 16/16 |
| 4 | 1,296 | **search** | **0/16** | 3.659e-03 | **0/16** |
| 4 | 1,296 | random | **NOT RUN** — process stopped here | | |

**Five of six cells are done.** Only the 4-node `random` control is missing, and
it is the cheap one (no mutation machinery, just 8 x 750-epoch fits per seed).
One command finishes the study:

```bash
OMP_NUM_THREADS=6 python experiments/scaling_regime.py --nodes 4 --seeds 16 --epochs 6000 --k 8 --arms random
```

The pre-registered null prediction for that cell is **~0.1/16 recovery and
~0.15/16 exact op-sets** — i.e. expect 0/16. If it comes back 0/16, the search
matched chance at 1,296 assignments having also matched it at 36, which is the
fifth independent null and closes the question.

Note the 4-node search result is already the informative half: **0/16 recovery
and 0/16 correct operator sets** in a space where the oracle scores 16/16. The
search found the right operators zero times out of sixteen.

Two notes on interpreting it when it lands. The `op-set` column in
`scaling.log` UNDERCOUNTS — it used an order-sensitive comparison, so
`[square, sin]` did not count as correct even though the chains are summed and
it is the same model. That is fixed in the script now, so a fresh run reports it
correctly; the old log's column is wrong. And the null predictions were
registered in advance: if the search has no selection ability, expect random
~0.1/16 recovery at 4 nodes and ~0.15/16 exact op-sets.

## What was established (all of this is done, logged, and written up)

Full detail is in `CHANGES.md`, `README.md` and `theory/structural_learning.md`.
The short version:

1. **The search's operator selection is indistinguishable from chance.** Four
   independent tests: 3/28 vs 0/28 (p=0.24); op-set 10.7% vs 5.6% chance
   (p=0.20); search-then-restart 3/28, identical to search alone; 2/16 vs 1/16
   at 2 nodes (p=1.0).
2. **Given correct operators, recovery is solved** — 28/28 with restarts, and
   16/16 even in the 1,296-assignment space. Operator selection is the whole
   bottleneck.
3. **Nesting, not depth, destroys exact recovery.** Identity-padded depth-2:
   4/4 at 3.1e-15. One nested nonlinearity (`gaussian(sin(.))`): 0/4, and still
   0/4 at 4x budget.
4. **Not mutating beats mutating** on the median in every version of the paired
   control (best case 0.86x).
5. Several of my own "fixes" were regressions or inert and have been reverted or
   repaired — the Xavier reset, the rolling stagnation reference. See CHANGES.md.

## Still open (tasks, in priority order)

1. Finish the two `scaling_regime` cells above.
2. **Re-run the Lorenz MLP head-to-head at >=16 seeds.** The 2.02x win is the
   project's strongest positive result and currently rests on 3 seeds — the same
   sample size that gave 3/8=38% for a quantity whose true value is 3/28=11%.
   `python experiments/tuned_rerun.py --seeds 16 --restarts 5`
3. **Re-run and RECORD the SINDy head-to-head.** It was measured this session
   (SINDy won 4/5, three by 9-10 orders of magnitude) but never written to disk,
   so those figures are recalled, not sourced. `experiments/sota_baselines.py`

## Two environment facts that will bite otherwise

- **Use the right interpreter.** The default `python` has no torch. Use
  `C:/Users/20102/miniconda3/envs/bacteria_gpu/python.exe`.
- **Seeding is not enough for reproducibility.** Results differ by
  `OMP_NUM_THREADS` because BLAS reductions are order-dependent — measured
  1.179647e-01 / 1.179661e-01 / 1.179634e-01 at 1/3/4 threads on an identical
  seed. It compounds through the stagnation threshold into different mutation
  counts (126 vs 137 on the same 6 seeds). **Set `OMP_NUM_THREADS` identically
  for any runs you intend to compare.**

## Your own sweeps — ACTION NEEDED BEFORE SHUTDOWN

Six processes that are NOT from this session were running:

```
run_hard_benchmark      x4   (started 22:23 x3, 00:56 x1)   ~5.6 CPU-hours each on the oldest
run_scaling_advantage   x2   (started 23:25, 01:24)
```

I could not find any `.csv` output on disk for them, and they take `--out`
filenames (`hard_flat_K34_Thigh.csv`, `hard_benchmark_flat_s44.csv`,
`_smoke4.csv`, ...) which suggests they write **at completion, not
incrementally**. If that is right, shutting down loses all of their work.

I did not touch them — they are yours. Check whether they have written anything
before you shut down.
