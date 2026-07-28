"""
Structure-Aware search: the four phases, implemented so they are repeatable.

The algorithm, in the order the phases run:

  Phase 0  WARM UP      Train weights on the fixed initial structure.
                        Structural search needs a roughly stationary objective;
                        comparing candidates while the whole network is still
                        far from convergence measures training noise.

  Phase 1  OPERATORS    Greedy coordinate descent over per-node operators.
                        Every node is visited every sweep. For each node,
                        every candidate operator is probed from the SAME
                        starting weights with the SAME fine-tune budget, and
                        the best is kept. Optionally a second tier probes
                        composites built from the winner.

  Phase 2  TOPOLOGY     Only entered when Phase 1 improved something.
                        Tries grow/shrink moves (add expert, deepen a chain,
                        prune an expert, shorten a chain). Growth moves are
                        function-preserving, so the model starts each move
                        computing exactly what it did before and the move is
                        judged purely on what training does with the capacity.
                        Any accepted move sends us back to Phase 1.

  Phase 3  STABILITY    Structure is stable when an operator sweep and a
                        topology round both accept nothing.

  Phase 4  COMPRESS     Shrink hidden_dim while val loss stays within
                        tolerance. Every width -- INCLUDING the incumbent's
                        own width -- is retrained from scratch for the same
                        number of epochs, so the comparison is equal-budget.

Everything is scored on a validation split. The test split is touched exactly
once, at the very end, in `finalise`.

Why greedy enumeration rather than annealed random proposals: the per-node
operator space has ~6 basis members. You can afford to try them all. Random
single-proposal search over a space that small mostly measures which
reinitialisation got luckier, and it leaves most nodes untouched -- so the
"discovered" architecture is largely whatever the initialiser picked.
"""

import copy
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.activations import (SEARCH_BASIS, basis_candidates,
                                composite_candidates, canonical_class)


# --------------------------------------------------------------------------
@dataclass
class SearchConfig:
    # Phase 0
    warmup_epochs: int = 400
    # Phase 1
    # 'greedy'     -- coordinate descent, one node at a time. Scales to any
    #                 number of nodes but can stall when two nodes must change
    #                 together to reach the optimum.
    # 'exhaustive' -- evaluate every operator assignment. Guaranteed optimal
    #                 for the basis, and structurally seed-stable, but the
    #                 space is len(basis) ** n_nodes so it only applies to
    #                 small models (an MLP -- num_chains=1 -- is 6**depth).
    search_mode: str = 'greedy'
    exhaustive_max_configs: int = 512
    exhaustive_screen_epochs: int = 60
    exhaustive_verify_top: int = 8
    exhaustive_refine_composites: bool = True
    # Measured on Lorenz by experiments/probe_fidelity.py, over 6 nodes x 2
    # seeds. "spearman" is agreement between the probe's ranking and the
    # full-training ranking; "regret" is how much worse the operator the probe
    # actually PICKS is than the true best, which is the number that matters:
    #
    #   probe@40,  r=1   spearman +0.35   regret 1.30x     240 epochs/node
    #   probe@40,  r=3   spearman +0.38   regret 1.26x     720
    #   probe@400, r=1   spearman +0.58   regret 1.21x    2400
    #   probe@400, r=3   spearman +0.52   regret 1.28x    7200
    #
    # Two things follow. No setting is trustworthy -- the best rank agreement
    # is 0.58. And a 10x budget increase buys only a 7% better pick, so the
    # probe is NOT the dominant error source: even a perfect probe recovers
    # about 25%, which cannot close the ~50% gap to a plain MLP baseline on
    # this task. Do not reach for a bigger probe expecting it to fix the search.
    #
    # 400 is kept as the quality default because it is best on both metrics;
    # probe@40 with r=3 is the better value at roughly a third of the cost.
    probe_epochs: int = 400
    # Number of independent initialisations per candidate. Extra restarts did
    # not help rank agreement (probe@400: +0.58 at r=1 against +0.52 at r=3)
    # -- see the reduction bias described under probe_reduce. At the cheap end
    # r=3 does improve the pick slightly (regret 1.26x against 1.30x) for 3x
    # the cost, which is the one place restarts earn anything.
    probe_restarts: int = 1
    # Where a candidate probe starts from. Both options are biased, in opposite
    # directions:
    #   'current' keeps the node's trained weights -- but those are already
    #             adapted to the operator being replaced, so the incumbent gets
    #             a home-field advantage and a better operator can lose for lack
    #             of catch-up time.
    #   'reset'   starts every candidate from the same fresh weights -- fair
    #             between candidates, but discards learned weights, so every
    #             candidate must re-learn inside the probe.
    #   'both'    probes from both and keeps each candidate's better score.
    # 'reset' measured best on the metric that matters: at probe@400 both start
    # modes tie on rank agreement (+0.58) but 'reset' picks far better
    # operators (regret 1.21x against 1.60x), because including the incumbent's
    # own weights as a starting point reintroduces its home-field advantage.
    probe_from: str = 'reset'
    # How to reduce a candidate's scores across its restarts into one number.
    #
    # 'min' is the intuitive choice and it is WRONG for ranking. The minimum of
    # R draws is a biased estimator whose bias grows with the candidate's
    # variance across initialisations, so it systematically flatters
    # high-variance operators (`sin` most of all, being multi-modal in w). The
    # bias cancels only if every candidate has the same spread, which they do
    # not. Measured on Lorenz: at probe@400, min-over-3-restarts scored
    # spearman +0.52 against full training while a single restart scored +0.58.
    #
    # 'median' estimates typical performance, which is the quantity a ranking
    # should be built on. Use 'min' only when the deployed model will itself be
    # restart-selected, so that "best achievable" is the honest target.
    probe_reduce: str = 'median'
    max_op_sweeps: int = 4
    use_composites: bool = True
    basis: Sequence[str] = tuple(SEARCH_BASIS)
    # WHICH nodes get searched, when there is not budget for all of them.
    #
    #   'all'            visit every node every sweep (ignores node_budget)
    #   'taylor_low'     lowest  mean |dL/dz * z| first  -- "repair the weakest"
    #   'taylor_high'    highest first                   -- "most leverage"
    #   'gradnorm_low'   lowest  mean ||dL/dz|| first    -- the original metric
    #   'gradnorm_high'  highest first
    #   'random'         shuffle
    #
    # This only has an effect when node_budget > 0. With exhaustive search, or
    # with greedy visiting every node, ranking is irrelevant because everything
    # gets searched anyway -- ranking matters exactly in the regime where
    # enumeration is unaffordable, which is the regime that matters for scaling.
    node_order: str = 'all'
    node_budget: int = 0            # 0 = no limit
    max_op_sweeps_note: str = ''    # free-text label for experiment logs
    # Phase 2
    topology_rounds: int = 2
    # Whether a topology round requires Phase 1 to have improved something
    # first. True matches the algorithm as originally specified -- "when we
    # find a mutation that increases the prediction accuracy we mutate the
    # architecture" -- and it is the default so the implementation stays
    # faithful to that.
    #
    # It has a measurable cost. On the analytic task, a run whose operator
    # sweep found no improvement skipped Phase 2 entirely and finished at test
    # 2.0e-02; letting topology run anyway reached 6.9e-04, roughly 29x better,
    # because add_chain and deepen_chain were where nearly all the gain was.
    # Set False to decouple them.
    topology_requires_op_gain: bool = True
    max_chains: int = 6
    max_depth: int = 5
    allow_growth: bool = True
    allow_pruning: bool = True
    # Phase 1/2 shared
    consolidate_epochs: int = 250
    # Phase 3.5 -- retrain the CHOSEN architecture this many times from scratch
    # and keep the best on validation.
    #
    # The architecture search and the weight optimisation fail independently,
    # and this is where the second one gets fixed. On Lorenz, exhaustive search
    # picks identity -> quadratic -> identity in 2 of 3 seeds (the right
    # structure for a polynomial system: square generates the xy and xz terms
    # via (a+b)^2), yet the weights exploit it in only 1 of 3, giving a bimodal
    # outcome -- 7.4e-05 on the lucky seed against ~2e-03 on the others.
    #
    # Restarts belong HERE and not in the candidate probe. Inside the probe they
    # bias the ranking (see probe_reduce) because different operators have
    # different spreads. After the structure is fixed there is nothing left to
    # rank, so extra draws are just independent tickets in the weight lottery.
    #
    # Any baseline compared against this must get the same restart budget.
    final_restarts: int = 1
    # L-BFGS iterations applied to the chosen architecture after Adam.
    #
    # Once the structure is fixed this is a smooth nonlinear least-squares
    # problem, and Adam is the wrong tool for the last few orders of magnitude:
    # a first-order method with an adaptive but noisy step plateaus around
    # 1e-5..1e-7, while a quasi-Newton method with a strong-Wolfe line search
    # drives the same objective to 1e-12 and below. Adam-then-LBFGS is the
    # standard recipe in scientific ML for exactly this reason.
    #
    # It matters here because the gap to a direct linear solve was assumed to be
    # structural, and it is not: for a target like Lorenz the architecture can
    # represent the answer EXACTLY (signed combinations of squared linear forms
    # span all quadratic polynomials), so landing 9 orders above it is an
    # optimisation failure, not a representational one.
    #
    # 0 disables. Applied only after selection, never inside a probe.
    lbfgs_iters: int = 0
    # Phase 4
    compress: bool = True
    compress_epochs: int = 250
    # Stage 1 is only a filter deciding which widths are worth a full retrain,
    # so it is generous. Stage 2 compares against the incumbent and is the
    # bar that actually decides what ships.
    compress_shortlist_tol: float = 0.5
    compress_accept_tol: float = 0.05
    compress_widths: Sequence[int] = (48, 32, 24, 16, 12, 8, 6, 4)
    compress_verify_top: int = 3
    # optimisation
    lr: float = 0.01
    probe_lr: float = 0.02
    min_delta: float = 1e-3         # relative improvement needed to accept
    seed: int = 0
    verbose: bool = True


@dataclass
class SearchTrace:
    """Everything needed to answer 'did the search actually search?'."""
    seed: int = 0
    init_structure: List[List[str]] = field(default_factory=list)
    final_structure: List[List[str]] = field(default_factory=list)
    nodes_probed: List[Tuple[int, int]] = field(default_factory=list)
    nodes_changed: List[Tuple[int, int]] = field(default_factory=list)
    evaluations: int = 0
    train_epochs: int = 0     # total gradient steps, for compute-matched arms
    op_moves: List[Dict[str, Any]] = field(default_factory=list)
    topology_moves: List[Dict[str, Any]] = field(default_factory=list)
    phase_log: List[Dict[str, Any]] = field(default_factory=list)
    importance_snapshot: Dict[str, float] = field(default_factory=dict)
    val_loss: float = float('nan')
    test_loss: float = float('nan')
    params: int = 0
    hidden_dim: int = 0
    wall_seconds: float = 0.0
    sweeps_run: int = 0

    def nodes_holding_init_op(self) -> int:
        """How many nodes still carry the operator the initialiser gave them.

        The key diagnostic for seed stability. If this is most of the network,
        the reported architecture is mostly a report of `random.choice` and
        cannot agree across seeds no matter how good the search rule is.
        """
        n = 0
        for c, (init_ops, final_ops) in enumerate(
                zip(self.init_structure, self.final_structure)):
            for i in range(min(len(init_ops), len(final_ops))):
                if init_ops[i] == final_ops[i]:
                    n += 1
        return n

    def coverage(self) -> float:
        total = sum(len(ops) for ops in self.final_structure)
        return len(set(self.nodes_probed)) / total if total else 0.0

    def summary(self) -> str:
        return (f"seed={self.seed} val={self.val_loss:.6g} test={self.test_loss:.6g} "
                f"params={self.params} H={self.hidden_dim} "
                f"evals={self.evaluations} coverage={self.coverage():.0%} "
                f"op_moves={len(self.op_moves)} topo_moves={len(self.topology_moves)} "
                f"held_init={self.nodes_holding_init_op()} "
                f"struct={self.final_structure}")


def seed_everything(seed: int, threads: Optional[int] = None):
    """Seed every generator that affects a run.

    `random` matters most: it drives operator choices and initial structure.
    Seeding only torch and numpy -- the previous state of the experiment
    scripts -- leaves the architecture search itself unseeded.

    SEEDING ALONE DOES NOT MAKE A RUN REPRODUCIBLE HERE. Multithreaded BLAS
    reductions sum in a nondeterministic order, so the same seed gives different
    numbers at different thread counts. Measured on GAI-A / recovery, identical
    seed and config, 1500 epochs:

        OMP_NUM_THREADS=1   test 1.179647445679e-01
        OMP_NUM_THREADS=3   test 1.179661080241e-01
        OMP_NUM_THREADS=4   test 1.179633960128e-01

    That is a 6th-significant-figure difference, which sounds harmless and is
    not: `GAIOptimizer` compares scores against a stagnation threshold, so a
    tie-break that lands differently shifts WHEN evolution fires, and over a few
    thousand epochs the runs diverge into different mutation counts entirely
    (126 vs 137 across the same 6 seeds). Comparisons made inside one process at
    one thread count are unaffected; comparisons made ACROSS runs are only
    reproducible if the thread count matches.

    Pass `threads` to pin it, or set OMP_NUM_THREADS identically for every run
    that will be compared.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if threads is not None:
        torch.set_num_threads(threads)


# --------------------------------------------------------------------------
class StructureSearch:
    def __init__(self,
                 model: nn.Module,
                 X_train, y_train, X_val, y_val,
                 X_test=None, y_test=None,
                 config: Optional[SearchConfig] = None,
                 loss_fn: Optional[Callable] = None,
                 device: str = 'cpu'):
        self.cfg = config or SearchConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = loss_fn or nn.MSELoss()
        self.rng = random.Random(self.cfg.seed)

        self.Xtr, self.ytr = self._prep(X_train, y_train)
        self.Xva, self.yva = self._prep(X_val, y_val)
        self.Xte, self.yte = ((None, None) if X_test is None
                              else self._prep(X_test, y_test))

        self.trace = SearchTrace(seed=self.cfg.seed)
        self._probed: set = set()
        self._changed: set = set()
        # Importance is reported, not used for decisions -- keep the hooks off
        # during the thousands of probe steps and enable them once at the end.
        if hasattr(self.model, 'set_tracking'):
            self.model.set_tracking(False)

    # -- data -------------------------------------------------------------
    def _prep(self, X, y):
        # Follow the ambient default dtype rather than forcing float32, so a
        # float64 run really is float64 end to end. Hardcoding float32 here fed
        # single-precision inputs to double-precision weights and raised
        # "mat1 and mat2 must have the same dtype".
        dt = torch.get_default_dtype()
        Xt = torch.as_tensor(np.asarray(X), dtype=dt, device=self.device)
        y_np = np.asarray(y)
        if np.issubdtype(y_np.dtype, np.integer):
            yt = torch.as_tensor(y_np, dtype=torch.long, device=self.device)
        else:
            yt = torch.as_tensor(y_np, dtype=dt, device=self.device)
            if yt.ndim == 1:
                yt = yt.unsqueeze(1)
        return Xt, yt

    # -- primitives -------------------------------------------------------
    def _loss(self, model: nn.Module, X, y) -> float:
        model.eval()
        with torch.no_grad():
            return float(self.criterion(model(X), y).item())

    def val_loss(self, model: Optional[nn.Module] = None) -> float:
        return self._loss(model or self.model, self.Xva, self.yva)

    def _train(self, model: nn.Module, epochs: int, lr: float,
               params: Optional[List[nn.Parameter]] = None) -> float:
        """Train for a fixed number of full-batch steps; return best val loss.

        Fixed budget, not early stopping: candidates must be compared under
        identical conditions or the comparison encodes the stopping rule.
        """
        if epochs <= 0:
            return self.val_loss(model)
        target = params if params is not None else list(model.parameters())
        target = [p for p in target if p.requires_grad]
        if not target:
            return self.val_loss(model)

        opt = optim.Adam(target, lr=lr)
        best = self.val_loss(model)
        self.trace.train_epochs += epochs
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            loss = self.criterion(model(self.Xtr), self.ytr)
            # A candidate can legitimately have no parameter in the graph -- a
            # quantum circuit made only of H/CNOT/CZ/I has nothing to train, and
            # exhaustive enumeration will propose exactly those. Score it and
            # move on rather than failing on a missing grad_fn.
            if not loss.requires_grad:
                return self.val_loss(model)
            loss.backward()
            opt.step()
            v = self.val_loss(model)
            if v < best:
                best = v
        return best

    def _better(self, new: float, ref: float) -> bool:
        """Is `new` a real improvement over `ref`? Sign-safe relative test."""
        return new < ref - self.cfg.min_delta * abs(ref)

    def _log(self, msg: str):
        if self.cfg.verbose:
            print(msg, flush=True)

    # -- Phase 0 ----------------------------------------------------------
    def warmup(self) -> float:
        self.trace.init_structure = self.model.get_structure()
        v = self._train(self.model, self.cfg.warmup_epochs, self.cfg.lr)
        self._log(f"[Phase 0] warmup: val={v:.6g} "
                  f"struct={self.trace.init_structure}")
        self.trace.phase_log.append({'phase': 'warmup', 'val': v})
        return v

    # -- Phase 1 ----------------------------------------------------------
    def _probe_starts(self, node, c: int, l: int, sweep: int,
                      current: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Starting states for a candidate probe, per cfg.probe_from."""
        mode = self.cfg.probe_from
        starts: List[Dict[str, Any]] = []
        if mode in ('current', 'both') and current is not None:
            starts.append(current)
        if mode in ('reset', 'both'):
            for r in range(max(1, self.cfg.probe_restarts)):
                starts.append(self._neutral_state(node, c, l, sweep * 97 + r))
        return starts or [self._neutral_state(node, c, l, sweep)]

    def _probe_best(self, node, op: str, c: int, l: int, sweep: int,
                    adapter_original: List[torch.Tensor],
                    current: Optional[Dict[str, Any]] = None
                    ) -> Tuple[float, Dict[str, Any], List[torch.Tensor]]:
        """Probe one operator from every configured start.

        Returns (score, weights, adapter) where `score` is reduced across
        restarts per cfg.probe_reduce but the weights returned are always the
        best run's -- the score decides the ranking, the weights are what we
        would keep if this candidate wins.
        """
        runs = []
        for start in self._probe_starts(node, c, l, sweep, current):
            runs.append(self._probe(node, op, start, adapter_original))

        scores = [r[0] for r in runs]
        finite = [s for s in scores if np.isfinite(s)]
        if not finite:
            return float('inf'), runs[0][1], runs[0][2]

        if self.cfg.probe_reduce == 'min':
            score = min(finite)
        elif self.cfg.probe_reduce == 'mean':
            score = float(np.mean(finite))
        else:
            score = float(np.median(finite))

        best_run = min(runs, key=lambda r: r[0])
        return score, best_run[1], best_run[2]

    def _neutral_state(self, node, c: int, l: int, sweep: int) -> Dict[str, Any]:
        """A deterministic fresh init for this node, identical per candidate.

        Every candidate operator -- including the one currently installed --
        is probed from THIS state. Probing from the node's trained weights
        instead would hand the incumbent a home-field advantage: its weights
        are already adapted to its own operator, so a better operator can lose
        simply because it has less time to catch up. Seeding from
        (seed, sweep, c, l) keeps it reproducible.
        """
        g = torch.Generator(device='cpu')
        g.manual_seed(hash((self.cfg.seed, sweep, c, l)) & 0x7FFFFFFF)
        fan_in = node.linear.in_features
        fan_out = node.linear.out_features
        bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
        w = torch.empty(fan_out, fan_in).uniform_(-bound, bound, generator=g)
        return {'op_name': node.op_name,
                'weight': w.to(self.device),
                'bias': torch.zeros(fan_out, device=self.device)}

    def _adapter_params(self) -> List[nn.Parameter]:
        """The read-out (and gate) -- trainable during a probe.

        These must be free or the comparison is rigged. A frozen read-out is
        still scaled for the operator being replaced, so any candidate that
        changes the output's scale or offset scores badly no matter how good
        it is. On the recovery target that alone was enough to hide the true
        answer: with the read-out frozen the search stalled on `sin,sin`
        (test 2.9e-4) instead of reaching `sin,square` (test 1.3e-14).
        """
        ps: List[nn.Parameter] = []
        if hasattr(self.model, 'final'):
            ps += list(self.model.final.parameters())
        if hasattr(self.model, 'gate'):
            ps += list(self.model.gate.parameters())
        return ps

    def _probe(self, node, op: str, start: Dict[str, Any],
               adapter_start: List[torch.Tensor]
               ) -> Tuple[float, Dict[str, Any], List[torch.Tensor]]:
        """Score one candidate operator for one node.

        Every candidate starts from exactly the same node weights AND the same
        read-out weights, trains the node plus the read-out for the same number
        of steps, and is scored on val. Both are restored afterwards so no
        candidate contaminates the next.
        """
        adapter = self._adapter_params()
        node.restore(start)
        with torch.no_grad():
            for p, saved in zip(adapter, adapter_start):
                p.copy_(saved)
        try:
            node.set_op(op)
        except ValueError:
            return float('inf'), start, adapter_start

        frozen = [(p, p.requires_grad) for p in self.model.parameters()]
        for p, _ in frozen:
            p.requires_grad_(False)
        trainable = list(node.parameters()) + adapter
        for p in trainable:
            p.requires_grad_(True)

        self._train(self.model, self.cfg.probe_epochs, self.cfg.probe_lr,
                    params=trainable)
        score = self.val_loss()
        probed = node.snapshot()
        probed_adapter = [p.detach().clone() for p in adapter]

        for p, flag in frozen:
            p.requires_grad_(flag)
        self.trace.evaluations += 1
        return score, probed, probed_adapter

    def _search_node(self, c: int, l: int, node, sweep: int) -> bool:
        """Find the best operator for one node. Returns True if it changed."""
        self._probed.add((c, l))
        current_op = node.op_name
        original = node.snapshot()
        adapter = self._adapter_params()
        adapter_original = [p.detach().clone() for p in adapter]

        # Tier 1: every basis operator plus whatever is currently installed,
        # all from the same neutral inits and the same budget.
        tier1 = basis_candidates(list(self.cfg.basis))
        if current_op not in tier1:
            tier1 = tier1 + [current_op]
        results: List[Tuple[float, str, Dict[str, Any], List[torch.Tensor]]] = []
        for op in tier1:
            score, snap, ad = self._probe_best(node, op, c, l, sweep,
                                               adapter_original, original)
            results.append((score, op, snap, ad))

        # Tier 2: composites built from the tier-1 winner -- the "activations
        # composed from basis functions" idea. A deterministic, enumerated set
        # rather than a 5%-probability random discovery, so runs repeat.
        if self.cfg.use_composites and results:
            anchor = min(results, key=lambda r: r[0])[1]
            for op in composite_candidates(anchor, list(self.cfg.basis)):
                score, snap, ad = self._probe_best(node, op, c, l, sweep,
                                                   adapter_original, original)
                results.append((score, op, snap, ad))

        best_score, best_op, best_snap, best_adapter = min(results,
                                                           key=lambda r: r[0])
        # The bar is the CURRENT operator measured under identical conditions,
        # not the incumbent's fully-trained loss.
        incumbent_score = min(s for s, op, _, _ in results if op == current_op)

        if best_op != current_op and self._better(best_score, incumbent_score):
            node.restore(best_snap)
            with torch.no_grad():
                for p, saved in zip(adapter, best_adapter):
                    p.copy_(saved)
            self._changed.add((c, l))
            self.trace.op_moves.append({
                'chain': c, 'layer': l, 'from': current_op, 'to': best_op,
                'probe_before': incumbent_score, 'probe_after': best_score,
            })
            self._log(f"    node({c},{l}): {current_op} -> {best_op}  "
                      f"probe {incumbent_score:.6g} -> {best_score:.6g}")
            return True

        node.restore(original)   # nothing better: undo all probing
        with torch.no_grad():
            for p, saved in zip(adapter, adapter_original):
                p.copy_(saved)
        self._log(f"    node({c},{l}): keep {current_op} "
                  f"(best alt {best_op} @ {best_score:.6g} "
                  f"vs {incumbent_score:.6g})")
        return False

    def _rank_nodes(self, sweep: int) -> List[Tuple[int, int, Any]]:
        """Order nodes for this sweep, and trim to node_budget.

        Importance is measured fresh each sweep, because it changes as the
        network trains -- a node that was carrying little load early may be
        load-bearing later.
        """
        nodes = list(self.model.iter_nodes())
        order = self.cfg.node_order
        budget = self.cfg.node_budget or len(nodes)

        if order == 'all':
            chosen = nodes
        elif order == 'random':
            chosen = list(nodes)
            random.Random(self.cfg.seed * 7717 + sweep).shuffle(chosen)
        else:
            imp = self._measure_importance()
            use_gradnorm = order.startswith('gradnorm')
            def score(c, l):
                k = f"{c},{l}|gradnorm" if use_gradnorm else f"{c},{l}"
                return imp.get(k, 0.0)
            chosen = sorted(nodes, key=lambda t: score(t[0], t[1]),
                            reverse=order.endswith('_high'))
            self.trace.phase_log.append({
                'phase': f'rank{sweep}', 'order': order,
                'ranking': [(c, l, score(c, l)) for c, l, _ in chosen],
            })

        chosen = chosen[:budget]
        if self.cfg.node_budget:
            self._log(f"    [{order}] searching {len(chosen)}/{len(nodes)} nodes: "
                      f"{[(c, l) for c, l, _ in chosen]}")
        return chosen

    def operator_sweep(self, incumbent: float, sweep: int) -> Tuple[float, bool]:
        """One pass over the nodes selected by cfg.node_order / node_budget.

        The default ('all', budget 0) visits every node every sweep, which is
        affordable at MLP depth and removes node selection as a variable. Set
        node_budget to study which nodes are worth searching when you cannot
        afford all of them -- see experiments/strategy_comparison.py.

        The whole sweep is reverted if consolidation cannot recover the val
        loss it started from -- probe scores are a proxy, so this is the guard
        that keeps a proxy-driven decision from degrading the model.
        """
        before_model = copy.deepcopy(self.model)
        before = incumbent
        any_changed = False
        for c, l, node in self._rank_nodes(sweep):
            if self._search_node(c, l, node, sweep):
                any_changed = True

        if not any_changed:
            return incumbent, False

        after = self._train(self.model, self.cfg.consolidate_epochs, self.cfg.lr)
        if after > before + self.cfg.min_delta * abs(before):
            self._log(f"    sweep reverted: consolidated val {after:.6g} "
                      f"did not recover {before:.6g}")
            self.model = before_model.to(self.device)
            for move in list(self.trace.op_moves):
                if move.get('sweep') is None:
                    move['sweep'] = sweep
                    move['reverted'] = True
            return before, False

        self._log(f"    sweep accepted: val {before:.6g} -> {after:.6g}")
        return after, True

    # -- Phase 1, exhaustive variant --------------------------------------
    def _flat_nodes(self) -> List[Tuple[int, int, Any]]:
        return list(self.model.iter_nodes())

    def _apply_flat_ops(self, ops: Sequence[str]):
        for (c, l, node), op in zip(self._flat_nodes(), ops):
            node.set_op(op)
            self._probed.add((c, l))

    def exhaustive_operator_search(self, incumbent: float) -> Tuple[float, bool]:
        """Evaluate EVERY operator assignment; keep the best.

        Only viable when the space is small, which for a per-layer operator
        choice it usually is: an MLP (num_chains=1) of depth d has len(basis)**d
        assignments -- 216 for six operators and three layers.

        This removes the failure mode greedy coordinate descent cannot escape.
        Coordinate descent measures a node's marginal value given the others,
        so when two nodes must change together it sees no improvement and
        stops. On the recovery target that is the difference between stalling
        at sin+sin (test ~2e-4) and reaching sin+square (test ~1e-14).

        Two stages: screen every assignment cheaply, then retrain the best few
        at full budget. Every assignment starts from the SAME weights, so the
        comparison isolates the operators.
        """
        nodes = self._flat_nodes()
        basis = list(self.cfg.basis)
        total = len(basis) ** len(nodes)
        if total > self.cfg.exhaustive_max_configs:
            self._log(f"[Phase 1] exhaustive space is {total} configurations "
                      f"(> {self.cfg.exhaustive_max_configs}); "
                      f"falling back to greedy")
            return self.operator_sweep(incumbent, 0)

        # Every assignment is trained from fresh weights, so this search cannot
        # see -- and would otherwise discard -- whatever the incoming model had
        # already learned. That matters on a re-run after a topology move: the
        # move leaves a trained model behind, and re-enumerating from scratch
        # measured 14x worse on the analytic task while still being installed.
        incoming_model = copy.deepcopy(self.model)
        incoming_val = incumbent

        # R shared starting points. EVERY assignment is trained from every one
        # of them and scored on its best, so the comparison stays controlled
        # while no assignment is condemned by a single unlucky initialisation.
        #
        # This matters more than it looks. With the operators forced to the
        # known-correct answer on the recovery target, only 2 of 12 inits reach
        # the exact solution -- the other 10 plateau, and they plateau at the
        # same value the best operator assignment reaches. A single-init search
        # is therefore measuring weight-optimisation luck, not operators.
        # `sin` suffers most: sin(pi*(w.x+b)) is periodic in w, so gradient
        # descent from a random w settles into a wrong frequency basin.
        restarts = max(1, self.cfg.probe_restarts)
        inits = []
        for r in range(restarts):
            seed_everything(self.cfg.seed * 7919 + r)
            fresh = type(self.model)(**{**self.model.hparams_dict,
                                        'fixed_structure': self.model.get_structure()})
            inits.append(copy.deepcopy(fresh.state_dict()))

        self._log(f"[Phase 1] exhaustive: screening {total} assignments "
                  f"over {len(nodes)} nodes x {restarts} init(s)")

        screened: List[Tuple[float, Tuple[str, ...], int]] = []
        for combo in itertools.product(basis, repeat=len(nodes)):
            best_v, best_r = float('inf'), 0
            for r, init_r in enumerate(inits):
                self.model.load_state_dict(init_r)   # restores weights, not ops
                self._apply_flat_ops(combo)
                v = self._train(self.model, self.cfg.exhaustive_screen_epochs,
                                self.cfg.lr)
                self.trace.evaluations += 1
                if v < best_v:
                    best_v, best_r = v, r
            screened.append((best_v, combo, best_r))
        init = inits[0]

        screened.sort(key=lambda t: t[0])
        self._log(f"    screen best: {screened[0][1]} @ {screened[0][0]:.6g}  "
                  f"| worst: {screened[-1][1]} @ {screened[-1][0]:.6g}")

        best = (float('inf'), None, 0)
        for v_screen, combo, r in screened[:self.cfg.exhaustive_verify_top]:
            self.model.load_state_dict(inits[r])
            self._apply_flat_ops(combo)
            v = self._train(self.model, self.cfg.consolidate_epochs, self.cfg.lr)
            self.trace.evaluations += 1
            self._log(f"    verify {list(combo)} init={r} "
                      f"screen={v_screen:.6g} full={v:.6g}")
            if v < best[0]:
                best = (v, combo, r)

        self.model.load_state_dict(inits[best[2]])
        self._apply_flat_ops(best[1])
        val = self._train(self.model, self.cfg.consolidate_epochs, self.cfg.lr)

        if not np.isfinite(incoming_val) or self._better(val, incoming_val) \
                or val <= incoming_val:
            for (c, l, _), op in zip(nodes, best[1]):
                self._changed.add((c, l))
            self.trace.op_moves.append({
                'move': 'exhaustive', 'to': list(best[1]),
                'val_before': incoming_val, 'val_after': val,
                'configs_evaluated': total,
            })
            self._log(f"[Phase 1] exhaustive best: {list(best[1])} "
                      f"val={val:.6g}")
        else:
            self._log(f"[Phase 1] exhaustive best {list(best[1])} @ {val:.6g} "
                      f"did not beat the incoming model ({incoming_val:.6g}); "
                      f"keeping it")
            self.model = incoming_model.to(self.device)
            return incoming_val, False

        if self.cfg.exhaustive_refine_composites:
            val = self._refine_composites(val)
        return val, True

    def _refine_composites(self, incumbent: float) -> float:
        """One greedy pass trying composites anchored on each node's operator.

        The composite space is too large to enumerate jointly, so this stays
        greedy -- but it starts from an assignment that is optimal over the
        basis rather than from wherever coordinate descent happened to land.
        """
        self._log("[Phase 1] composite refinement")
        before_model = copy.deepcopy(self.model)
        before = incumbent
        changed = False

        for c, l, node in self._flat_nodes():
            current = node.op_name
            adapter = self._adapter_params()
            adapter_original = [p.detach().clone() for p in adapter]
            original = node.snapshot()

            results = [self._probe_best(node, current, c, l, 0, adapter_original)
                       + (current,)]
            for op in composite_candidates(current, list(self.cfg.basis)):
                score, snap, ad = self._probe_best(node, op, c, l, 0,
                                                   adapter_original)
                results.append((score, snap, ad, op))

            best_score, best_snap, best_ad, best_op = min(results,
                                                          key=lambda r: r[0])
            base_score = min(r[0] for r in results if r[3] == current)
            if best_op != current and self._better(best_score, base_score):
                node.restore(best_snap)
                with torch.no_grad():
                    for p, saved in zip(adapter, best_ad):
                        p.copy_(saved)
                self._changed.add((c, l))
                changed = True
                self.trace.op_moves.append({
                    'chain': c, 'layer': l, 'from': current, 'to': best_op,
                    'probe_before': base_score, 'probe_after': best_score,
                })
                self._log(f"    node({c},{l}): {current} -> {best_op}")
            else:
                node.restore(original)
                with torch.no_grad():
                    for p, saved in zip(adapter, adapter_original):
                        p.copy_(saved)

        if not changed:
            self._log("    no composite improved on the basis assignment")
            return incumbent

        after = self._train(self.model, self.cfg.consolidate_epochs, self.cfg.lr)
        if after > before + self.cfg.min_delta * abs(before):
            self._log(f"    refinement reverted: {after:.6g} did not "
                      f"recover {before:.6g}")
            self.model = before_model.to(self.device)
            return before
        self._log(f"    refinement accepted: {before:.6g} -> {after:.6g}")
        return after

    # -- Phase 2 ----------------------------------------------------------
    def _topology_moves(self) -> List[Tuple[str, Callable[[], bool], str]]:
        moves: List[Tuple[str, Callable[[], bool], str]] = []
        n_chains = len(self.model.chains)
        if self.cfg.allow_growth:
            if n_chains < self.cfg.max_chains:
                moves.append(('add_chain', lambda: self.model.add_chain(), 'grow'))
            for i in range(n_chains):
                if len(self.model.chains[i].layers) < self.cfg.max_depth:
                    moves.append((f'deepen_chain[{i}]',
                                  (lambda k: lambda: self.model.deepen_chain(k))(i),
                                  'grow'))
        if self.cfg.allow_pruning:
            if n_chains > 1:
                for i in range(n_chains):
                    moves.append((f'prune_chain[{i}]',
                                  (lambda k: lambda: self.model.prune_chain(k))(i),
                                  'shrink'))
            for i in range(n_chains):
                if len(self.model.chains[i].layers) > 1:
                    moves.append((f'shrink_chain[{i}]',
                                  (lambda k: lambda: self.model.shrink_chain(k))(i),
                                  'shrink'))
        return moves

    def topology_round(self, incumbent: float) -> Tuple[float, bool]:
        """Try each topology move; keep the best that genuinely helps."""
        base_model = copy.deepcopy(self.model)
        base_params = self.model.param_count()
        best = {'val': incumbent, 'name': None, 'model': None, 'params': base_params}

        for name, apply_move, kind in self._topology_moves():
            self.model = copy.deepcopy(base_model)
            if not apply_move():
                continue
            self.model.to(self.device)
            v = self._train(self.model, self.cfg.consolidate_epochs, self.cfg.lr)
            params = self.model.param_count()
            self.trace.evaluations += 1

            # A shrink only needs to hold the line; a grow must actually pay
            # for the parameters it adds.
            ok = (v <= incumbent + self.cfg.min_delta * abs(incumbent)
                  if kind == 'shrink' else self._better(v, incumbent))
            self._log(f"    topo {name:20s} val={v:.6g} params={params} "
                      f"{'ACCEPT' if ok else 'reject'}")
            self.trace.topology_moves.append({
                'move': name, 'kind': kind, 'val': v, 'params': params,
                'accepted': bool(ok),
            })
            if ok and (v < best['val'] or
                       (kind == 'shrink' and params < best['params']
                        and v <= incumbent + self.cfg.min_delta * abs(incumbent))):
                best = {'val': v, 'name': name,
                        'model': copy.deepcopy(self.model), 'params': params}

        if best['name'] is None:
            self.model = base_model.to(self.device)
            self._log("    topology: no move accepted")
            return incumbent, False

        self.model = best['model'].to(self.device)
        self._log(f"    topology: took {best['name']} "
                  f"val={best['val']:.6g} params={best['params']}")
        return best['val'], True

    # -- Phase 3.5 --------------------------------------------------------
    def _train_from_scratch(self, hidden: int, epochs: int, restarts: int,
                            tag: str = '') -> Tuple[float, nn.Module]:
        """Rebuild the current structure at `hidden` and train it `restarts`
        times from independent initialisations, returning the best."""
        best_val, best_model = float('inf'), None
        for r in range(max(1, restarts)):
            seed_everything(self.cfg.seed * 10007 + r * 101 + hidden)
            cand = self.model.resize(hidden).to(self.device)
            v = self._train(cand, epochs, self.cfg.lr)
            self.trace.evaluations += 1
            if v < best_val:
                best_val, best_model = v, cand
            if tag and restarts > 1:
                self._log(f"      {tag} restart {r}: val={v:.6g}")
        return best_val, best_model

    def _lbfgs_polish(self, model: nn.Module, incumbent: float) -> float:
        """Quasi-Newton refinement of a fixed architecture. Reverts if worse.

        L-BFGS can diverge on a badly conditioned objective, so the previous
        weights are kept and restored unless validation actually improves.
        """
        iters = self.cfg.lbfgs_iters
        if iters <= 0:
            return incumbent
        before = copy.deepcopy(model.state_dict())
        params = [p for p in model.parameters() if p.requires_grad]
        opt = optim.LBFGS(params, max_iter=iters, history_size=50,
                          tolerance_grad=1e-14, tolerance_change=1e-16,
                          line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            model.train()
            loss = self.criterion(model(self.Xtr), self.ytr)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except (RuntimeError, ValueError) as exc:
            self._log(f"      lbfgs aborted: {type(exc).__name__}")
            model.load_state_dict(before)
            return incumbent

        after = self.val_loss(model)
        if not np.isfinite(after) or after > incumbent:
            model.load_state_dict(before)
            self._log(f"      lbfgs {incumbent:.4g} -> {after:.4g}: reverted")
            return incumbent
        self._log(f"      lbfgs {incumbent:.4g} -> {after:.4g} "
                  f"({incumbent / max(after, 1e-300):.3g}x)")
        return after

    def final_polish(self, incumbent: float) -> float:
        """Retrain the chosen architecture from scratch and keep the best run.

        The structure is already fixed at this point, so this only addresses the
        weight-optimisation lottery. The incumbent counts as one of the draws,
        so this can never make the result worse.
        """
        R = max(1, self.cfg.final_restarts)
        h = self.model.hparams_dict['hidden_dim']
        budget = self.cfg.warmup_epochs + self.cfg.consolidate_epochs

        if R > 1:
            self._log(f"[Phase 3.5] final polish: {R} independent retrains of "
                      f"{self.model.get_structure()}")
            best_val, best_model = self._train_from_scratch(h, budget, R,
                                                            'polish')
            if best_val < incumbent:
                self.model, incumbent = best_model, best_val
                self._log(f"[Phase 3.5] restarts -> val {incumbent:.6g}")
            else:
                self._log(f"[Phase 3.5] no restart beat the incumbent "
                          f"({incumbent:.6g}); keeping it")

        if self.cfg.lbfgs_iters > 0:
            self._log(f"[Phase 3.5] L-BFGS refinement "
                      f"({self.cfg.lbfgs_iters} iters)")
            incumbent = self._lbfgs_polish(self.model, incumbent)
        return incumbent

    # -- Phase 4 ----------------------------------------------------------
    def compress(self, incumbent: float) -> float:
        """Reduce hidden_dim without giving up performance.

        Two stages, because the two things being controlled for are different:

        1. RANK widths with an equal-budget, from-scratch proxy. Every width --
           including the incumbent's own -- gets a fresh model and the same
           number of epochs, so the comparison isolates width. (Comparing a
           briefly-trained small model against a fully-searched incumbent, the
           previous behaviour, rejects everything for reasons unrelated to width.)

        2. VERIFY the surviving candidates, smallest first, with the full
           training budget, and accept only one that matches the incumbent's
           val loss within tolerance.

        Stage 2 exists because a proxy score is not a result. Selecting on the
        proxy alone shrank the model while making val loss 100x worse and
        reported it as a success.
        """
        h_now = self.model.hparams_dict['hidden_dim']
        full_budget = self.cfg.warmup_epochs + self.cfg.consolidate_epochs
        widths = sorted([w for w in self.cfg.compress_widths if w < h_now])
        if not widths:
            self._log(f"[Phase 4] no smaller width to try at H={h_now}")
            return incumbent

        # -- stage 1: equal-budget ranking
        proxy: Dict[int, float] = {}
        for h in [h_now] + widths:
            cand = self.model.resize(h).to(self.device)
            proxy[h] = self._train(cand, self.cfg.compress_epochs, self.cfg.lr)
            self.trace.evaluations += 1
            self._log(f"    [rank] H={h:3d} params={cand.param_count():6d} "
                      f"proxy={proxy[h]:.6g}")

        best_proxy = min(proxy.values())
        threshold = best_proxy + self.cfg.compress_shortlist_tol * abs(best_proxy)
        shortlist = [h for h in widths if proxy[h] <= threshold]
        self._log(f"[Phase 4] shortlist (proxy <= {threshold:.6g}): {shortlist}")

        # -- stage 2: full-budget verification against the incumbent.
        # Candidates get the same restart budget the incumbent had in Phase 3.5;
        # otherwise a polished incumbent would out-draw every single-run
        # candidate and block reductions for reasons unrelated to width.
        incumbent_bar = incumbent + self.cfg.compress_accept_tol * abs(incumbent)
        for h in shortlist[:self.cfg.compress_verify_top]:
            v, cand = self._train_from_scratch(h, full_budget,
                                               self.cfg.final_restarts)
            ok = v <= incumbent_bar
            self._log(f"    [verify] H={h:3d} params={cand.param_count():6d} "
                      f"val={v:.6g} vs incumbent {incumbent:.6g} "
                      f"{'ACCEPT' if ok else 'reject'}")
            if ok:
                self.model = cand
                self._log(f"[Phase 4] compressed H {h_now} -> {h} "
                          f"({cand.param_count()} params, val={v:.6g})")
                return v

        self._log(f"[Phase 4] no width reduction held up; keeping H={h_now}")
        return incumbent

    # -- driver -----------------------------------------------------------
    def run(self) -> SearchTrace:
        t0 = time.time()
        seed_everything(self.cfg.seed)

        incumbent = self.warmup()

        stable = False
        topo_changed = False
        for sweep in range(self.cfg.max_op_sweeps):
            if self.cfg.search_mode == 'exhaustive':
                # Exhaustive is already optimal over the basis, so repeating it
                # on an unchanged topology would just redo the same work. But a
                # topology move adds or removes nodes, and those new nodes have
                # never been searched -- they carry the ops of whichever chain
                # they were cloned from. Re-run whenever the topology moved.
                if sweep == 0 or topo_changed:
                    incumbent, ops_changed = \
                        self.exhaustive_operator_search(incumbent)
                else:
                    ops_changed = False
            else:
                self._log(f"[Phase 1] operator sweep {sweep + 1}")
                incumbent, ops_changed = self.operator_sweep(incumbent, sweep)
            self.trace.sweeps_run = sweep + 1

            topo_changed = False
            gate_open = ops_changed or not self.cfg.topology_requires_op_gain
            if gate_open and sweep < self.cfg.topology_rounds:
                self._log(f"[Phase 2] topology round {sweep + 1}")
                incumbent, topo_changed = self.topology_round(incumbent)

            self.trace.phase_log.append({
                'phase': f'sweep{sweep + 1}', 'val': incumbent,
                'ops_changed': ops_changed, 'topo_changed': topo_changed,
            })

            if not ops_changed and not topo_changed:
                self._log(f"[Phase 3] structure stable after sweep {sweep + 1}")
                stable = True
                break

        if not stable:
            self._log("[Phase 3] budget exhausted before structure stabilised")

        incumbent = self.final_polish(incumbent)

        if self.cfg.compress:
            incumbent = self.compress(incumbent)

        self.trace.wall_seconds = time.time() - t0
        return self.finalise(incumbent)

    def _measure_importance(self) -> Dict[str, float]:
        """One tracked forward/backward, purely to report per-node importance."""
        if not hasattr(self.model, 'set_tracking'):
            return {}
        self.model.set_tracking(True)
        for _, _, node in self.model.iter_nodes():
            node.reset_metrics()
        self.model.train()
        self.model.zero_grad()
        self.criterion(self.model(self.Xtr), self.ytr).backward()
        out = {}
        for c, l, node in self.model.iter_nodes():
            m = node.get_metrics()
            out[f"{c},{l}"] = m['taylor']
            out[f"{c},{l}|gradnorm"] = m['gradnorm']
        self.model.set_tracking(False)
        self.model.zero_grad()
        return out

    def finalise(self, val: float) -> SearchTrace:
        """Record results. The test split is evaluated here and nowhere else."""
        tr = self.trace
        tr.final_structure = self.model.get_structure()
        tr.nodes_probed = sorted(self._probed)
        tr.nodes_changed = sorted(self._changed)
        tr.val_loss = val
        tr.params = self.model.param_count()
        tr.hidden_dim = self.model.hparams_dict['hidden_dim']
        tr.importance_snapshot = self._measure_importance()
        if self.Xte is not None:
            tr.test_loss = self._loss(self.model, self.Xte, self.yte)
        return tr


# --------------------------------------------------------------------------
# Controls -- the arms the search has to beat to mean anything
# --------------------------------------------------------------------------
def fixed_architecture_control(model, X_train, y_train, X_val, y_val,
                               X_test, y_test, config: SearchConfig,
                               loss_fn=None, device='cpu') -> SearchTrace:
    """Same model, same total epochs, NO structural search.

    If this matches the searched arm, the operator search is contributing
    nothing and the weights are doing all the work.
    """
    s = StructureSearch(model, X_train, y_train, X_val, y_val, X_test, y_test,
                        config, loss_fn, device)
    seed_everything(config.seed)
    s.trace.init_structure = s.model.get_structure()
    total = (config.warmup_epochs
             + config.max_op_sweeps * config.consolidate_epochs)
    val = s._train(s.model, total, config.lr)
    return s.finalise(val)


def random_search_control(model_factory: Callable[[random.Random], nn.Module],
                          X_train, y_train, X_val, y_val, X_test, y_test,
                          config: SearchConfig, n_candidates: int,
                          loss_fn=None, device='cpu') -> SearchTrace:
    """Best-of-N random architectures at a matched evaluation budget.

    The standard sanity check for any architecture search: did you beat random
    sampling given the same number of evaluations?
    """
    seed_everything(config.seed)
    rng = random.Random(config.seed)
    best_trace, best_val = None, float('inf')

    for i in range(n_candidates):
        cand = model_factory(rng)
        s = StructureSearch(cand, X_train, y_train, X_val, y_val,
                            X_test, y_test, config, loss_fn, device)
        s.trace.init_structure = cand.get_structure()
        v = s._train(cand, config.consolidate_epochs, config.lr)
        if v < best_val:
            best_val, best_trace = v, s.finalise(v)
            best_trace.evaluations = n_candidates
    return best_trace


# --------------------------------------------------------------------------
def structural_agreement(structures: List[List[List[str]]],
                         modulo_equivalence: bool = True) -> Dict[str, float]:
    """Mean pairwise agreement between architectures from different seeds.

    With modulo_equivalence, operators are compared by equivalence class, so
    a run picking `sin` and another picking `cos` count as agreeing -- they are
    the same function here, because the surrounding affine maps absorb the
    difference. Comparing raw strings reports instability that has no
    functional content.

    `null` is the agreement expected from independent uniform draws over the
    observed operator vocabulary -- the bar the search must clear to be doing
    anything at all.
    """
    def flatten(s):
        out = []
        for ops in s:
            for op in ops:
                out.append(canonical_class(op) if modulo_equivalence else op)
        return out

    flat = [flatten(s) for s in structures]
    vocab = sorted({o for f in flat for o in f})
    common = min((len(f) for f in flat), default=0)

    agreements = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            a, b = flat[i][:common], flat[j][:common]
            if common:
                agreements.append(sum(x == y for x, y in zip(a, b)) / common)

    return {
        'mean_agreement': float(np.mean(agreements)) if agreements else float('nan'),
        'std_agreement': float(np.std(agreements)) if agreements else float('nan'),
        'null_agreement': 1.0 / len(vocab) if vocab else float('nan'),
        'vocab_size': len(vocab),
        'compared_slots': common,
        'n_pairs': len(agreements),
    }


def slot_frequencies(structures: List[List[List[str]]],
                     modulo_equivalence: bool = True) -> Dict[str, Dict[str, int]]:
    """Per-position operator histogram across seeds.

    Marginal stability ("slot (0,1) picks a periodic operator in 9/10 seeds")
    can hold even when no two seeds agree on the whole architecture -- and it
    is the reportable version of the claim.
    """
    counts: Dict[str, Dict[str, int]] = {}
    for s in structures:
        for c, ops in enumerate(s):
            for l, op in enumerate(ops):
                key = f"({c},{l})"
                name = canonical_class(op) if modulo_equivalence else op
                counts.setdefault(key, {})
                counts[key][name] = counts[key].get(name, 0) + 1
    return counts
