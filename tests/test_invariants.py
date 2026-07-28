"""
Correctness invariants. Run directly:  python tests/test_invariants.py

These are the properties the rest of the project's claims rest on. Each one
corresponds to a bug that was actually present.
"""

import copy
import math
import os
import random
import sys

import numpy as np
import sympy
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.activations import (BASIS_OPS, SEARCH_BASIS, canonical_class,
                                composite_candidates, composite_name, parse_op,
                                to_sympy, to_torch)
from models.adaptive_neural_model import (GatedMatrixGGLEN, MatrixGGLEN,
                                          MatrixChain)
from models.symbolic_neuron import MatrixSymbolicNode
from training.structure_search import (SearchConfig, StructureSearch,
                                       seed_everything, structural_agreement)

FAILURES = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ''))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------
def test_torch_sympy_agreement():
    """The exported formula must be the function the network computes.

    Regression test for: `sin_of_square` executed as sin(x)**2 but exported as
    sin(x**2), because forward and export parsed operator names separately.
    """
    print("\ntorch/sympy operator agreement")
    x = sympy.Symbol('x')
    t = torch.linspace(-2.5, 2.5, 21, dtype=torch.float64)

    ops = list(BASIS_OPS)
    for a in ('sin', 'square', 'tanh'):
        ops += composite_candidates(a, SEARCH_BASIS)
    ops += ['sin_of_square_of_tanh', 'gaussian_x_sin_of_square',
            'relu_plus_tanh_of_sin']

    worst, worst_op = 0.0, None
    for op in dict.fromkeys(ops):
        f = to_torch(op)
        g = sympy.lambdify(x, to_sympy(op, x), 'numpy')
        a = f(t).numpy()
        b = np.asarray(g(t.numpy()), dtype=np.float64)
        d = float(np.max(np.abs(a - b)))
        if d > worst:
            worst, worst_op = d, op
    check('all operators agree to 1e-9', worst < 1e-9,
          f'worst {worst:.2e} on {worst_op} over {len(set(ops))} operators')


def test_name_roundtrip_and_precedence():
    print("\nname parsing")
    check("'a_of_b' means a(b(x))",
          parse_op('sin_of_square') == ('of', ('atom', 'sin'), ('atom', 'square')))
    check('_of_ binds tighter than _x_',
          parse_op('tanh_x_relu_of_sin')[0] == 'mul')
    check('commutative composites are canonically ordered',
          composite_name('sin', 'cos', 'mul') == composite_name('cos', 'sin', 'mul'))

    raised = False
    try:
        parse_op('nonexistent_op')
    except ValueError:
        raised = True
    check('unknown operator raises instead of silently becoming identity', raised)


def test_declared_equivalences():
    """sin==cos and tanh==sigmoid, given the surrounding affine maps."""
    print("\ndeclared equivalence classes")
    u = torch.linspace(-3, 3, 41, dtype=torch.float64)
    # cos(u) == sin(pi * (u/pi + 1/2)); the preceding linear layer supplies both
    lhs = torch.cos(u)
    rhs = torch.sin(torch.pi * (u / torch.pi + 0.5))
    check('cos is reachable from sin by reweighting',
          float((lhs - rhs).abs().max()) < 1e-12)
    # sigmoid(u) == 1/2 + 1/2*tanh(u/2); the FOLLOWING linear layer absorbs both
    lhs = torch.sigmoid(u)
    rhs = 0.5 + 0.5 * torch.tanh(u / 2)
    check('sigmoid is reachable from tanh by reweighting',
          float((lhs - rhs).abs().max()) < 1e-12)
    check('equivalence classes agree', canonical_class('sin') == canonical_class('cos'))


def test_tool_dict_isolation():
    """A composite discovered by one node must not appear in another.

    Regression test for: every node held a reference to one module-level dict,
    so discoveries leaked process-wide and results depended on run order.
    """
    print("\noperator registry isolation")
    a = MatrixSymbolicNode(3, 3, 'tanh')
    b = MatrixSymbolicNode(3, 3, 'tanh')
    a.set_op('sin_of_square')
    check('nodes do not share a registry', 'sin_of_square' not in b.tools,
          f'a has {len(a.tools)} ops, b has {len(b.tools)}')

    import models.activations as A
    check('the module-level registry was not mutated',
          'sin_of_square' not in A.ACTIVATIONS)


def test_export_matches_forward():
    """Whole-model formula export must reproduce the model numerically."""
    print("\nmodel formula export")
    seed_everything(3)
    for readout in ('sum', 'concat'):
        m = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=3, num_chains=2,
                        chain_depth=2, rng=random.Random(3), readout=readout)
        m.chains[0].layers[0].set_op('sin_of_square')
        m.chains[1].layers[1].set_op('gaussian_x_tanh')
        m.eval()

        xs = torch.randn(6, 2, dtype=torch.float32)
        with torch.no_grad():
            want = m(xs).numpy().ravel()

        formula = m.export_formula(['x0', 'x1'])
        exprs = sympy.sympify(formula)
        fn = sympy.lambdify([sympy.Symbol('x0'), sympy.Symbol('x1')],
                            exprs, 'numpy')
        got = np.array([np.asarray(fn(*row)).ravel()[0]
                        for row in xs.numpy()], dtype=np.float64)
        d = float(np.max(np.abs(want - got)))
        check(f'export == forward ({readout} readout)', d < 1e-4, f'max diff {d:.2e}')

    raised = False
    try:  # the 784-input MNIST case used to hang instead of refusing
        MatrixGGLEN(input_dim=784, output_dim=10, hidden_dim=32,
                    num_chains=2, chain_depth=3).export_formula(
                        [f'x{i}' for i in range(784)])
    except ValueError:
        raised = True
    check('intractable export refuses instead of hanging', raised)


def test_function_preserving_topology():
    print("\nfunction-preserving topology growth")
    seed_everything(5)
    for cls, kw in ((MatrixGGLEN, {'readout': 'sum'}),
                    (MatrixGGLEN, {'readout': 'concat'}),
                    (GatedMatrixGGLEN, {})):
        m = cls(input_dim=3, output_dim=2, hidden_dim=5, num_chains=2,
                chain_depth=2, rng=random.Random(5), **kw)
        m.eval()
        xs = torch.randn(8, 3)
        with torch.no_grad():
            y0 = m(xs)
            m.add_chain()
            d_add = float((m(xs) - y0).abs().max())
            m.deepen_chain(0)
            d_deep = float((m(xs) - y0).abs().max())
        tag = f"{cls.__name__}{kw.get('readout', '')}"
        check(f'{tag}: add_chain preserves output', d_add < 1e-5, f'{d_add:.2e}')
        check(f'{tag}: deepen_chain preserves output', d_deep < 1e-5, f'{d_deep:.2e}')


def test_prune_keeps_gate_aligned():
    """Pruning a middle expert must drop that expert's gate row, not the last."""
    print("\ngate alignment after pruning")
    seed_everything(7)
    m = GatedMatrixGGLEN(input_dim=3, output_dim=1, hidden_dim=4, num_chains=3,
                         chain_depth=1, rng=random.Random(7))
    before = m.gate[-1].weight.detach().clone()
    m.prune_chain(1)
    after = m.gate[-1].weight.detach()
    check('gate rows 0 and 2 survive pruning chain 1',
          torch.allclose(after[0], before[0]) and torch.allclose(after[1], before[2]))
    check('gate width tracks chain count', after.shape[0] == len(m.chains))


def test_checkpoint_roundtrip():
    """state_dict alone cannot carry the architecture -- op_name is a string."""
    print("\ncheckpoint round-trip")
    seed_everything(11)
    m = MatrixGGLEN(input_dim=3, output_dim=2, hidden_dim=4, num_chains=2,
                    chain_depth=2, rng=random.Random(11))
    m.chains[0].layers[1].set_op('sin_of_gaussian')
    m.eval()
    xs = torch.randn(5, 3)
    with torch.no_grad():
        want = m(xs)

    restored = MatrixGGLEN.from_checkpoint(m.checkpoint())
    restored.eval()
    with torch.no_grad():
        got = restored(xs)
    check('structure survives a checkpoint',
          restored.get_structure() == m.get_structure(),
          str(restored.get_structure()))
    check('weights survive a checkpoint', float((want - got).abs().max()) < 1e-6)


def test_search_determinism():
    """Same seed must give the same architecture."""
    print("\nsearch determinism")
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (300, 2)).astype(np.float32)
    y = (np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1).astype(np.float32)

    def once():
        seed_everything(42)
        cfg = SearchConfig(seed=42, warmup_epochs=60, probe_epochs=15,
                           consolidate_epochs=60, max_op_sweeps=1,
                           use_composites=False, topology_rounds=0,
                           allow_growth=False, allow_pruning=False,
                           compress=False, verbose=False)
        m = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=3, num_chains=2,
                        chain_depth=1, rng=random.Random(42))
        return StructureSearch(m, X[:180], y[:180], X[180:240], y[180:240],
                               X[240:], y[240:], cfg).run()

    a, b = once(), once()
    check('same seed -> same structure', a.final_structure == b.final_structure,
          str(a.final_structure))
    check('same seed -> same val loss', abs(a.val_loss - b.val_loss) < 1e-9)
    check('every node was probed', a.coverage() == 1.0, f'{a.coverage():.0%}')


def test_agreement_metric():
    print("\nstructural agreement metric")
    same = [[['sin']], [['cos']]]            # equivalent, different names
    r = structural_agreement(same, modulo_equivalence=True)
    check('sin vs cos counts as agreement modulo equivalence',
          r['mean_agreement'] == 1.0)
    r2 = structural_agreement(same, modulo_equivalence=False)
    check('sin vs cos counts as disagreement on raw strings',
          r2['mean_agreement'] == 0.0)


def test_homotopy_swap_is_function_preserving():
    """A homotopy operator swap must start bit-identical and end exact.

    This is the property that makes the swap worth doing: at t=0 the network
    computes exactly what it computed before, so a mutation costs nothing and
    the accept/reject test measures the OPERATOR rather than the damage from
    re-initialising the node.
    """
    print("\nhomotopy operator swap")
    seed_everything(17)
    node = MatrixSymbolicNode(5, 5, 'tanh')
    x = torch.randn(8, 5)
    before = node(x).detach().clone()

    node.begin_swap('square')
    d0 = float((node(x) - before).abs().max())
    check('t=0 is bit-identical to before the swap', d0 == 0.0, f'{d0:.2e}')

    node.set_swap_t(0.5)
    mid = node(x).detach()
    check('t=0.5 differs from both endpoints',
          float((mid - before).abs().max()) > 0)

    node.set_swap_t(1.0)
    import models.activations as A
    want = A.to_torch('square')(node.dropout(node.linear(x)))
    d1 = float((node(x) - want).abs().max())
    check('t=1 equals the new operator exactly', d1 == 0.0, f'{d1:.2e}')
    check('swap finalises the operator', node.op_name == 'square'
          and not node.swapping)

    node2 = MatrixSymbolicNode(5, 5, 'tanh')
    b2 = node2(x).detach().clone()
    node2.begin_swap('sin'); node2.set_swap_t(0.7); node2.cancel_swap()
    check('cancel restores the original exactly',
          node2.op_name == 'tanh' and float((node2(x) - b2).abs().max()) == 0.0)


def test_reset_scales_and_avoids_stationary_points():
    """The reset defaults must stay the NARROW original draw.

    A Xavier-scaled draw for non-square nodes is the textbook choice and it
    measured worse: recovery on sin(pi*x1)+x2^2 fell 3/8 -> 0/8 seeds, because
    at width 1 a node is Linear(2->1) and Xavier gives U(-1.414, 1.414) against
    U(-0.05, 0.05). The narrow draw is a start-simple-and-grow prior that lets
    sin(pi*w*x) grow into the right frequency instead of starting in a wrong
    basin. This test exists so that reasoning-from-first-principles cannot
    silently re-break it. Both behaviours stay reachable via the options.
    """
    print("\nmutation reset initialisation")
    seed_everything(23)
    xavier = math.sqrt(6.0 / (3 + 32)) / math.sqrt(3.0)   # uniform -> std

    wide = MatrixSymbolicNode(3, 32, 'tanh')     # in_dim != out_dim
    check('non-square reset defaults to the narrow draw',
          wide.reset_scale == 'small')
    wide.reset_weights_near_identity()
    std = float(wide.linear.weight.std())
    narrow = 0.05 / math.sqrt(3.0)
    check('default non-square reset is U(-0.05,0.05), NOT Xavier',
          abs(std - narrow) < 0.25 * narrow,
          f'std {std:.4f} vs narrow {narrow:.4f} / xavier {xavier:.4f}')

    wide.reset_scale = 'xavier'
    wide.reset_weights_near_identity()
    std_x = float(wide.linear.weight.std())
    check("reset_scale='xavier' still reaches the wide draw",
          std_x > 0.5 * xavier, f'std {std_x:.4f} vs xavier std {xavier:.4f}')

    # Nudging square/gaussian off d/dz = 0 measured neutral (3/8 -> 2/8, same
    # best loss). Off by default, but must still work when asked for.
    for op in ('square', 'gaussian'):
        n = MatrixSymbolicNode(4, 4, op)
        n.reset_weights_near_identity()
        check(f'{op} reset leaves bias at 0 by default',
              float(n.linear.bias.abs().max()) == 0.0)
        n.reset_offset_stationary = True
        n.reset_weights_near_identity()
        b = float(n.linear.bias.abs().mean())
        check(f'{op} opt-in reset is off its stationary point', b > 0.0,
              f'bias {b:.3f}')


def test_reset_flag_is_live_only_in_reset_mode():
    """`mutate_reset_weights` is a no-op under mutation_mode='homotopy'.

    The homotopy path restores backup_state after evolve_structure runs, which
    undoes the weight reset before it can affect anything. Any experiment that
    builds a reset-vs-transfer comparison out of `mutate_reset_weights` ALONE is
    therefore comparing two identical configurations that differ only in how
    many draws the discarded reset took from the global RNG.
    experiments/mutation_mechanics.py did exactly that. This pins the
    interaction so the next such comparison fails loudly instead of quietly
    reporting noise.
    """
    print("\nmutate_reset_weights x mutation_mode")
    from training.trainer import GAIOptimizer

    def mutate_once(mode, reset):
        seed_everything(5)
        m = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=4, num_chains=1,
                        chain_depth=2, rng=random.Random(5))
        opt = GAIOptimizer(m, mutation_mode=mode, mutate_reset_weights=reset)
        before = {(c, l): n.linear.weight.detach().clone()
                  for c, l, n in m.iter_nodes()}
        opt._trigger_evolution(-1.0, 'test')
        c, l = opt.active_node
        node = m.chains[c].layers[l]
        moved = float((node.linear.weight - before[(c, l)]).abs().max())
        return moved

    for reset in (True, False):
        moved = mutate_once('homotopy', reset)
        check(f'homotopy leaves weights untouched (reset={reset})',
              moved == 0.0, f'max |dW| {moved:.2e}')

    moved_reset = mutate_once('reset', True)
    check("mutation_mode='reset' + reset=True DOES reinitialise",
          moved_reset > 0.0, f'max |dW| {moved_reset:.2e}')

    moved_xfer = mutate_once('reset', False)
    check("mutation_mode='reset' + reset=False transfers weights",
          moved_xfer == 0.0, f'max |dW| {moved_xfer:.2e}')


def test_mutation_log_epoch_contract():
    """mutation_log[i]['epoch'] is the epoch whose score is still PRE-mutation.

    Evolution fires at the end of an epoch, after that epoch's validation score
    has already been appended to history. So for a logged epoch e,
    history[e] is the last pre-mutation reading and history[e+1] is the first
    one that reflects the change. Analysis code that treats history[e] as the
    post-mutation value reports a shock of 1.0x for a full weight reset, which
    is how this was found. e+1 must always be a valid index.
    """
    print("\nmutation_log epoch contract")
    from training.trainer import GAIOptimizer
    seed_everything(31)
    X = torch.randn(64, 2)
    y = (torch.sin(math.pi * X[:, :1]) + X[:, 1:] ** 2)
    m = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=4, num_chains=1,
                    chain_depth=2, rng=random.Random(31))
    # min_rel_delta=1.0 demands a 100% relative improvement to count as
    # progress, so the stagnation counter accrues every epoch and mutations are
    # guaranteed to fire. With the default 1e-3 a smoothly descending loss
    # resets the counter every epoch and nothing ever mutates.
    opt = GAIOptimizer(m, loss_fn=torch.nn.MSELoss(), lr=0.01, patience=5,
                       grace_period=3, mutation_mode='reset',
                       use_annealing=False, min_rel_delta=1.0)
    _, history = opt.fit(X, y, X, y, epochs=200, name='logtest')

    log = opt.mutation_log
    check('mutations were actually logged', len(log) > 0, f'{len(log)} events')
    check('every logged epoch leaves a valid e+1 index',
          all(e['epoch'] + 1 < len(history) for e in log),
          f"max epoch {max(e['epoch'] for e in log)}, history {len(history)}")
    check('logged epochs are non-decreasing',
          all(b['epoch'] >= a['epoch'] for a, b in zip(log, log[1:])))
    check('every event records the node it touched',
          all(e['chain'] is not None and e['layer'] is not None for e in log))
    judged = [e for e in log if e['accepted'] is not None]
    check('all but at most the last event are judged',
          len(judged) >= len(log) - 1,
          f'{len(judged)}/{len(log)} judged')


def test_validation_is_measured_in_eval_mode():
    """fit() must score validation with dropout OFF.

    `torch.no_grad()` disables gradients but NOT dropout, so validating while
    the model is still in train() mode gives a score that is both noisy and
    biased high. At dropout=0.3 five readings of a frozen model spanned
    1.2327-1.3092 against a true 1.1158. That score drives best-model tracking,
    stagnation detection and mutation accept/reject, so the noise would
    propagate into every structural decision.
    """
    print("\nvalidation runs with dropout disabled")
    from training.trainer import GAIOptimizer
    seed_everything(41)
    X = np.random.randn(120, 3).astype(np.float32)
    y = np.random.randn(120, 1).astype(np.float32)
    m = MatrixGGLEN(input_dim=3, output_dim=1, hidden_dim=8, num_chains=2,
                    chain_depth=2, dropout=0.3, rng=random.Random(41))

    seen = {}
    orig = m.forward

    def spy(x):
        # Record the training flag at each forward; the validation forward is
        # the one running under no_grad.
        seen.setdefault('modes', []).append(
            (m.training, torch.is_grad_enabled()))
        return orig(x)

    m.forward = spy  # type: ignore
    opt = GAIOptimizer(m, loss_fn=torch.nn.MSELoss(), lr=0.01, patience=10_000)
    opt.fit(X, y, X, y, epochs=3, name='evaltest')
    m.forward = orig  # type: ignore

    val_calls = [tr for tr, grad in seen['modes'] if not grad]
    check('validation forwards happened', len(val_calls) >= 3,
          f'{len(val_calls)} no-grad forwards')
    check('every validation forward ran with training=False',
          not any(val_calls), f'train-mode validation forwards: {sum(val_calls)}')

    # And the score itself must be deterministic for a fixed model.
    m.eval()
    with torch.no_grad():
        a = float(torch.nn.MSELoss()(m(torch.as_tensor(X)), torch.as_tensor(y)))
        b = float(torch.nn.MSELoss()(m(torch.as_tensor(X)), torch.as_tensor(y)))
    check('eval-mode score is deterministic', a == b, f'{a:.6f} vs {b:.6f}')


def test_fit_with_restarts_splits_budget_and_keeps_best():
    """fit_with_restarts must split the budget, use fresh models, keep the best.

    The three ways this could silently be wrong: spend `epochs` per restart
    instead of `epochs/restarts` (an unfair comparison against a single run),
    reuse one model across restarts (defeating the point), or return something
    other than the best-on-validation run.
    """
    print("\nfit_with_restarts")
    from training.trainer import fit_with_restarts
    seed_everything(19)
    X = np.random.randn(80, 2).astype(np.float32)
    y = np.sin(np.pi * X[:, :1]) + X[:, 1:] ** 2

    # Count factory CALLS and record the seeds it was handed. Comparing id()
    # values here would be flaky: CPython recycles ids once an object is freed,
    # and each restart's model is garbage-collected before the next is built, so
    # two genuinely distinct models can share an id.
    calls = {'n': 0, 'seeds': []}

    def factory(s):
        calls['n'] += 1
        calls['seeds'].append(s)
        return MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=1, num_chains=2,
                           chain_depth=1, rng=random.Random(s))

    best, info = fit_with_restarts(factory, X, y, X, y, epochs=120, restarts=4,
                                   seed0=0, name='rtest', lr=0.01,
                                   patience=10_000, loss_fn=torch.nn.MSELoss())
    check('budget is split, not multiplied',
          info['epochs_per_restart'] == 30, f"{info['epochs_per_restart']} epochs")
    check('every restart built a fresh model from the factory',
          calls['n'] == 4, f"factory called {calls['n']} times")
    check('each restart got a distinct seed',
          len(set(calls['seeds'])) == 4, f"seeds {calls['seeds']}")
    check('all restarts ran', len(info['runs']) == 4)
    vals = [r['val'] for r in info['runs']]
    check('returns the best-on-validation run',
          abs(info['best_val'] - min(vals)) < 1e-12,
          f"{info['best_val']:.6e} vs min {min(vals):.6e}")
    check('returned model scores the reported best', best is not None)
    try:
        fit_with_restarts(factory, X, y, X, y, epochs=10, restarts=0)
        check('restarts=0 rejected', False)
    except ValueError:
        check('restarts=0 rejected', True)


def test_optimizer_owns_the_models_real_tensors():
    """After a mutation or a revert, Adam must step the model's ACTUAL tensors.

    `_revert` and the homotopy path both replace `model.chains` with a
    `deepcopy`, which creates entirely new parameter tensors. If the optimizer
    were not rebound afterwards it would keep stepping the orphaned ones and
    training would silently become a no-op -- no error, no NaN, just a model
    that stops learning. This checks tensor IDENTITY, not equality, and then
    confirms a real step still moves the weights.
    """
    print("\noptimizer owns the model's tensors")
    from training.trainer import GAIOptimizer
    X = np.random.uniform(-1, 1, (150, 2)).astype(np.float32)
    y = (np.sin(math.pi * X[:, :1]) + X[:, 1:] ** 2).astype(np.float32)

    for mode in ('reset', 'homotopy'):
        seed_everything(11)
        m = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=4, num_chains=2,
                        chain_depth=2, rng=random.Random(11))
        opt = GAIOptimizer(m, loss_fn=torch.nn.MSELoss(), lr=0.01,
                           mutation_mode=mode)

        def owned():
            mp = {id(p) for p in opt.model.parameters()}
            op = {id(p) for g in opt.optimizer.param_groups for p in g['params']}
            return mp, op

        opt._trigger_evolution(-1.0, 'chk')
        mp, op = owned()
        check(f'[{mode}] optimizer owns exactly the model params after mutation',
              mp == op, f'stale {len(op - mp)}, missing {len(mp - op)}')

        opt._revert('chk')
        mp, op = owned()
        check(f'[{mode}] optimizer owns exactly the model params after revert',
              mp == op, f'stale {len(op - mp)}, missing {len(mp - op)}')

        before = [p.detach().clone() for p in opt.model.parameters()]
        opt.model.train()
        opt.optimizer.zero_grad()
        torch.nn.MSELoss()(opt.model(torch.as_tensor(X)),
                           torch.as_tensor(y)).backward()
        opt.optimizer.step()
        moved = sum(1 for a, b in zip(before, opt.model.parameters())
                    if not torch.equal(a, b))
        check(f'[{mode}] a step still moves the weights after revert',
              moved == len(before), f'{moved}/{len(before)} tensors changed')


def test_optimizer_honours_lr():
    """GAIOptimizer must not silently reset lr to 0.001 after a mutation."""
    print("\nGAIOptimizer learning rate")
    from training.trainer import GAIOptimizer
    seed_everything(13)
    m = MatrixGGLEN(input_dim=2, output_dim=1, hidden_dim=3, num_chains=1,
                    chain_depth=2, rng=random.Random(13))
    opt = GAIOptimizer(m, lr=0.037)
    check('initial lr honoured',
          opt.optimizer.param_groups[0]['lr'] == 0.037)
    opt._trigger_evolution(-1.0, 'test')
    check('lr survives a mutation',
          opt.optimizer.param_groups[0]['lr'] == 0.037,
          f"got {opt.optimizer.param_groups[0]['lr']}")
    opt.active_node = (0, 0)
    opt.backup_state = copy.deepcopy(m.state_dict())
    opt._revert('test')
    check('lr survives a revert',
          opt.optimizer.param_groups[0]['lr'] == 0.037,
          f"got {opt.optimizer.param_groups[0]['lr']}")


def main():
    torch.set_num_threads(4)
    print("=" * 72)
    print(" INVARIANT TESTS")
    print("=" * 72)
    for fn in (test_torch_sympy_agreement, test_name_roundtrip_and_precedence,
               test_declared_equivalences, test_tool_dict_isolation,
               test_export_matches_forward, test_function_preserving_topology,
               test_prune_keeps_gate_aligned, test_checkpoint_roundtrip,
               test_search_determinism, test_agreement_metric,
               test_homotopy_swap_is_function_preserving,
               test_reset_scales_and_avoids_stationary_points,
               test_reset_flag_is_live_only_in_reset_mode,
               test_mutation_log_epoch_contract,
               test_validation_is_measured_in_eval_mode,
               test_optimizer_owns_the_models_real_tensors,
               test_fit_with_restarts_splits_budget_and_keeps_best,
               test_optimizer_honours_lr):
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            FAILURES.append(f'{fn.__name__} (raised {type(exc).__name__})')

    print("\n" + "=" * 72)
    if FAILURES:
        print(f" {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"   - {f}")
        return 1
    print(" ALL INVARIANTS HOLD")
    return 0


if __name__ == '__main__':
    sys.exit(main())
