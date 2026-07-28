# Theory: Node Importance and Operator Selection

## What changed, and why

The original design ranked nodes by the average gradient norm at their output
and mutated the lowest-ranked one:

$$ I_{\text{grad}}(n_i) = \frac{1}{T} \sum_{t=1}^{T} \lVert \nabla_{z_i} \mathcal{L} \rVert $$

Two problems made this the wrong basis for a search.

**1. It has a systematic depth bias.** Backpropagated gradients shrink with
distance from the loss. In any chain the *earliest* layer therefore has the
smallest gradient norm almost regardless of its contribution, so it is ranked
"least important" and receives nearly every mutation proposal. Combined with a
finite epoch budget, the practical consequence was that most nodes were never
searched at all — they finished a run still carrying the operator the
initialiser happened to draw for them.

**2. Gradient magnitude is not contribution.** A well-trained, load-bearing
node has a *small* gradient precisely because it is near a local optimum.
$I_{\text{grad}}$ conflates "this node is unconverged" with "this node matters."

## Current importance metric (diagnostic)

Importance is now the first-order Taylor estimate of the loss change from
zeroing a node's output:

$$ I(n_i) = \frac{1}{T} \sum_{t=1}^{T} \mathrm{mean}\left| \nabla_{z_i}\mathcal{L} \odot z_i \right| $$

Multiplying by the activation removes most of the depth bias, because a node
that matters has large activations where the gradient is large. This is the
criterion used in Taylor-expansion pruning (Molchanov et al.), and it is what
`importance_mode='taylor'` computes. `importance_mode='gradnorm'` still
computes the original quantity so the two can be compared directly.

## Node selection no longer uses importance

`training/structure_search.py` visits **every node, every sweep**, in a fixed
order. Importance is recorded in the search trace for analysis but does not
decide who gets searched.

The reason is that the per-node operator space is small — six basis operators
after removing exact redundancies (see `models/activations.py`). When a
coordinate has cardinality six you can evaluate all of it. Ranking nodes to
decide who gets *one* random proposal is the right move only when the space is
too large to enumerate, and it trades away guaranteed coverage for nothing.

Empirically, the project's own `assets/pendulum_sweep_results.csv` showed the
importance-guided strategy losing to random mutation across an 18-configuration
sweep, which is the signature you would expect from a ranking rule that
concentrates proposals on whichever node happens to have the smallest gradient.

## Operator selection

For each node, every candidate operator is evaluated under identical
conditions:

1. the node is reset to a **fixed** initialisation, the same one for every
   candidate at that node (seeded from `(seed, sweep, chain, layer)`);
2. the node **and the read-out layer** are fine-tuned for the same number of
   steps — the read-out must be free, or a candidate that changes the output's
   scale is penalised for that rather than judged on its shape;
3. the candidate is scored on the validation split.

The operator currently installed is evaluated the same way, so the bar is the
incumbent measured under identical conditions rather than the incumbent's
fully-trained loss, which would carry a home-field advantage.

A whole sweep is reverted if consolidation cannot recover the validation loss
it started from. Probe scores are a cheap proxy; this is the guard that stops a
proxy from degrading the model.

## Known limitation

Coordinate-wise selection measures a node's marginal value *given the others*.
When two nodes must change together to reach the optimum, no single-node step
sees the improvement, and the search stops at a local optimum. This is visible
in `experiments/operator_recovery.py`: on a target that is exactly
`sin(pi*x1) + x2**2`, runs that reach `sin`+`square` hit a test loss around
1e-14, while runs that stall at `sin`+`sin` or `sin`+`gaussian` plateau near
1e-3 to 1e-4 and cannot escape.

The effect is worse when chain outputs are summed before the read-out
(`readout='sum'`), because then a single shared coefficient must serve every
chain and one chain's error cannot be corrected by another's. `readout='concat'`
gives each chain its own read-out coefficient and decouples the credit
assignment.
