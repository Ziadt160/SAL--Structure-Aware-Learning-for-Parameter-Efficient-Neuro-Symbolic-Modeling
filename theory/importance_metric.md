# Theory: Importance-Based Structural Adaptation

To evolve an architecture efficiently, the system must distinguish between high-value functional units and redundant "dead" nodes. This is achieved through the **Importance Metric**.

## Gradient-Norm Sensitivity
The importance of a node $n_i$ is defined by the average magnitude of the gradient of the loss with respect to that node's output:

$$ I(n_i) = \frac{1}{T} \sum_{t=1}^{T} ||\nabla_{z_i} \mathcal{L}|| $$

where:
- $z_i$ is the output of node $n_i$ after activation.
- $\mathcal{L}$ is the objective function.
- $T$ is the number of observations (steps) over which the metric is accumulated.

### Why Gradient Norm?
A node with a near-zero gradient norm indicates that it is not contributing significantly to the reduction of the loss. This could be due to:
- **Saturation**: The activation function is stuck in a flat region (e.g., deep into the tails of a Sigmoid).
- **Redundancy**: Other paths in the network have already captured the necessary features.
- **Disconnection**: The weights leading from this node have been pushed to zero.

## Structural Mutation Strategy
Nodes with the lowest $I(n_i)$ are prioritized for **Surgical Mutation**. By changing the internal mathematics (activation function) of a low-importance node, we force the optimizer to re-evaluate its utility.

### Simulated Annealing Threshold
Mutations are governed by the probability $P$:

$$ P(accept) = \min(1, \exp(\frac{\Delta S}{T_{temp}})) $$

where $\Delta S$ is the change in validation performance and $T_{temp}$ is the current "temperature" of the system, which cools over time to ensure convergence.
