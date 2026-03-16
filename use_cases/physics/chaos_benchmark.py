
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from gai_moe import GatedMatrixGGLEN
from gai_unified import GAIOptimizer

# --- 1. SYSTEM DEFINITION: Hénon-Heiles ---
def henon_heiles(t, state):
    x, y, px, py = state
    # Hamiltonian Derivatives
    # H = 0.5(px^2 + py^2) + 0.5(x^2 + y^2) + (x^2y - y^3/3)
    dxdt = px
    dydt = py
    dpxdt = -(x + 2*x*y)
    dpydt = -(y + x**2 - y**2)
    return [dxdt, dydt, dpxdt, dpydt]

def hamiltonian_energy(states):
    # Expects states shape [N, 4] or [4]
    if len(states.shape) == 1: states = states.reshape(1, -1)
    x = states[:, 0]
    y = states[:, 1]
    px = states[:, 2]
    py = states[:, 3]
    V = 0.5*(x**2 + y**2) + (x**2*y - y**3/3)
    T = 0.5*(px**2 + py**2)
    return V + T

# --- 2. DATA GENERATION ---
print("Generating Chaos (Hénon-Heiles)...")
t_span = (0, 200)
t_eval = np.linspace(0, 200, 4000)
# Chaotic Initial Condition (Energy approx 1/6 = 0.1666)
y0 = [0.0, -0.2, 0.45, 0.0] 
print(f"Initial Energy: {hamiltonian_energy(np.array(y0))[0]:.5f}")

sol = solve_ivp(henon_heiles, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
data = sol.y.T # Shape [4000, 4]

# Prepare Inputs (State t) and Targets (State t+1)
X = data[:-1]
y = data[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # Shuffle=False to keep trajectory segments if we wanted, but standard split is okay for mapping learning. Actually for chaos, random split is harder? No, map learning is local.

print(f"Data Shapes: X_train {X_train.shape}, X_test {X_test.shape}")

# --- 3. MODEL TRAINING ---
print("\nInitializing GatedMatrixGGLEN (Physics Mode)...")
# 4 inputs -> 4 outputs (Next State)
model = GatedMatrixGGLEN(input_dim=4, output_dim=4, hidden_dim=64, num_chains=4, chain_depth=3, dropout=0.0) 
# Note: Low/No dropout usually better for precise physics regression unless noisy

optimizer = GAIOptimizer(model, 
                         lr=0.01, 
                         patience=30, 
                         grace_period=10,
                         initial_temp=0.1, # Low temp for fine tuning
                         use_annealing=True,
                         mutation_strategy='importance', 
                         task_type='regression') # Use MSE

print("Training (Learning the Hamiltonian Flow)...")
optimizer.fit(X_train, y_train, X_test, y_test, epochs=1500, name="ChaosGAI")

# --- 4. RECURSIVE STABILITY TEST ---
print("\nRunning Recursive Stability Test (500 steps autonomous rollout)...")
model.eval()
initial_state = X_test[0] # Start from a test point
test_ground_truth = []
outputs = []

current_state = torch.FloatTensor(initial_state).unsqueeze(0) # [1, 4]
outputs.append(initial_state)

with torch.no_grad():
    for _ in range(500):
        next_state = model(current_state)
        outputs.append(next_state.numpy()[0])
        current_state = next_state # Feedback loop

pred_traj = np.array(outputs)
truth_traj = y_test[:500] # Compare with actual future if we didn't shuffle. 
# Wait, if we did shuffle=False, X_test is the end of the trajectory.
# But solve_ivp guarantees continuity only if we didn't shuffle.
# I used shuffle=False, so X_test lines up with y_test.
# Actually X_test[0] predicts y_test[0], which is X_test[1] etc.
# So truth is effectively X_test[:501]

# Calculate Energies
energies_pred = hamiltonian_energy(pred_traj)
energies_truth = hamiltonian_energy(X_test[:501])

# --- 5. PLOTTING ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Orbit (x vs y)
axes[0].plot(X_test[:501, 0], X_test[:501, 1], 'k-', alpha=0.5, label='Truth (Integrator)')
axes[0].plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', label='GAI (Autonomous)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Spatial Orbit (Chaos)')
axes[0].legend()

# Plot 2: Phase Space (x vs px)
axes[1].plot(X_test[:501, 0], X_test[:501, 2], 'k-', alpha=0.5)
axes[1].plot(pred_traj[:, 0], pred_traj[:, 2], 'r--')
axes[1].set_xlabel('x')
axes[1].set_ylabel('px')
axes[1].set_title('Phase Space Check')

# Plot 3: Energy Conservation
axes[2].plot(energies_truth, 'k-', label='Truth H (Conserved)')
axes[2].plot(energies_pred, 'r-', label='GAI H (Drift?)')
axes[2].set_xlabel('Time Steps')
axes[2].set_ylabel('Total Energy H')
axes[2].set_title('Conservation of Energy Test')
axes[2].legend()

plt.tight_layout()
plt.savefig('chaos_breakthrough.png')
print("Saved analysis to 'chaos_breakthrough.png'")
