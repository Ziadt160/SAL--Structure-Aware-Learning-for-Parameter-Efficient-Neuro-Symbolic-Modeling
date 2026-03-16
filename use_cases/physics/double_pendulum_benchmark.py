
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import Model and Optimizer
# Assuming these are in the python path or same directory
try:
    from gai_unified import GatedMatrixGGLEN, GAIOptimizer
except ImportError:
    # If not found, try adding current dir to path (for standalone execution)
    import sys
    import os
    sys.path.append(os.getcwd())
    from gai_unified import GatedMatrixGGLEN, GAIOptimizer

import argparse

# --- ARGS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Double Pendulum Benchmark")
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--chains', type=int, default=4, help='Number of experts')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--strategy', type=str, default='importance', choices=['importance', 'random'])
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--l1', type=float, default=0.0, help='L1 Regularization')
    return parser.parse_args()

# --- 1. PHYSICS SIMULATION (HAMILTONIAN) ---
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0
G = 9.81

def hamiltonian_derivs(t, y):
    """
    Equations of motion for Double Pendulum from Hamiltonian dynamics.
    State y = [theta1, theta2, p1, p2]
    """
    q1, q2, p1, p2 = y
    
    # Denominator often used
    delta = q1 - q2
    
    c = np.cos(q1 - q2)
    s = np.sin(q1 - q2)
    
    # Hamiltonian derivatives
    # dq/dt = dH/dp
    # dp/dt = -dH/dq
    
    # Determinant of the mass matrix M
    # M = [[2, cos(delta)], [cos(delta), 1]] * l^2 etc
    # Let's simply write the inverted mass matrix form
    denom = 2.0 - c**2
    
    h1 = (p1*p2*s) / (2-c**2)
    h2 = ((p1**2 + 2*p2**2 - 2*p1*p2*c) * s * c) / ((2-c**2)**2) # Simplified derivative of 1/(2-c^2)
    
    dp1_val = -(h1 - h2) - 2*G*np.sin(q1)
    dp2_val = -(-h1 + h2) - G*np.sin(q2) # dC/dq2 = s
    
    dq1 = (p1 - p2 * c) / denom
    dq2 = (2.0 * p2 - p1 * c) / denom
    
    return [dq1, dq2, dp1_val, dp2_val]

def get_total_energy(state):
    """
    Calculates H = T + V for a given state [q1, q2, p1, p2]
    """
    q1, q2, p1, p2 = state
    c = np.cos(q1 - q2)
    denom = 2.0 - c**2
    
    # Kinetic
    T = (p1**2 + 2*p2**2 - 2*p1*p2*c) / (2 * denom)
    
    # Potential
    V = -2*G*np.cos(q1) - G*np.cos(q2)
    
    return T + V

def generate_data(dt=0.005, t_span=10):
    # Initial Condition: Almost upright but chaotic
    y0 = [np.pi/2, np.pi/2, 0.0, 0.0] 
    t_eval = np.arange(0, t_span, dt)
    
    sol = solve_ivp(hamiltonian_derivs, [0, t_span], y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    return sol.y.T, t_eval

if __name__ == "__main__":
    args = parse_args()
    
    # --- 2. EXECUTION ---
    print(f">>> Double Pendulum Benchmark | Config: {args}")
    print(">>> Generating Synthetic Data (Hamiltonian Dynamics)...")
    data, time_points = generate_data()
    print(f"    Data Shape: {data.shape}")

    # Prepare Input/Output pairs (Next Step Prediction)
    X = data[:-1]
    y = data[1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

    # --- 3. MODEL TRAINING ---
    print("\n>>> Training GatedMatrixGGLEN...")
    # Initialize Model
    model = GatedMatrixGGLEN(input_dim=4, output_dim=4, 
                             hidden_dim=args.hidden_dim, 
                             num_chains=args.chains, 
                             dropout=args.dropout)
                             
    optimizer = GAIOptimizer(model, lr=0.005, patience=50, 
                             mutation_strategy=args.strategy, 
                             l1_lambda=args.l1,
                             task_type='regression')

    # Train
    preds = optimizer.fit(X_train, y_train, X_test, y_test, epochs=args.epochs, name="PendulumBot")

    # --- 4. AUTONOMOUS ROLLOUT (The Torture Test) ---
    print("\n>>> Starting Autonomous Rollout & Energy Check...")
    # Take the FIRST point of the test set as the seed
    current_state = X_test[0]
    rollout_states = [current_state]
    energies = [get_total_energy(current_state)]
    ground_truth_energies = [get_total_energy(X_test[i]) for i in range(len(X_test))]

    # How many steps to hallucinate?
    max_steps = len(X_test) - 1
    steps = min(400, max_steps)
    print(f"    Rollout Steps: {steps} (Limited by Test Set size: {len(X_test)})")

    model.eval()

    with torch.no_grad():
        for _ in range(steps):
            # Prepare input
            inp = torch.FloatTensor(current_state).unsqueeze(0)
            
            # Predict next state
            pred = model(inp).numpy()[0]
            
            # Update
            current_state = pred
            rollout_states.append(current_state)
            energies.append(get_total_energy(current_state))

    rollout_states = np.array(rollout_states)
    energies = np.array(energies)
    # Ensure we take exactly the matching ground truth
    gt_energies = np.array(ground_truth_energies[:len(energies)])
    
    # Metric: Energy Conservation Error (MSE of Energy drift)
    energy_mse = mean_squared_error(gt_energies, energies)
    print(f"\n[METRIC] Energy MSE: {energy_mse:.6f}")

    # --- 5. VISUALIZATION ---
    filename = f"pendulum_{args.strategy}_ch{args.chains}_dr{int(args.dropout*100)}.png"
    print(f"\n>>> Generating Visualization '{filename}'...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6)) # Added 4th plot for trajectory diff

    # Plot 1: Configuration Space (q1 vs q2)
    start_test_idx = 0
    gt_segment = X_test[start_test_idx : start_test_idx+steps+1]

    axes[0].plot(gt_segment[:, 0], gt_segment[:, 1], 'k--', alpha=0.6, label='Ground Truth')
    axes[0].plot(rollout_states[:, 0], rollout_states[:, 1], 'r-', linewidth=2, label='AI Rollout')
    axes[0].set_xlabel(r'$\theta_1$')
    axes[0].set_ylabel(r'$\theta_2$')
    axes[0].set_title('Configuration Space Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Phase Space (q2 vs p2) - often chaotic
    axes[1].plot(gt_segment[:, 1], gt_segment[:, 3], 'k--', alpha=0.6, label='Ground Truth')
    axes[1].plot(rollout_states[:, 1], rollout_states[:, 3], 'b-', linewidth=2, label='AI Rollout')
    axes[1].set_xlabel(r'$\theta_2$')
    axes[1].set_ylabel(r'$p_2$')
    axes[1].set_title('Phase Space ($\theta_2, p_2$)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Energy Conservation (The Proof)
    t_axis = np.arange(len(energies))
    axes[2].plot(t_axis, gt_energies, 'k--', label='Physics (Truth)')
    axes[2].plot(t_axis, energies, 'g-', linewidth=2, label='AI (Learned)')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylabel('Total Energy (H)')
    axes[2].set_title(f'Energy (MSE={energy_mse:.4f})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    # Set y-limits to zoom in on energy conservation if close
    e_mean = np.mean(gt_energies)
    axes[2].set_ylim(e_mean - 5, e_mean + 5)

    # Plot 4: Gate Behavior (Optional but cool)
    if hasattr(model, 'gate'):
        # Show gate weights for the rollout
        with torch.no_grad():
            rollout_tensor = torch.FloatTensor(rollout_states)
            gate_logits = model.gate(rollout_tensor)
            gate_weights = torch.nn.functional.softmax(gate_logits, dim=1).numpy()
        
        # Plot heatmap
        im = axes[3].imshow(gate_weights.T, aspect='auto', interpolation='nearest', cmap='viridis')
        axes[3].set_title('Expert Activation during Rollout')
        axes[3].set_xlabel('Time Steps')
        axes[3].set_ylabel('Expert ID')
        plt.colorbar(im, ax=axes[3])


    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Done. Saved to {os.getcwd()}/{filename}")
