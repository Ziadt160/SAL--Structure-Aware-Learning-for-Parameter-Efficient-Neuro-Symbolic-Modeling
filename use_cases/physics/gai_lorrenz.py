import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import copy
import math
import sys

# Start Import Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.adaptive_neural_model import MatrixGGLEN
from models.activations import ACTIVATIONS
# End Import Fix

# --- CONFIGURATION ---
torch.manual_seed(42)
np.random.seed(42)
HIDDEN_DIM = 32 # Wider to capture interactions

# --- 1. DATA: THE LORENZ ATTRACTOR (CHAOS) ---
def lorenz_system(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def generate_lorenz_data(n_points=2000):
    dt = 0.01
    t = np.arange(0, n_points * dt, dt)
    state0 = [1.0, 1.0, 1.0]
    states = odeint(lorenz_system, state0, t)
    
    # We want to predict the DERIVATIVES (dx, dy, dz) from the State (x, y, z)
    derivatives = []
    for s in states:
        derivatives.append(lorenz_system(s, 0))
    
    return torch.FloatTensor(states), torch.FloatTensor(derivatives)

# --- 2. TRAINING WITH ANNEALING ---
def train_chaos_gglen():
    X, Y = generate_lorenz_data()
    
    # GGLEN needs to find specific non-linearities (Square, Identity)
    # Using Core Component
    model = MatrixGGLEN(input_dim=3, output_dim=3, num_chains=3, chain_depth=3, hidden_dim=HIDDEN_DIM)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    history = []
    mutation_history = {}
    
    best_loss = float('inf')
    prev_loss_at_mutation = float('inf')
    epochs_since_imp = 0
    patience = 200
    
    backup_state = None
    backup_chains = None
    just_mutated = False
    grace_period = 100
    
    active_op = None
    active_node = None
    
    INITIAL_TEMP = 0.1
    TOTAL_EPOCHS = 100000

    print("Task: Discovering the Laws of Chaos (Lorenz Attractor)")
    print("Mode: Simulated Annealing with Interaction Search")
    print("-" * 50)

    for epoch in range(TOTAL_EPOCHS):
        progress = epoch / TOTAL_EPOCHS
        current_temp = INITIAL_TEMP * (1 - progress)
        
        model.train()
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, Y)
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        history.append(curr_loss)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: MSE {curr_loss:.6f} | Temp {current_temp:.4f}")

        # Annealing Logic
        if curr_loss < best_loss:
            best_loss = curr_loss
            epochs_since_imp = 0
        else:
            epochs_since_imp += 1

        if just_mutated and epochs_since_imp > grace_period:
            delta = curr_loss - prev_loss_at_mutation
            
            if delta < 0: accept_prob = 1.0
            elif current_temp > 0: accept_prob = math.exp(-delta / current_temp)
            else: accept_prob = 0.0
            
            if random.random() < accept_prob:
                status = "ACCEPTED"
                if delta > 0: status += " (RISKY)"
                print(f"  [Result] '{active_op.upper()}' {status}. Loss: {curr_loss:.6f}")
                just_mutated = False
                epochs_since_imp = 0
                prev_loss_at_mutation = curr_loss
            else:
                print(f"  [Result] '{active_op.upper()}' REJECTED. Loss: {curr_loss:.6f}")
                model.load_state_dict(backup_state)
                # DEEP COPY CHAINS NEEDED for revert
                # Core model handles this if using GAIOptimizer, but here we are manual.
                # Since we are using Core MatrixGGLEN, model.chains is a ModuleList of MatrixChains.
                model.chains = copy.deepcopy(backup_chains) 

                optimizer = optim.Adam(model.parameters(), lr=0.005)
                c, l = active_node
                if (c,l) not in mutation_history: mutation_history[(c,l)] = set()
                mutation_history[(c,l)].add(active_op)
                just_mutated = False
                epochs_since_imp = 0

        # Evolution
        if epochs_since_imp > patience and not just_mutated:
            print(f"\n--- Stagnation at {curr_loss:.6f} ---")
            backup_state = copy.deepcopy(model.state_dict())
            backup_chains = copy.deepcopy(model.chains)
            prev_loss_at_mutation = curr_loss
            
            # Use Core Evolve Structure
            c_id, l_id = model.evolve_structure(mutation_history)
            if c_id is not None:
                active_node = (c_id, l_id)
                active_op = model.chains[c_id].layers[l_id].op_name
                just_mutated = True
                epochs_since_imp = 0
                optimizer = optim.Adam(model.parameters(), lr=0.01)
            else:
                break

    return model, X, Y, history

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    model, X, Y, history = train_chaos_gglen()

    # --- VISUALIZATION ---
    t_test = np.linspace(0, 20, 2000)
    state0 = [1.0, 1.0, 1.0] # Same start
    
    # True System
    states_true = odeint(lorenz_system, state0, t_test)
    
    # GGLEN Simulation
    states_pred = [state0]
    dt = t_test[1] - t_test[0]
    
    model.eval()
    curr_state = torch.FloatTensor(state0)
    
    with torch.no_grad():
        for _ in range(len(t_test)-1):
            derivs = model(curr_state.unsqueeze(0)).squeeze(0)
            curr_state = curr_state + derivs * dt
            states_pred.append(curr_state.numpy())
            
    states_pred = np.array(states_pred)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(states_true[:,0], states_true[:,1], states_true[:,2], lw=0.5, color='black')
    ax1.set_title("True Lorenz Attractor")
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot(states_pred[:,0], states_pred[:,1], states_pred[:,2], lw=0.5, color='red')
    ax2.set_title("GGLEN Discovery (AI Generated)")
    
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(history)
    plt.yscale('log')
    plt.title("Evolution Loss (Learning Physics)")
    plt.show()

    print("\n--- Final Laws of Physics (Architecture) ---")
    for i, chain in enumerate(model.chains):
        ops = " -> ".join([l.op_name.upper() for l in chain.layers])
        print(f"Chain {i}: [ {ops} ]")