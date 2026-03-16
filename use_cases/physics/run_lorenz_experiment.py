import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys

# Ensure we can import from local
sys.path.append(os.getcwd())
from gai_unified import GatedMatrixGGLEN, SymbolicJudge, GAIOptimizer, ACTIVATIONS

# --- CONFIGURATION ---
STEPS = 5000
DT = 0.01
MSE_EPOCHS = 2000
JUDGE_PRETRAIN_EPOCHS = 1000
JUDGE_ADVERSARIAL_EPOCHS = 2000

# ==========================================
# 0. LORENZ PHYSICS
# ==========================================
def lorenz_system(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def get_lorenz_data(steps=STEPS, dt=DT):
    t = np.arange(0, steps * dt, dt)
    state0 = [1.0, 1.0, 1.0]
    states = odeint(lorenz_system, state0, t)
    
    # Calculate Derivatives
    derivs = []
    for s in states:
        derivs.append(lorenz_system(s, 0))
    derivs = np.array(derivs)
    
    return torch.FloatTensor(states), torch.FloatTensor(derivs)

# ==========================================
# PHASE 1: DIRECT MSE REGRESSION (Derivatives)
# ==========================================
def run_mse_experiment():
    print("\n" + "="*50)
    print("PHASE 1: Direct MSE Regression (Learning Equations)")
    print("="*50)
    
    states, derivs = get_lorenz_data()
    
    # Split
    split = int(0.8 * len(states))
    X_train, y_train = states[:split], derivs[:split]
    X_test, y_test = states[split:], derivs[split:]
    
    # Model: Predict Derivatives from State
    # 3 Inputs (x,y,z) -> 3 Outputs (dx, dy, dz)
    model = GatedMatrixGGLEN(input_dim=3, output_dim=3, hidden_dim=32, num_chains=3)
    
    loss_fn = nn.MSELoss()
    
    # Use GAIOptimizer
    optimizer = GAIOptimizer(model, lr=0.01, task_type='regression', loss_fn=loss_fn)
    
    print(">>> Training with MSE Loss...")
    optimizer.fit(X_train, y_train, X_test, y_test, epochs=MSE_EPOCHS, name="Lorenz-MSE")
    
    # Validation & Visualization
    print("\n>>> Validating MSE Model...")
    model.eval()
    
    # Autonomous Rollout (Integration)
    pred_states = [states[split].numpy()]
    curr_state = states[split].unsqueeze(0) # [1, 3]
    
    with torch.no_grad():
        for _ in range(len(X_test) - 1):
            # dx = f(x)
            d = model(curr_state)
            # Euler Integration: x_new = x + dx * dt
            next_state = curr_state + d * DT
            pred_states.append(next_state[0].numpy())
            curr_state = next_state
            
    pred_states = np.array(pred_states)
    true_states = X_test.numpy()
    
    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(true_states[:,0], true_states[:,1], true_states[:,2], 'k-', lw=0.5, alpha=0.6, label='True')
    ax1.plot(pred_states[:,0], pred_states[:,1], pred_states[:,2], 'r-', lw=0.8, label='Pred (MSE)')
    ax1.set_title("Phase 1: MSE Regression (Equation Discovery)")
    ax1.legend()
    
    # Print Architecture
    print_architecture(model, "Lorenz-MSE")
    
    return fig

# ==========================================
# PHASE 2: SYMBOLIC JUDGE (Adversarial)
# ==========================================
def run_judge_experiment():
    print("\n" + "="*50)
    print("PHASE 2: Symbolic Judge (Adversarial Learning)")
    print("="*50)
    
    states, _ = get_lorenz_data() # Derivatives interactions implicit in trajectory
    
    # For Judge, we predict Next State directly (or derivatives, let's stick to Derivatives for consistency)
    # Actually, Judge usually looks at (State, Next_State) tuples or just State manifolds.
    # User asked for "using symbolic judge".
    # Let's train a Generator to produce derivatives such that the Resulting State looks "Real" to the Judge.
    
    # 1. Train Judge
    print(">>> Pre-training Symbolic Judge...")
    judge = SymbolicJudge(input_dim=3, hidden_dim=64, depth=3)
    j_opt = torch.optim.Adam(judge.parameters(), lr=0.005)
    bce = nn.BCEWithLogitsLoss()
    
    # Data for judge
    real_data = states
    
    for epoch in range(JUDGE_PRETRAIN_EPOCHS):
        idx = np.random.randint(0, len(real_data), 64)
        real_batch = real_data[idx]
        
        # Fake 1: Noise
        fake_noise = torch.randn_like(real_batch) * 20
        # Fake 2: Shuffled Dimensions (breaking correlations)
        fake_shuf = real_batch.clone()
        fake_shuf = fake_shuf[torch.randperm(64)] 
        
        # Train
        j_opt.zero_grad()
        loss = bce(judge(real_batch), torch.zeros(64,1)) + \
               bce(judge(fake_noise), torch.ones(64,1))
        loss.backward()
        j_opt.step()
        
        if epoch % 200 == 0:
            print(f"    Judge Epoch {epoch}: Loss {loss.item():.4f}")
            
    # 2. Train Generator (Student)
    print("\n>>> Training Student with Judge Feedback...")
    student = GatedMatrixGGLEN(input_dim=3, output_dim=3, hidden_dim=32)
    
    # Loss: MSE(Pred, True) + Lambda * BCE(Judge(Pred_State), Real)
    # Wait, pure Judge loss is hard (no gradient direction). 
    # Usually users want "Physics Enhanced" loss.
    # I will use a hybrid: 90% MSE (Reality) + 10% Judge (Style/Manifold).
    # If I use ONLY judge, it might collapse.
    
    mse_loss = nn.MSELoss()
    
    def hybrid_loss(pred_derivs, targets_derivs, inputs_state):
        # 1. Physics Accuracy
        l_mse = mse_loss(pred_derivs, targets_derivs)
        
        # 2. Judge Satisfaction
        # Predict next state
        next_state = inputs_state + pred_derivs * DT
        # Ask judge
        judge_score = judge(next_state) # Logits. Target = 0 (Real)
        l_judge = bce(judge_score, torch.zeros_like(judge_score))
        
        return l_mse + 0.1 * l_judge

    optimizer = GAIOptimizer(student, lr=0.01, task_type='regression', loss_fn=hybrid_loss)
    
    split = int(0.8 * len(states))
    X_train, y_train = states[:split], derivs_train = get_lorenz_data()[1][:split] # Need derivs again
    X_test, y_test = states[split:], derivs_test = get_lorenz_data()[1][split:]
    
    optimizer.fit(X_train, y_train, X_test, y_test, epochs=JUDGE_ADVERSARIAL_EPOCHS, name="Lorenz-Judge")
    
    # Validation & Visualization
    print("\n>>> Validating Judge-Guided Model...")
    student.eval()
    pred_states = [states[split].numpy()]
    curr_state = states[split].unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(len(X_test) - 1):
            d = student(curr_state)
            next_state = curr_state + d * DT
            pred_states.append(next_state[0].numpy())
            curr_state = next_state
            
    pred_states = np.array(pred_states)
    true_states = X_test.numpy()
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot(true_states[:,0], true_states[:,1], true_states[:,2], 'k-', lw=0.5, alpha=0.6, label='True')
    ax.plot(pred_states[:,0], pred_states[:,1], pred_states[:,2], 'g-', lw=0.8, label='Pred (Judge)')
    ax.set_title("Phase 2: Symbolic Judge (Hybrid Loss)")
    ax.legend()
    
    print_architecture(student, "Lorenz-Judge")
    
    return fig

# ==========================================
# UTILS
# ==========================================
def print_architecture(model, name):
    print(f"\n[{name}] Discovered Architecture:", flush=True)
    input_vars = ['x', 'y', 'z']
    if hasattr(model, 'chains'):
        for i, chain in enumerate(model.chains):
            ops = [layer.op_name for layer in chain.layers if hasattr(layer, 'op_name')]
            print(f"  Expert {i} Ops: {ops}")
            try:
                print(f"  Expert {i} Eq: {chain.export_formula(input_vars)}")
            except:
                pass

if __name__ == "__main__":
    fig1 = run_mse_experiment()
    # Save first plot
    plt.savefig('lorenz_mse.png')
    print("Saved lorenz_mse.png")
    
    fig2 = run_judge_experiment()
    # Combine or save separate? 
    # run_judge_experiment returns a partial figure (subplot 2).
    # Actually let's just create new figures in the functions or cleaner:
    # Re-plot everything at end? No, functions return data or handle plotting.
    # The functions above created subplots on new figures? 
    # run_mse created "fig = plt.figure", "add_subplot(1,2,1)". 
    # run_judge created "fig = plt.figure", "add_subplot(1,2,2)".
    # This might be messy if I wanted them side by side.
    # I'll just save separately.
    
    plt.savefig('lorenz_judge.png')
    print("Saved lorenz_judge.png")
