import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp
import gai # Import the user's legacy/other library

# --- IMPORT YOUR ARCHITECTURE ---
# Assuming you saved the large library code as 'gai_unified.py'
try:
    from gai_unified import GatedMatrixGGLEN, SymbolicJudge, GAIOptimizer, ACTIVATIONS
except ImportError:
    print("Error: Could not import 'gai_unified'. Please ensure the file exists.")
    exit()

# --- ADAPTER ---
# Adapt gai.OPERATIONS (which take x, w, b) to gai_unified (which take z)
# gai_unified nodes already compute z = Wx + b via nn.Linear.
# So we pass w=1, b=0 to the gai functions to apply them to z directly.
GAI_TOOLS = {
    name: lambda x, func=op: func(x, 1.0, 0.0) 
    for name, op in gai.OPERATIONS.items()
}
print(f"Loaded tools from gai.py: {list(GAI_TOOLS.keys())}")

# --- CONFIGURATION ---
EPOCHS_PHASE_1 = 10000  # Detective Training (Needs time to learn the manifold)
EPOCHS_PHASE_2 = 2000   # Student Training
BATCH_SIZE = 64
LR_JUDGE = 0.005
LR_STUDENT = 0.002

# ==========================================
# 1. PHYSICS DATA GENERATION (Double Pendulum)
# ==========================================
def double_pendulum_dynamics(t, state):
    theta1, theta2, p1, p2 = state
    delta = theta1 - theta2
    denom1 = (16 - 9 * np.cos(delta)**2)
    dt1 = (6 * (2 * p1 - 3 * np.cos(delta) * p2)) / denom1
    dt2 = (6 * (8 * p2 - 3 * np.cos(delta) * p1)) / denom1
    dp1 = -(dt1 * dt2 * np.sin(delta)) - (3 * 9.81 * np.sin(theta1))
    dp2 = (dt1 * dt2 * np.sin(delta)) - (9.81 * np.sin(theta2))
    return [dt1, dt2, dp1, dp2]

def get_total_energy(state):
    theta1, theta2, p1, p2 = state
    delta = theta1 - theta2
    denom1 = (16 - 9 * np.cos(delta)**2)
    dt1 = (6 * (2 * p1 - 3 * np.cos(delta) * p2)) / denom1
    dt2 = (6 * (8 * p2 - 3 * np.cos(delta) * p1)) / denom1
    T = 0.5 * (dt1**2 + dt2**2 + 2*dt1*dt2*np.cos(delta))
    y1 = -np.cos(theta1)
    y2 = -(np.cos(theta1) + np.cos(theta2))
    V = 9.81 * (y1 + y2)
    return T + V

def generate_data(dt=0.01, t_span=20):
    # Initial Condition: High energy to ensure chaos
    initial_state = [np.pi/2, np.pi/2, 0, 0] 
    t_eval = np.linspace(0, t_span, int(t_span/dt))
    sol = solve_ivp(double_pendulum_dynamics, [0, t_span], initial_state, t_eval=t_eval, rtol=1e-10)
    return sol.y.T, t_eval

# ==========================================
# 2. MAIN EXPERIMENT LOOP
# ==========================================
def run_experiment():
    print(">>> PHASE 0: Data Generation & Normalization")
    data, t = generate_data(dt=0.01, t_span=40) # 40s of data for better distribution
    
    # --- CRITICAL FIX: NORMALIZATION ---
    # Neural networks struggle when inputs have vastly different scales (angles ~3, momentum ~10).
    # We use StandardScaler to bring everything to mean=0, std=1.
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)
    
    # Create Next-Step Prediction Pairs
    X = data_norm[:-1]
    y = data_norm[1:] # Targets are used only for validation tracking, not for training the Student (Judge is the teacher)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    mu = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    print(f"    Real Data Mean (Norm): {mu}")
    print(f"    Real Data Std  (Norm): {std}")
    
    # ---------------------------------------------------------
    # PHASE 1: THE DETECTIVE (Training SymbolicJudge)
    # ---------------------------------------------------------
    print("\n>>> PHASE 1: The Detective (Training SymbolicJudge)")
    
    # We use depth=3 because Physics energy is quadratic (p^2). 
    # A depth of 2 (Linear->Act->Linear) might be too simple.
    judge = SymbolicJudge(input_dim=4, hidden_dim=64, depth=3, allowed_tools=GAI_TOOLS)
    
    # --- CRITICAL FIX: RANDOM INITIALIZATION ---
    # By default, MatrixChain initializes with 'identity' (Linear).
    # A linear judge CANNOT distinguish chaotic Hamiltonian dynamics from noise (Loss sticks at 1.38).
    # We must start with random non-linear tools to have any capacity.
    print("    [Judge] Randomizing initial structure...")
    import random
    available_ops = list(GAI_TOOLS.keys())
    for i, layer in enumerate(judge.chain.layers):
        # Avoid identity if possible to force non-linearity
        ops = [k for k in available_ops if k != 'identity']
        if ops:
            new_op = random.choice(ops)
            layer.op_name = new_op
            print(f"    [Judge] Layer {i} set to '{new_op}'")
            
    j_opt = torch.optim.Adam(judge.parameters(), lr=LR_JUDGE)
    
    j_opt = torch.optim.Adam(judge.parameters(), lr=LR_JUDGE)
    bce_loss = nn.BCEWithLogitsLoss() # Best loss for "Real vs Fake" classification
    
    loss_history = []
    
    # Track Best Model
    import copy
    best_separation = -1.0 # Maximize this
    best_judge_state = None
    
    for epoch in range(EPOCHS_PHASE_1):
        # A. Sample Real Physics (Label = 0)
        idx = np.random.randint(0, len(X_train), BATCH_SIZE)
        real_batch = torch.FloatTensor(X_train[idx])
        
        # B. Sample Fake Physics (Label = 1)
        # 1. Pure Noise (Gaussian matching data stats)
        fake_noise = np.random.normal(0, 1, (BATCH_SIZE // 4, 4))
        
        # 2. "Broken Physics" (Real data with shuffled columns)
        shuffled_real = X_train[np.random.randint(0, len(X_train), BATCH_SIZE // 4)].copy()
        for i in range(shuffled_real.shape[1]):
            np.random.shuffle(shuffled_real[:, i])
            
        # 3. EXTREME OOD (The Fix for Overflow)
        # Teach Judge that huge inputs (e.g. 50 sigma) are definitely Fake
        fake_ood = np.random.uniform(-50, 50, (BATCH_SIZE // 2, 4))
            
        fake_batch = torch.FloatTensor(np.concatenate([fake_noise, shuffled_real, fake_ood], axis=0))
        
        # C. Forward & Loss
        score_real = judge(real_batch)
        score_fake = judge(fake_batch)
        
        # We want Real -> 0, Fake -> 1
        real_labels = torch.zeros_like(score_real)
        fake_labels = torch.ones_like(score_fake)
        
        loss_real = bce_loss(score_real, real_labels)
        loss_fake = bce_loss(score_fake, fake_labels)
        total_loss = loss_real + loss_fake
        
        j_opt.zero_grad()
        total_loss.backward()
        j_opt.step()
        
        if epoch % 1000 == 0:
            # Sigmoid converts logits to probability (0..1) for readable logs
            p_real = torch.sigmoid(score_real).mean().item()
            p_fake = torch.sigmoid(score_fake).mean().item()
            separation_score = p_fake - p_real
            
            print(f"    [Judge] Epoch {epoch}: Loss {total_loss.item():.4f} (Prob Real {p_real:.2f} vs Fake {p_fake:.2f} | Sep {separation_score:.4f})")
            loss_history.append(total_loss.item())
            
            # --- EVOLUTION STEP ---
            # Try to mutate the judge structure to find better features
            # We skip the first few epochs to let weights settle
            if epoch > 2000:
                judge.evolve_structure(history_tracker={}) # Passing empty dict for simplicity in this demo loop

            # --- CHECKPOINT: Save Best Model (Max Separation) ---
            # We want to maximize the difference between Real and Fake
            if separation_score > best_separation:
                best_separation = separation_score
                best_judge_state = copy.deepcopy(judge.state_dict())
                print(f"    [Checkpoint] New Best Judge found at epoch {epoch} (Separation: {best_separation:.4f})")

    print(f"    Judge Trained. Best Separation: {best_separation:.4f}")
    if best_judge_state is not None:
        print("    Restoring best judge state...")
        judge.load_state_dict(best_judge_state)
        
    # Print Final Structure
    print("\n    >>> Final Evolved Judge Structure <<<")
    for i, layer in enumerate(judge.chain.layers):
        print(f"    Layer {i}: {layer.op_name}")
    print("    " + "="*30)
    
    # ---------------------------------------------------------
    # PHASE 2: THE STUDENT (Training GatedMatrixGGLEN)
    # ---------------------------------------------------------
    print("\n>>> PHASE 2: The Student (Training GatedMatrixGGLEN with Judge Loss)")
    
    student = GatedMatrixGGLEN(input_dim=4, output_dim=4, hidden_dim=64, num_chains=4)
    
    # --- THE CUSTOM ADVERSARIAL LOSS ---
    def judge_loss_fn(preds, targets):
        """
        The Student ignores the ground truth 'targets'.
        It only cares about satisfying the Judge.
        """
        # Ask Judge: "Does this look like real physics?"
        judge_logits = judge(preds)
        
        # Student wants Judge to output 0 (Real)
        target_real = torch.zeros_like(judge_logits)
        
        # Loss is BCE(Judge(Preds), 0)
        loss = bce_loss(judge_logits, target_real)
        return loss
    
    # We use the GAIOptimizer to handle the evolutionary part of the Student
    student_opt = GAIOptimizer(student, lr=LR_STUDENT, task_type='regression', loss_fn=judge_loss_fn)
    
    # Pass data (y_train/y_test are ignored by the loss function but needed for the API)
    student_opt.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS_PHASE_2, name="StudentBot")

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------
    print("\n>>> Evaluation: Generative Capabilities")
    
    # 1. Rollout (Autonomous Physics Simulation)
    start_idx = 0
    current_state_norm = X_test[start_idx] # Normalized start state
    
    rollout_norm = [current_state_norm]
    
    student.eval()
    with torch.no_grad():
        for i in range(400):
            inp = torch.FloatTensor(current_state_norm).unsqueeze(0)
            pred = student(inp).numpy()[0]
            
            # Safety Check
            if not np.all(np.isfinite(pred)):
                print(f"!!! Rollout diverged at step {i} !!!")
                break
                
            # Clamp to prevent localized explosions from cascading
            # 10 sigma is huge for normalized data
            pred = np.clip(pred, -10, 10) 
            
            current_state_norm = pred
            rollout_norm.append(current_state_norm)
            
    # 2. De-Normalize to check Physics
    rollout_real = scaler.inverse_transform(np.array(rollout_norm))
    energies = [get_total_energy(s) for s in rollout_real]
    
    # Ground Truth for Comparison
    ground_truth_norm = X_test[start_idx:start_idx+401]
    ground_truth_real = scaler.inverse_transform(ground_truth_norm)
    gt_energies = [get_total_energy(s) for s in ground_truth_real]

    # 3. Visualization
    plt.figure(figsize=(18, 6))
    
    # Plot A: Config Space (Trajectory)
    plt.subplot(1, 3, 1)
    plt.plot(ground_truth_real[:, 0], ground_truth_real[:, 1], 'k--', alpha=0.5, label='Real Physics')
    plt.plot(rollout_real[:, 0], rollout_real[:, 1], 'r-', linewidth=2, label='AI (Judge-Guided)')
    plt.title("Configuration Space (Theta1 vs Theta2)")
    plt.xlabel("Theta 1")
    plt.ylabel("Theta 2")
    plt.legend()
    
    # Plot B: Energy Conservation (The Kill Shot)
    plt.subplot(1, 3, 2)
    plt.plot(gt_energies, 'k--', label='True Energy')
    plt.plot(energies, 'g-', linewidth=2, label='AI Energy')
    plt.title("Symplectic Stability (Energy)")
    plt.xlabel("Time Step")
    plt.ylabel("Hamiltonian")
    plt.legend()
    
    # Plot C: Judge's Manifold (What did the Detective learn?)
    plt.subplot(1, 3, 3)
    # Scan a slice of phase space (Normalized coords -3 to 3)
    q1s = np.linspace(-3, 3, 50)
    p1s = np.linspace(-3, 3, 50)
    grid_scores = np.zeros((50, 50))
    base_state = np.zeros(4) 
    
    with torch.no_grad():
        for i, q in enumerate(q1s):
            for j, p in enumerate(p1s):
                state = base_state.copy()
                state[0] = q # Theta 1
                state[2] = p # Momentum 1
                score = torch.sigmoid(judge(torch.FloatTensor(state).unsqueeze(0))).item()
                grid_scores[j, i] = score 
                
    plt.imshow(grid_scores, extent=[-3,3,-3,3], origin='lower', cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(label='Judge Prob (0=Real, 1=Fake)')
    plt.xlabel('Theta 1 (Normalized)')
    plt.ylabel('Momentum 1 (Normalized)')
    plt.title("The Detective's Learned Manifold")
    
    plt.tight_layout()
    plt.savefig('symbolic_judge_results.png')
    print("Results saved to symbolic_judge_results.png")

if __name__ == "__main__":
    run_experiment()