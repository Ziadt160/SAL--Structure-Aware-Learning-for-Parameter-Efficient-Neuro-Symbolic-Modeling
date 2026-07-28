import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2

# --- IMPORT GAI UNIFIED ---
try:
    from gai_unified import GatedMatrixGGLEN, SymbolicJudge, GAIOptimizer, ACTIVATIONS
except ImportError:
    print("Error: Could not import 'gai_unified'. Please ensure the file exists.")
    exit()

# --- CONFIGURATION (FULL POWER) ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DATA_PATH = r"d:\Quantum Projects\GAI\data\EColi_Merged_df.csv"
TARGET_COL = "CIP"
# NUM_FEATURES will be set dynamically based on data
# NUM_FEATURES = 50   # Top 50 Genes (Legacy)
EPOCHS_PHASE_1 = 5000 # Detective Training
EPOCHS_PHASE_2 = 1000 # Student Training
BATCH_SIZE = 128

# Model Config
HIDDEN_DIM = 128    # High capacity
DEPTH = 4           # Deeper reasoning
NUM_CHAINS = 8      # More Student Experts

# ==========================================
# 1. DATA PROCESSING
# ==========================================
def load_and_process_data():
    print(f"Loading data from {DATA_PATH}...")
    # Load limited columns to check structure if needed, but here we load full for SelectKBest
    # To save memory, we might need a smarter approach if 17k cols is too huge.
    # But let's try loading.
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

    # Filter for Target
    if TARGET_COL not in df.columns:
        print(f"Error: Target {TARGET_COL} not found.")
        exit()
        
    # Drop NaNs in Target
    df = df.dropna(subset=[TARGET_COL])
    
    # Gene Features: Assuming they start after metadata. 
    # Based on previous inspection, first few cols are metadata.
    # We will assume columns 14 onwards are genes (0/1).
    # Double check: 'MLST', 'Isolate', 'Year' -> Metadata.
    feature_start_idx = 14
    X_raw = df.iloc[:, feature_start_idx:]
    y_raw = df[TARGET_COL]
    
    # 1. Cleaning Features (Drop non-numeric if any)
    # The dataset seems to be binary genes (0/1), but let's coerce errors.
    print("Processing features...")
    X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 2. Encoding Target
    print(f"Encoding Target '{TARGET_COL}'...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw.astype(str)) # Handle mixed types 'R'/'S'
    # Check mapping
    print(f"Classes: {le.classes_}") 
    # Usually 'R' -> 1, 'S' -> 0 if alphabetical? 
    # 'S' comes after 'R'? No. 'R', 'S'. 'R' is 0, 'S' is 1.
    # We want Resistance=1 usually. Let's fix if needed. 
    # Actually symbolic judge doesn't care about semantics, just consistency.
    
    # 3. Use ALL Features (User Request)
    print(f"Using ALL {X_raw.shape[1]} features (Skipping Feature Selection)...")
    # X_selected = X_raw.values # Convert to numpy array
    # Ensure numeric
    X_selected = X_raw.to_numpy()
    
    selected_names = X_raw.columns.tolist()
    # Update global NUM_FEATURES relative to this run
    global NUM_FEATURES
    NUM_FEATURES = X_selected.shape[1]
    
    return X_selected, y_encoded, selected_names

# ==========================================
# 2. MAIN EXPERIMENT
# ==========================================
def run_experiment():
    print(">>> PHASE 0: Data Loading")
    X, y, feature_names = load_and_process_data()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # Normalize? 
    # Genes are 0/1, but Neural Nets like centered data.
    # Let's simple Standardize.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    
    # ---------------------------------------------------------
    # PHASE 1: THE DETECTIVE (SymbolicJudge)
    # ---------------------------------------------------------
    print("\n>>> PHASE 1: The Detective (Training SymbolicJudge)")
    # Input: Genes (50) + Label (1) = 51
    judge = SymbolicJudge(input_dim=NUM_FEATURES + 1, hidden_dim=HIDDEN_DIM, depth=DEPTH)
    
    # Randomize Init
    import random
    available_ops = list(ACTIVATIONS.keys())
    # Note: SymbolicJudge uses MatrixChain which uses ACTIVATIONS by default if Allowed_tools=None
    # But let's be explicit if we want to force randomness
    available_ops = [k for k in ACTIVATIONS.keys() if k != 'identity']
    for i, layer in enumerate(judge.chain.layers): 
        if available_ops:
            layer.op_name = random.choice(available_ops)
            
    optimizer = torch.optim.Adam(judge.parameters(), lr=0.001)
    bce_loss = nn.BCEWithLogitsLoss()
    
    X_t_tensor = torch.FloatTensor(X_train)
    y_t_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    best_separation = -1.0
    best_judge_state = None
    
    for epoch in range(EPOCHS_PHASE_1):
        # A. Real Data: (Gene, Label)
        # Sample batch
        idx = np.random.randint(0, len(X_train), BATCH_SIZE)
        real_genes = X_t_tensor[idx]
        real_labels = y_t_tensor[idx]
        real_input = torch.cat([real_genes, real_labels], dim=1)
        
        # B. Fake Data 
        # Type 1: Mismatched (Gene, Flipped Label) - Critical for learning consistency
        fake_labels_flip = 1 - real_labels
        fake_input_flip = torch.cat([real_genes, fake_labels_flip], dim=1)
        
        # Type 2: Random Noise (Gene, Random) - Regularization
        noise_labels = torch.rand_like(real_labels)
        fake_input_noise = torch.cat([real_genes, noise_labels], dim=1)
        
        fake_input = torch.cat([fake_input_flip, fake_input_noise], dim=0)
        
        # C. Forward
        score_real = judge(real_input)
        score_fake = judge(fake_input)
        
        # Real -> 0, Fake -> 1
        target_real = torch.zeros_like(score_real)
        target_fake = torch.ones_like(score_fake)
        
        loss = bce_loss(score_real, target_real) + bce_loss(score_fake, target_fake)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            p_real = torch.sigmoid(score_real).mean().item()
            p_fake = torch.sigmoid(score_fake).mean().item()
            sep = p_fake - p_real
            print(f"    [Judge] Epoch {epoch}: Loss {loss.item():.4f} (Sep {sep:.4f})")
            
            # Save Best
            if sep > best_separation:
                best_separation = sep
                best_judge_state = copy.deepcopy(judge.state_dict())
            
            # Evolve
            if epoch > 1000:
                 judge.evolve_structure({}, logger_prefix="Judge")

    if best_judge_state:
        print(f"    Restoring best Judge (Sep {best_separation:.4f})")
        judge.load_state_dict(best_judge_state)

    print("\n    >>> Final Judge Structure <<<")
    for i, layer in enumerate(judge.chain.layers):
        print(f"    Layer {i}: {layer.op_name}")

    # ---------------------------------------------------------
    # PHASE 2: THE STUDENT (GatedMatrixGGLEN)
    # ---------------------------------------------------------
    print("\n>>> PHASE 2: The Student (GatedMatrixGGLEN)")
    student = GatedMatrixGGLEN(
        input_dim=NUM_FEATURES, 
        output_dim=1, 
        hidden_dim=HIDDEN_DIM, 
        num_chains=NUM_CHAINS, 
        chain_depth=DEPTH
    )
    
    # Custom Loss: Fool the Judge
    # Now GAIOptimizer supports loss signatures with inputs: loss(preds, targets, inputs)
    def judge_student_loss(preds, targets, inputs):
        """
        Custom Objective:
        1. Judge Consistency: Judge(cat(Gene, PredLabel)) -> 0 (Real)
        2. (Optional) Supervised Accuracy if we wanted mixed training
        """
        # preds are logits. Convert to prob for Judge.
        probs = torch.sigmoid(preds)
        
        # Judge expects (Gene, Label)
        # inputs is X_batch (Genes)
        judge_input = torch.cat([inputs, probs], dim=1)
        
        # Ask Judge
        judge_score = judge(judge_input)
        
        # We want Judge Output -> 0 (Real)
        target_real = torch.zeros_like(judge_score)
        
        # Loss
        loss = bce_loss(judge_score, target_real)
        return loss

    # Use GAIOptimizer for Phase 2
    # This demonstrates that the Judge is just an optional Loss Function 
    # passed to the standard optimizer.
    print(f"\n    [Info] Using GAIOptimizer with Custom Symbolic Judge Loss.")
    student_opt = GAIOptimizer(
        student, 
        lr=0.001, 
        task_type='classification', # or regression, acts as base 
        loss_fn=judge_student_loss # <--- PLUG IN ANY LOSS HERE
    )
    
    # Fit
    # Note: validation metrics in GAIOptimizer are distinct from optimization loss
    student_opt.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS_PHASE_2, name="StudentBot")
            
    # Evaluation handled by GAIOptimizer fit() returning final preds? 
    # Or just manual eval.
    print("\n>>> Evaluation (Test Set)")
    student.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
        
        logits = student(X_test_t)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        acc = (preds == y_test_t).float().mean().item()
        print(f"Final Test Accuracy: {acc:.2%}")
        
if __name__ == "__main__":
    run_experiment()
