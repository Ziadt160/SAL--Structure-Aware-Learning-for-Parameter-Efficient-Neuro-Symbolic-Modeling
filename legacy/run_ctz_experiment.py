import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import copy
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import GAI components
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from modules.core import GatedMatrixGGLEN, SymbolicJudge, GAIOptimizer
    from modules.core.activations import ACTIVATIONS
except ImportError:
    print("Error: Could not import 'modules.core'. Check python path.")
    exit()

# --- CONFIGURATION ---
DATA_PATH = r"d:\Quantum Projects\GAI\data\EColi_Merged_df.csv"
TARGET_ANTIBIOTIC = "CTZ"

# Experiment Settings
NUM_GROUPS = 3           # 3 Variants of Gated GAI
JUDGES_PER_GROUP = 1     # 1 Judge per GAI
TOTAL_EPOCHS = 2000      # Total Co-Evolution Epochs
BATCH_SIZE = 128
EVOLUTION_PATIENCE = 50  # Epochs before mutation check

# Model Settings
HIDDEN_DIM = 64
DEPTH = 3
NUM_EXPERTS = 3          # Gated GAI Experts

class EvolutionaryGroup:
    """
    Encapsulates 1 Gated GAI (Student) and N Symbolic Judges.
    """
    def __init__(self, group_id, input_dim, device='cpu'):
        self.group_id = group_id
        self.device = device
        self.input_dim = input_dim # Features Only
        
        # Student: GatedMatrixGGLEN
        # Input: Gene Features, Output: Logits (1) for Binary Class
        self.student = GatedMatrixGGLEN(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=HIDDEN_DIM,
            num_chains=NUM_EXPERTS,
            chain_depth=DEPTH
        ).to(device)
        
        self.student_opt = torch.optim.Adam(self.student.parameters(), lr=0.001)
        self.student_history = {} # For mutation tracking
        
        # Judges: List of SymbolicJudge
        # Input: Gene Features + Predicted Label/Prob (1) = input_dim + 1
        self.judges = []
        self.judge_opts = []
        self.judge_histories = []
        
        for i in range(JUDGES_PER_GROUP):
            judge = SymbolicJudge(
                input_dim=input_dim + 1,
                hidden_dim=HIDDEN_DIM,
                depth=DEPTH
            ).to(device)
            
            # Diverse Initialization
            self._randomize_judge(judge)
            
            self.judges.append(judge)
            self.judge_opts.append(torch.optim.Adam(judge.parameters(), lr=0.001))
            self.judge_histories.append({})
            
        self.bce = nn.BCEWithLogitsLoss()
        
    def _randomize_judge(self, judge):
         ops = [k for k in ACTIVATIONS.keys() if k != 'identity']
         for layer in judge.chain.layers:
             if ops: layer.op_name = random.choice(ops)

    def train_step(self, X_batch, y_real_batch):
        """
        Co-evolution step.
        X_batch: Real Genes
        y_real_batch: Real Labels (True Resistance)
        """
        # ==========================
        # 1. TRAIN JUDGES (The Detective)
        # ==========================
        # Goal: Rate (RealGenes, RealLabel) -> 0 (Real)
        #       Rate (RealGenes, StudentPred) -> 1 (Fake)
        
        # Generate Student Predictions (Fake Data)
        with torch.no_grad():
            student_logits = self.student(X_batch)
            student_probs = torch.sigmoid(student_logits)
            
        # Inputs for Judge
        real_pair = torch.cat([X_batch, y_real_batch], dim=1)
        fake_pair = torch.cat([X_batch, student_probs], dim=1)
        
        total_judge_loss = 0
        judge_scores = [] # Monitor separation
        
        for i, judge in enumerate(self.judges):
            opt = self.judge_opts[i]
            opt.zero_grad()
            
            # Forward
            pred_real = judge(real_pair)
            pred_fake = judge(fake_pair)
            
            # Loss: BCE(Real, 0) + BCE(Fake, 1)
            loss = self.bce(pred_real, torch.zeros_like(pred_real)) + \
                   self.bce(pred_fake, torch.ones_like(pred_fake))
                   
            loss.backward()
            opt.step()
            
            total_judge_loss += loss.item()
            
            # Metric: Separation (Sigmoid(Fake) - Sigmoid(Real)) 
            # Ideally -> 1 - 0 = 1.
            with torch.no_grad():
                sep = (torch.sigmoid(pred_fake).mean() - torch.sigmoid(pred_real).mean()).item()
                judge_scores.append(sep)
                
        # ==========================
        # 2. TRAIN STUDENT (The escape artist)
        # ==========================
        # Goal: Rate (RealGenes, StudentPred) -> 0 (Real / FOOL THE JUDGE)
        # AND: Minimize Classification Loss (optional, ground truth supervision)
        # If we ONLY use judges, it's pure unsupervised GAN (likely collapse mode for classification).
        # We should mix in Ground Truth supervision, so Judges acts as regularizers/auxiliary losses.
        
        self.student_opt.zero_grad()
        
        # Forward
        s_logits = self.student(X_batch)
        s_probs = torch.sigmoid(s_logits)
        
        # A. Ground Truth Loss (Task Performance)
        task_loss = self.bce(s_logits, y_real_batch)
        
        # B. Adversarial Loss (Fool Judges)
        # We want Judge(FakePair) -> 0
        fake_pair_grad = torch.cat([X_batch, s_probs], dim=1)
        
        adv_loss = 0
        for judge in self.judges:
            # We freeze judge here logically, though PyTorch graph handles it if we don't zero judge grads
            # Judge outputs raw score.
            j_score = judge(fake_pair_grad)
            adv_loss += self.bce(j_score, torch.zeros_like(j_score))
            
        adv_loss /= len(self.judges)
        
        # Combined Loss
        # detailed balance: Alpha * Task + Beta * Adv
        total_student_loss = task_loss + 0.1 * adv_loss
        
        total_student_loss.backward()
        self.student_opt.step()
        
        return task_loss.item(), adv_loss.item(), judge_scores

    def evolve(self, epoch_idx):
        # Evolve Student
        # Only evolve if stuck? Or periodic? 
        # Let's do periodic for activity
        if epoch_idx > 100 and epoch_idx % EVOLUTION_PATIENCE == 0:
            self.student.evolve_structure(self.student_history, logger_prefix=f"Grp{self.group_id}-Student")
            
        # Evolve Judges
        if epoch_idx > 100 and epoch_idx % EVOLUTION_PATIENCE == 0:
            for i, judge in enumerate(self.judges):
                judge.evolve_structure(self.judge_histories[i], logger_prefix=f"Grp{self.group_id}-Judge{i}")

def load_data():
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Filter valid targets
    if TARGET_ANTIBIOTIC not in df.columns:
        raise ValueError(f"{TARGET_ANTIBIOTIC} not in dataset")
        
    df = df.dropna(subset=[TARGET_ANTIBIOTIC])
    
    # Features (Assume 14 onwards are genes)
    X = df.iloc[:, 14:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # Target
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_ANTIBIOTIC].astype(str))
    
    print(f"Data Shape: {X.shape}, Target Distribution: {np.bincount(y)}")
    return X, y

def main():
    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    input_dim = X_train.shape[1]
    
    # 2. Initialize Groups
    groups = [EvolutionaryGroup(i, input_dim, device) for i in range(NUM_GROUPS)]
    
    print(f"\nInitialized {NUM_GROUPS} Evolutionary Groups.")
    print(f"Each has 1 Gated GAI (Student) and {JUDGES_PER_GROUP} Symbolic Judges.")
    
    # 3. Training Loop
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    
    X_v = torch.FloatTensor(X_test).to(device)
    y_v = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    results = []
    
    try:
        for epoch in range(TOTAL_EPOCHS):
            # Batching could be added, doing full batch for simplicity/speed on tabular
            idx = torch.randperm(X_t.size(0))[:BATCH_SIZE]
            X_batch = X_t[idx]
            y_batch = y_t[idx]
            
            epoch_logs = []
            
            for grp in groups:
                t_loss, a_loss, j_scores = grp.train_step(X_batch, y_batch)
                grp.evolve(epoch)
                
                # Check validation accuracy
                if epoch % 100 == 0:
                    with torch.no_grad():
                        val_logits = grp.student(X_v)
                        acc = ((torch.sigmoid(val_logits) > 0.5) == y_v).float().mean().item()
                    
                    log = {
                        "group": grp.group_id,
                        "epoch": epoch,
                        "task_loss": t_loss,
                        "adv_loss": a_loss,
                        "judge_separation": np.mean(j_scores),
                        "val_acc": acc
                    }
                    epoch_logs.append(log)
            
            if epoch % 100 == 0:
                print(f"\n--- Epoch {epoch} ---")
                for log in epoch_logs:
                    print(f"Grp {log['group']}: Acc {log['val_acc']:.2%} | JudgeSep {log['judge_separation']:.4f} | AdvLoss {log['adv_loss']:.4f}")
                    
    except KeyboardInterrupt:
        print("\nStopping early...")

    # 4. Final Analysis & Export
    print("\n\n>>> FINAL RESULTS <<<")
    final_summary = []
    
    for grp in groups:
        # Final formulas
        # Gated GAI is complex, let's export formulas of first chain as sample
        sample_formula = grp.student.chains[0].export_formula([f"x{i}" for i in range(input_dim)])
        
        # Judge formulas
        judge_formulas = [j.chain.export_formula([f"f{i}" for i in range(input_dim+1)]) for j in grp.judges]
        
        summary = {
            "group_id": grp.group_id,
            "student_formula_sample": sample_formula[:200] + "...", # Truncate
            "judge_formulas": [f[:100]+"..." for f in judge_formulas]
        }
        final_summary.append(summary)
        
        print(f"\n[Group {grp.group_id}]")
        print(f"  Student Sample Formula: {summary['student_formula_sample']}")
        print(f"  Judge Formulas:")
        for idx, form in enumerate(summary['judge_formulas']):
            print(f"    J{idx}: {form}")

    # Save to JSON
    with open('ctz_k3_experiment_results.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
        
if __name__ == "__main__":
    main()
