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
try:
    from gai_unified import GatedMatrixGGLEN, SymbolicJudge, GAIOptimizer, ACTIVATIONS
except ImportError:
    print("Error: Could not import 'gai_unified'.")
    exit()

# --- CONFIGURATION ---
DATA_PATH = r"d:\Quantum Projects\GAI\data\EColi_Merged_df.csv"
TARGET_ANTIBIOTIC = "CTZ"

# Experiment Settings
NUM_GROUPS = 3           # 3 Variants
JUDGES_PER_GROUP = 1     # 1 Judge per GAI
EPOCHS_PHASE_1 = 3000    # Judge Training (Law Making)
EPOCHS_PHASE_2 = 1000    # Student Training (Law Abiding)
BATCH_SIZE = 128
EVOLUTION_PATIENCE = 50 

# Model Settings
HIDDEN_DIM = 64
DEPTH = 3
NUM_EXPERTS = 3

class SequentialGroup:
    """
    1 Student, 1 Judge. Sequential Training.
    """
    def __init__(self, group_id, input_dim, device='cpu'):
        self.group_id = group_id
        self.device = device
        
        # Student
        self.student = GatedMatrixGGLEN(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=HIDDEN_DIM,
            num_chains=NUM_EXPERTS,
            chain_depth=DEPTH
        ).to(device)
        self.student_opt = torch.optim.Adam(self.student.parameters(), lr=0.001)
        self.student_history = {}
        
        # Judge
        self.judge = SymbolicJudge(
            input_dim=input_dim + 1,
            hidden_dim=HIDDEN_DIM,
            depth=DEPTH
        ).to(device)
        self.judge_opt = torch.optim.Adam(self.judge.parameters(), lr=0.001)
        self.judge_history = {}
        
        # Randomize Judge Init
        ops = [k for k in ACTIVATIONS.keys() if k != 'identity']
        for layer in self.judge.chain.layers:
             if ops: layer.op_name = random.choice(ops)
             
        self.bce = nn.BCEWithLogitsLoss()

    def train_phase_1(self, X_batch, y_real_batch):
        """
        Train Judge ONLY. Student Frozen.
        Judge learns to distinguish Real vs 'Untrained Student'.
        """
        self.student.eval() # Freeze Student logic
        self.judge.train()
        
        with torch.no_grad():
            s_logits = self.student(X_batch)
            s_probs = torch.sigmoid(s_logits)
            
        real_pair = torch.cat([X_batch, y_real_batch], dim=1)
        fake_pair = torch.cat([X_batch, s_probs], dim=1)
        
        self.judge_opt.zero_grad()
        
        # Loss: Real->0, Fake->1
        pred_real = self.judge(real_pair)
        pred_fake = self.judge(fake_pair)
        loss = self.bce(pred_real, torch.zeros_like(pred_real)) + \
               self.bce(pred_fake, torch.ones_like(pred_fake))
               
        loss.backward()
        self.judge_opt.step()
        
        with torch.no_grad():
            sep = (torch.sigmoid(pred_fake).mean() - torch.sigmoid(pred_real).mean()).item()
            
        return loss.item(), sep
        
    def train_phase_2(self, X_batch, y_real_batch):
        """
        Train Student ONLY. Judge Frozen.
        Student learns to Fool Judge + Solve Task.
        """
        self.student.train()
        self.judge.eval() # Freeze Judge logic
        
        self.student_opt.zero_grad()
        
        s_logits = self.student(X_batch)
        s_probs = torch.sigmoid(s_logits)
        
        # A. Task Loss (Ground Truth)
        task_loss = self.bce(s_logits, y_real_batch)
        
        # B. Judge "Law" Loss
        # Judge(Fake) -> 0 (Real)
        fake_pair = torch.cat([X_batch, s_probs], dim=1)
        j_score = self.judge(fake_pair)
        law_loss = self.bce(j_score, torch.zeros_like(j_score))
        
        # Combined
        total_loss = task_loss + 0.1 * law_loss
        
        total_loss.backward()
        self.student_opt.step()
        
        return task_loss.item(), law_loss.item()

    def evolve_judge(self):
        self.judge.evolve_structure(self.judge_history, logger_prefix=f"Grp{self.group_id}-Judge")

    def evolve_student(self):
        self.student.evolve_structure(self.student_history, logger_prefix=f"Grp{self.group_id}-Student")


def load_data():
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET_ANTIBIOTIC not in df.columns:
        raise ValueError(f"{TARGET_ANTIBIOTIC} not in dataset")
    df = df.dropna(subset=[TARGET_ANTIBIOTIC])
    X = df.iloc[:, 14:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_ANTIBIOTIC].astype(str))
    print(f"Data Shape: {X.shape}")
    return X, y

def main():
    device = 'cpu'
    
    # 1. Load Data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_dim = X_train.shape[1]
    
    # 2. Init
    groups = [SequentialGroup(i, input_dim, device) for i in range(NUM_GROUPS)]
    
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_v = torch.FloatTensor(X_test)
    y_v = torch.FloatTensor(y_test).unsqueeze(1)

    print("\n>>> PHASE 1: JUDGE TRAINING (Establishing the Law) <<<")
    # Judges learn to spot untrained students
    for epoch in range(EPOCHS_PHASE_1):
        idx = torch.randperm(X_t.size(0))[:BATCH_SIZE]
        X_b = X_t[idx]
        y_b = y_t[idx]
        
        logs = []
        for grp in groups:
            loss, sep = grp.train_phase_1(X_b, y_b)
            if epoch > 50 and epoch % EVOLUTION_PATIENCE == 0:
                grp.evolve_judge()
            logs.append(sep)
            
        if epoch % 200 == 0:
            print(f"Ep {epoch}: Mean Separation {np.mean(logs):.4f}")

    print("\n>>> PHASE 2: STUDENT TRAINING (Obeying the Law) <<<")
    # Students learn to solve task + obey fixed judge
    for epoch in range(EPOCHS_PHASE_2):
        idx = torch.randperm(X_t.size(0))[:BATCH_SIZE]
        X_b = X_t[idx]
        y_b = y_t[idx]
        
        logs = []
        for grp in groups:
            t_loss, l_loss = grp.train_phase_2(X_b, y_b)
            if epoch > 50 and epoch % EVOLUTION_PATIENCE == 0:
                grp.evolve_student()
            
            # Validation Acc
            if epoch % 200 == 0:
                with torch.no_grad():
                    logits = grp.student(X_v)
                    acc = ((torch.sigmoid(logits) > 0.5) == y_v).float().mean().item()
                logs.append(acc)
                
        if epoch % 200 == 0:
            print(f"Ep {epoch}: Mean Val Acc {np.mean(logs):.2%}")

    # Export
    print("\n>>> EXPORTING RESULTS <<<")
    final_summary = []
    for grp in groups:
        sample_formula = grp.student.chains[0].export_formula([f"x{i}" for i in range(input_dim)])
        judge_formula = grp.judge.chain.export_formula([f"f{i}" for i in range(input_dim+1)])
        
        summary = {
            "group_id": grp.group_id,
            "student_formula": sample_formula[:200],
            "judge_formula": judge_formula[:200]
        }
        final_summary.append(summary)
        print(f"Grp {grp.group_id}")
        print(f"  Judge: {judge_formula[:100]}...")
        print(f"  Student: {sample_formula[:100]}...")

    with open('ctz_k3_sequential_results.json', 'w') as f:
        json.dump(final_summary, f, indent=2)

if __name__ == "__main__":
    main()
