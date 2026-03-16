import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import copy
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# Start Import Fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.adaptive_neural_model import MatrixGGLEN
# End Import Fix

# --- CONFIGURATION ---
torch.manual_seed(42)
np.random.seed(42)
HIDDEN_DIM = 32
BATCH_SIZE = 64
TOTAL_EPOCHS = 1000 
INITIAL_TEMP = 0.5

# --- 3. DATA LOADING ---
def load_ecoli_data():
    print("Loading E. Coli Data...")
    # Fix Path to be relative to the script or project root
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    data_path = os.path.join(base_path, 'data', 'EColi_Merged_df.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return None, None, None, None

    df = pd.read_csv(data_path)
    
    df = df.dropna(subset=['CTZ'])
    y = df['CTZ'].map({'R': 1, 'S': 0})
    valid_mask = y.notna()
    df = df[valid_mask]
    y = y[valid_mask]
    
    feature_cols = [c for c in df.columns if c.startswith('group_')]
    X = df[feature_cols].values.astype(np.float32)
    y = y.values.astype(np.float32).reshape(-1, 1)
    
    print(f"  Data Shape: X={X.shape}, y={y.shape}")
    print(f"  Class Balance: {y.mean():.2%} Positives (Resistance)")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TRAINING ---
def train_ecoli_gai():
    X_train, X_test, y_train, y_test = load_ecoli_data()
    if X_train is None: return None, None
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    input_dim = X_train.shape[1]
    
    # Use Core MatrixGGLEN
    model = MatrixGGLEN(input_dim=input_dim, output_dim=1, num_chains=2, chain_depth=3, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    mutation_history = {}
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = 50 
    
    backup_state = None
    backup_chains = None
    just_mutated = False
    grace_period = 20
    
    active_op = None
    active_node = None
    prev_val_acc_at_mutation = 0.0
    
    print(f"Task: E. Coli Resistance Prediction (CTZ)")
    print("Mode: Gradient-Guided Structural Search")
    print("-" * 50)

    for epoch in range(TOTAL_EPOCHS):
        progress = epoch / TOTAL_EPOCHS
        current_temp = INITIAL_TEMP * (1 - progress)
        
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        history.append(curr_loss)
        
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t)
            test_acc = ((torch.sigmoid(test_preds) > 0.5) == y_test_t).float().mean().item()
            train_acc = ((torch.sigmoid(preds) > 0.5) == y_train_t).float().mean().item()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {curr_loss:.4f} | Train Acc {train_acc:.2%} | Val Acc {test_acc:.2%} | Temp {current_temp:.4f}")

        if test_acc > best_val_acc:
            best_val_acc = test_acc
            epochs_no_improve = 0
            backup_state = copy.deepcopy(model.state_dict())
            backup_chains = copy.deepcopy(model.chains)
        else:
            epochs_no_improve += 1
            
        if just_mutated and epochs_no_improve > grace_period:
            delta = test_acc - prev_val_acc_at_mutation
            
            if delta > 0: 
                accept_prob = 1.0 
            else:
                 if current_temp > 0.01:
                    accept_prob = math.exp(delta / current_temp)
                 else:
                    accept_prob = 0.0
            
            if random.random() < accept_prob:
                print(f"  [Result] '{active_op.upper()}' ACCEPTED. Val Acc: {test_acc:.2%} (Delta: {delta:.2%})")
                just_mutated = False
                epochs_no_improve = 0 
            else:
                print(f"  [Result] '{active_op.upper()}' REJECTED. Val Acc: {test_acc:.2%} (Delta: {delta:.2%})")
                model.load_state_dict(backup_state)
                model.chains = copy.deepcopy(backup_chains)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                c, l = active_node
                if (c,l) not in mutation_history: mutation_history[(c,l)] = set()
                mutation_history[(c,l)].add(active_op)
                
                just_mutated = False
                epochs_no_improve = 0

        # Create Mutation?
        if epochs_no_improve > patience and not just_mutated:
            print(f"\n--- Stagnation at Val Acc {best_val_acc:.2%} ---")
            
            backup_state = copy.deepcopy(model.state_dict())
            backup_chains = copy.deepcopy(model.chains)
            prev_val_acc_at_mutation = test_acc
            
            c_id, l_id = model.evolve_structure(mutation_history)
            
            if c_id is not None:
                active_node = (c_id, l_id)
                active_op = model.chains[c_id].layers[l_id].op_name
                just_mutated = True
                epochs_no_improve = 0
                optimizer = optim.Adam(model.parameters(), lr=0.001) 
            else:
                print("No more mutations available.")
                break
        
        if test_acc > 0.999:
            print("Perfect Validation Accuracy achieved.")
            break

    print(f"Best Validation Accuracy: {best_val_acc:.2%}")
    return model, history

if __name__ == "__main__":
    model, history = train_ecoli_gai()
