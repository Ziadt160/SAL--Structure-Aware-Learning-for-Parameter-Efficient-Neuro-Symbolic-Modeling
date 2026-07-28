
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import random
import argparse
from gai_unified import MatrixGGLEN, GAIOptimizer
from gai_moe import GatedMatrixGGLEN

# --- CONFIGURATION (DEFAULTS) ---
DEFAULT_ANTIBIOTICS = ['TZP']
DEFAULT_EPOCHS = 3000
HIDDEN_DIM = 32

def parse_args():
    parser = argparse.ArgumentParser(description="GAI MoE Benchmark runner")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs')
    parser.add_argument('--antibiotics', nargs='+', default=DEFAULT_ANTIBIOTICS, help='List of antibiotics')
    parser.add_argument('--no_annealing', action='store_true', help='Disable annealing')
    parser.add_argument('--strategy', type=str, default='importance', choices=['importance', 'random'])
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--l1', type=float, default=0.0, help='L1 Regularization')
    parser.add_argument('--chains', type=int, default=3, help='Number of expert chains')
    args = parser.parse_args()
    return args

def analyze_gate_behavior(model, X_data, y_data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Get the Gate's opinion (Probabilities for each chain)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_data)
        # We only need the gate output
        gate_logits = model.gate(X_tensor)
        gate_weights = torch.nn.functional.softmax(gate_logits, dim=1).numpy()
    
    # 2. Plot
    plt.figure(figsize=(12, 6))
    
    # Sort samples by their target class (Sensitive vs Resistant) to see if Gate correlates with class
    sort_idx = np.argsort(y_data)
    sorted_weights = gate_weights[sort_idx]
    sorted_y = y_data[sort_idx]
    
    # Plot heatmap of Expert Usage
    sns.heatmap(sorted_weights.T, cmap="viridis", cbar_kws={'label': 'Expert Confidence'})
    plt.xlabel("Samples (Sorted by Resistance: Left=Sensitive, Right=Resistant)")
    plt.ylabel("Expert Chain ID")
    plt.title("Did the Manager Specialize? (Gating Weights Analysis)")
    
    # Add a line showing where the class flips
    flip_idx = np.where(sorted_y == 1)[0][0]
    plt.axvline(x=flip_idx, color='red', linestyle='--', label='Class Boundary (S -> R)')
    plt.legend()
    
    plt.savefig('gate_analysis.png')
    print("Saved Gate Analysis to gate_analysis.png")

def run_moe_benchmark():
    args = parse_args()
    print(f"Starting MoE Comparative Benchmark...", flush=True)
    print(f"Config: Antibiotics={args.antibiotics}, Epochs={args.epochs}, Chains={args.chains}", flush=True)
    print(f"        Annealing={'OFF' if args.no_annealing else 'ON'}, Dropout={args.dropout}, L1={args.l1}", flush=True)
    
    print("Loading Data...", flush=True)
    df = pd.read_csv('data/EColi_Merged_df.csv')
    feature_cols = [c for c in df.columns if c.startswith('group_')]
    
    results = []

    for ab in args.antibiotics:
        print(f"\n--- Benchmarking {ab} ---", flush=True)
        sub_df = df.dropna(subset=[ab])
        y = sub_df[ab].map({'R': 1, 'S': 0})
        # Clean NaNs in target
        valid_idx = y.notna()
        sub_df = sub_df[valid_idx]
        y = y[valid_idx].values
        
        X = sub_df[feature_cols].values.astype(np.float32)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. XGBoost (Baseline)
        print(f"  Training XGBoost...", flush=True)
        xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
        xgb_clf.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb_clf.predict(X_test))
        print(f"    XGBoost Accuracy: {xgb_acc:.2%}", flush=True)
        
        # 2. Standard GAI
        print(f"  Training Standard GAI...", flush=True)
        gai_std = MatrixGGLEN(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, num_chains=args.chains, dropout=args.dropout)
        trainer_std = GAIOptimizer(gai_std, patience=30, grace_period=10, 
                                   initial_temp=0.5, use_annealing=not args.no_annealing,
                                   mutation_strategy=args.strategy, l1_lambda=args.l1)
        # Suppress verbose logs slightly by running fewer epochs per print if possible, but fit doesn't support that.
        # We rely on fit's internal logging.
        gai_std_preds = trainer_std.fit(X_train, y_train, X_test, y_test, epochs=args.epochs, name="STD")
        gai_std_acc = accuracy_score(y_test, gai_std_preds)
        print(f"    Standard GAI Accuracy: {gai_std_acc:.2%}", flush=True)
        
        # 3. MoE GAI
        print(f"  Training Gated MoE GAI...", flush=True)
        gai_moe = GatedMatrixGGLEN(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, num_chains=args.chains, dropout=args.dropout)
        trainer_moe = GAIOptimizer(gai_moe, patience=30, grace_period=10, 
                                   initial_temp=0.5, use_annealing=not args.no_annealing,
                                   mutation_strategy=args.strategy, l1_lambda=args.l1)
        gai_moe_preds = trainer_moe.fit(X_train, y_train, X_test, y_test, epochs=args.epochs, name="MoE")
        gai_moe_acc = accuracy_score(y_test, gai_moe_preds)
        print(f"    MoE GAI Accuracy: {gai_moe_acc:.2%}", flush=True)
        
        results.append({
            'Antibiotic': ab,
            'Model': 'XGBoost',
            'Accuracy': xgb_acc
        })
        results.append({
            'Antibiotic': ab,
            'Model': 'Standard GAI',
            'Accuracy': gai_std_acc
        })
        results.append({
            'Antibiotic': ab,
            'Model': 'MoE GAI',
            'Accuracy': gai_moe_acc
        })

        if args.chains > 1:
            analyze_gate_behavior(gai_moe, X_test, y_test)

    # --- PLOTTING ---
    res_df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(res_df.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x='Antibiotic', y='Accuracy', hue='Model', palette='deep')
    plt.title("Comparative Benchmark: MoE GAI vs Standard GAI vs XGBoost")
    plt.ylim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig('moe_benchmark.png')
    print("\nSaved plot to 'moe_benchmark.png'")

if __name__ == "__main__":
    run_moe_benchmark()
