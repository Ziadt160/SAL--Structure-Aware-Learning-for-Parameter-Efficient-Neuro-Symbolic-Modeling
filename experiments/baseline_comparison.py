import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import random
import argparse
import sys
import os

# Dynamic path resolution to support running from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.adaptive_neural_model import MatrixGGLEN, GatedMatrixGGLEN
from training.trainer import GAIOptimizer

# --- CONFIGURATION (DEFAULTS) ---
DEFAULT_ANTIBIOTICS = ['CTZ', 'TZP']
DEFAULT_EPOCHS = 10000
TRIALS = 3
HIDDEN_DIM = 32

def parse_args():
    parser = argparse.ArgumentParser(description="GAI Benchmark runner")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs per antibiotic')
    parser.add_argument('--antibiotics', nargs='+', default=DEFAULT_ANTIBIOTICS, help='List of antibiotics to benchmark')
    parser.add_argument('--temp', type=float, default=0.5, help='Initial temperature for annealing')
    parser.add_argument('--no_annealing', action='store_true', help='Disable simulated annealing (Strict Hill Climbing)')
    parser.add_argument('--strategy', type=str, default='importance', choices=['importance', 'random'], help='Evolution strategy: "importance" (default) or "random" (slow method)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (0.0 to 1.0)')
    parser.add_argument('--l1', type=float, default=0.0, help='L1 Regularization strength (e.g. 0.0001)')
    args = parser.parse_args()
    return args

# --- MAIN BENCHMARK ---
def run_benchmark():
    args = parse_args()
    print(f"Starting Benchmark (GAI Unified Class Version)...", flush=True)
    print(f"Config: Antibiotics={args.antibiotics}, Epochs={args.epochs}, Temp={args.temp}", flush=True)
    print(f"        Annealing={'OFF' if args.no_annealing else 'ON'}, Strategy='{args.strategy}'", flush=True)
    print(f"        Dropout={args.dropout}, L1_Lambda={args.l1}", flush=True)
    
    print("Loading Data (1-2 mins)...", flush=True)
    print("Loading Data (1-2 mins)...", flush=True)
    # Robust data path handling
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_path = os.path.join(project_root, 'data', 'EColi_Merged_df.csv')
    
    if not os.path.exists(data_path):
        # Fallback if structure is different or running differently
        data_path = 'data/EColi_Merged_df.csv'
        
    df = pd.read_csv(data_path)
    print("Data Loaded.", flush=True)
    
    feature_cols = [c for c in df.columns if c.startswith('group_')]
    
    results = []
    
    print(f"{' Antibiotic ':^12} | {' RF ':^8} | {' XGB ':^8} | {' GAI ':^8}", flush=True)
    print("-" * 46, flush=True)

    for ab in args.antibiotics:
        if ab not in df.columns: continue
        
        sub_df = df.dropna(subset=[ab])
        y = sub_df[ab].map({'R': 1, 'S': 0})
        sub_df = sub_df[y.notna()]
        y = y[y.notna()].values
        
        # if len(y) < 50: continue 
        
        X = sub_df[feature_cols].values.astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. XGBoost
        xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        xgb_clf.fit(X_train, y_train)
        xgb_preds = xgb_clf.predict(X_test)
        
        # 3. GAI (Unified Class)
        model = GatedMatrixGGLEN(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, dropout=args.dropout)
        use_annealing = not args.no_annealing
        trainer = GAIOptimizer(model, patience=30, grace_period=10, 
                               initial_temp=args.temp, 
                               use_annealing=use_annealing,
                               mutation_strategy=args.strategy,
                               l1_lambda=args.l1)
        gai_preds, _ = trainer.fit(X_train, y_train, X_test, y_test, epochs=args.epochs, name=ab)
        
        # --- Print Architecture ---
        print(f"\n[{ab}] Final Evolved Architecture (Gated MoE):", flush=True)
        # Create dummy var names
        input_vars = [f"x{i}" for i in range(X_train.shape[1])]
        
        if hasattr(model, 'chains'):
            for i, chain in enumerate(model.chains):
                print(f"  Expert {i}:", flush=True)
                # Print Ops matching standard extraction style
                ops = [layer.op_name for layer in chain.layers if hasattr(layer, 'op_name')]
                print(f"    Ops: {ops}", flush=True)
                
                # Print Formula
                try:
                    # We might have too many input vars for clean printing if dimensionality is high (e.g. 100+)
                    # Usage of export_formula might be verbose. Truncating vars if needed?
                    # E. coli data has ~200+ features?
                    # Let's try exporting with generic "x" array if too large
                    if len(input_vars) > 10:
                        # Just print structure, formula might be huge
                        print(f"    Formula: (Too many inputs {len(input_vars)} for symbolic render)", flush=True)
                    else:
                        formula = chain.export_formula(input_vars)
                        print(f"    Formula: {formula}", flush=True)
                except Exception as e:
                    print(f"    Formula Error: {e}", flush=True)
        
        # Metrics
        gai_preds_binary = (gai_preds > 0.5).astype(int)
        
        for name, preds in [('XGB', xgb_preds), ('GAI', gai_preds_binary)]:
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            results.append({'Antibiotic': ab, 'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec})
        
        curr_res = {r['Model']: r['Accuracy'] for r in results if r['Antibiotic'] == ab}
        print(f"{ab:^12} | {curr_res.get('RF',0):.1%} | {curr_res.get('XGB',0):.1%} | {curr_res.get('GAI',0):.1%}", flush=True)

    # --- PLOTTING ---
    res_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    
    sns.barplot(data=res_df, x='Antibiotic', y='Accuracy', hue='Model', ax=axes[0])
    axes[0].set_title('Accuracy')
    axes[0].legend(loc='upper right')
    
    sns.barplot(data=res_df, x='Antibiotic', y='Recall', hue='Model', ax=axes[1])
    axes[1].set_title('Recall')
    
    sns.barplot(data=res_df, x='Antibiotic', y='Precision', hue='Model', ax=axes[2])
    axes[2].set_title('Precision')
    
    plt.tight_layout()
    plt.savefig('comparative_benchmark.png')
    print("\nBenchmark Plot saved to 'comparative_benchmark.png'", flush=True)

if __name__ == "__main__":
    run_benchmark()
