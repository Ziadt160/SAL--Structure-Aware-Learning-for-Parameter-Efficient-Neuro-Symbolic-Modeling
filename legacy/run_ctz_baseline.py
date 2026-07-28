import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gai_unified import GatedMatrixGGLEN, GAIOptimizer

DATA_PATH = r"d:\Quantum Projects\GAI\data\EColi_Merged_df.csv"
TARGET_ANTIBIOTIC = "CTZ"

NUM_RUNS = 3
EPOCHS = 500
HIDDEN_DIM = 64
DEPTH = 3
NUM_EXPERTS = 3

def load_data():
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET_ANTIBIOTIC not in df.columns: raise ValueError(f"{TARGET_ANTIBIOTIC} missing")
    df = df.dropna(subset=[TARGET_ANTIBIOTIC])
    X = df.iloc[:, 14:].apply(pd.to_numeric, errors='coerce').fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_ANTIBIOTIC].astype(str))
    return X, y

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_dim = X_train.shape[1]
    
    results = []
    
    def print_tools(mdl, label):
        print(f"\n--- {label} Tools ---")
        for i, chain in enumerate(mdl.chains):
            ops = [layer.op_name for layer in chain.layers]
            print(f"  Expert {i}: {ops}")
            
    for i in range(NUM_RUNS):
        print(f"\n>>> Baseline Run {i+1}/{NUM_RUNS} <<<")
        model = GatedMatrixGGLEN(input_dim, 1, HIDDEN_DIM, NUM_EXPERTS, DEPTH, dropout=0.01)
        
        print_tools(model, "INITIAL")
        
        # Use Standard GAIOptimizer (defaults to BCE/Adam/Evolution)
        opt = GAIOptimizer(model, lr=0.001, task_type='classification', patience=50, mutation_strategy="random")
        
        final_preds = opt.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS, name=f"Baseline-{i}")
        
        print_tools(model, "FINAL")
        
        # Eval
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
        with torch.no_grad():
            logits = model(torch.FloatTensor(X_test))
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y_test_t).float().mean().item()
            
        # formula = model.chains[0].export_formula([f"x{k}" for k in range(input_dim)])
        
        results.append({
            "run_id": i,
            "accuracy": acc,
            # "formula_sample": formula[:200]
        })
        print(f"Run {i} Final Acc: {acc:.2%}")
        
    with open('ctz_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
