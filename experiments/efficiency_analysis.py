import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import copy
import traceback

# Ensure GAI modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.adaptive_neural_model import MatrixGGLEN
from training.trainer import GAIOptimizer

# --- CONFIGURATION ---
SEARCH_EPOCHS = 50
SWEEP_EPOCHS = 50
INITIAL_HIDDEN = 64
DEPTH = 3
CHAINS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- DATA LOADING ---
def get_data():
    print("Loading MNIST Data...", end=" ")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Flatten for GAI
    X_train = train_set.data.float().view(-1, 784).numpy() / 255.0 
    X_train = (X_train - 0.5) / 0.5
    y_train = train_set.targets.numpy()
    
    X_test = test_set.data.float().view(-1, 784).numpy() / 255.0
    X_test = (X_test - 0.5) / 0.5
    y_test = test_set.targets.numpy()
    print("Done.")
    return X_train, y_train, X_test, y_test

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment():
    X_train, y_train, X_test, y_test = get_data()
    
    print("\n=======================================")
    print(" PHASE 1: Evolutionary Structure Search")
    print("=======================================")
    
    # 1. Initialize Base Model for Search
    search_model = MatrixGGLEN(
        input_dim=784, 
        output_dim=10, 
        hidden_dim=INITIAL_HIDDEN, 
        num_chains=CHAINS,
        chain_depth=DEPTH
    ).to(DEVICE)
    
    optimizer = GAIOptimizer(
        search_model, 
        patience=10, 
        grace_period=5, 
        initial_temp=0.5, 
        loss_fn=nn.CrossEntropyLoss()
    )
    
    print(f"Searching for best structure (Epochs: {SEARCH_EPOCHS}, Hidden: {INITIAL_HIDDEN})...")
    _, history = optimizer.fit(
        X_train, y_train, 
        X_test, y_test, 
        epochs=SEARCH_EPOCHS, 
        name="Search"
    )
    
    # 2. Extract Best Structure from the Optimizer's best saved state
    if optimizer.best_ever_state:
        search_model.load_state_dict(optimizer.best_ever_state)
    
    best_structure = search_model.get_structure()
    best_acc = max(history) if history else 0
    
    print(f"\n[Phase 1 Complete] Best Search Accuracy: {best_acc:.4f} (Neg Loss)")
    print(f"Found Structure: {best_structure}")
    
    print("\n=======================================")
    print(" PHASE 2: Parameter Efficiency Sweep")
    print("=======================================")
    
    hidden_dims = [128, 96, 64, 48, 32, 24, 16, 12, 8]
    results = [] 
    
    for h_dim in hidden_dims:
        print(f"\n--- Testing Reduced Model (Hidden Dim: {h_dim}) ---")
        
        # Instantiate with FIXED structure
        model = MatrixGGLEN(
            input_dim=784,
            output_dim=10,
            hidden_dim=h_dim,
            num_chains=CHAINS,
            chain_depth=DEPTH,
            fixed_structure=best_structure
        ).to(DEVICE)
        
        params = count_parameters(model)
        
        # Train without evolution (patience > epochs)
        opt = GAIOptimizer(
            model,
            patience=999, # Disable evolution
            lr=0.001,
            loss_fn=nn.CrossEntropyLoss()
        )
        
        try:
            _, hist = opt.fit(
                X_train, y_train,
                X_test, y_test,
                epochs=SWEEP_EPOCHS,
                name=f"Sweep-H{h_dim}"
            )
            
            final_acc = max(hist) if hist else -100
            
            # Quick Evaluation of Accuracy
            model.eval()
            with torch.no_grad():
                out = model(torch.FloatTensor(X_test).to(DEVICE))
                preds = torch.argmax(out, dim=1).cpu().numpy()
                acc = np.mean(preds == y_test)
                
            print(f"-> Hidden: {h_dim} | Params: {params} | Accuracy: {acc:.4%}")
            results.append({
                'hidden_dim': h_dim,
                'params': params,
                'accuracy': acc,
                'neg_loss': final_acc
            })
        except Exception as e:
            print(f"-> Hidden: {h_dim} Failed: {e}")
            traceback.print_exc()

    print("\n=======================================")
    print(" RESULTS")
    print("=======================================")
    print(f"{'Hidden':<10} | {'Params':<10} | {'Accuracy':<10}")
    print("-" * 35)
    for r in results:
        print(f"{r['hidden_dim']:<10} | {r['params']:<10} | {r['accuracy']:.4%}")
        
    # Plotting
    res_dicts = sorted(results, key=lambda x: x['params'])
    if not res_dicts:
        print("No results to plot.")
        return

    params_x = [r['params'] for r in res_dicts]
    acc_y = [r['accuracy'] for r in res_dicts]
    
    plt.figure(figsize=(10, 6))
    plt.plot(params_x, acc_y, marker='o', linestyle='-', color='b')
    plt.title(f"Parameter Efficiency: Best GAI Structure\n(Ops: {best_structure})")
    plt.xlabel("Number of Parameters")
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.xscale('log') 
    
    # Annotate points
    for i, txt in enumerate([r['hidden_dim'] for r in res_dicts]):
        plt.annotate(f"H={txt}", (params_x[i], acc_y[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.savefig('nas_efficiency_sweep.png')
    print("\nSaved plot to nas_efficiency_sweep.png")

if __name__ == "__main__":
    run_experiment()
