import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Fix OpenMP conflict
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure GAI modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from modules.core import MatrixGGLEN, GAIOptimizer

# --- CONFIGURATION ---
EPOCHS = 20
DEVICE = 'cpu' # Force CPU for stability

# --- DATA ---
def get_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    X_train = train_set.data.float().view(-1, 784).numpy() / 255.0 
    X_train = (X_train - 0.5) / 0.5
    y_train = train_set.targets.numpy()
    
    X_test = test_set.data.float().view(-1, 784).numpy() / 255.0
    X_test = (X_test - 0.5) / 0.5
    y_test = test_set.targets.numpy()
    return X_train, y_train, X_test, y_test

# --- BASELINE MODEL ---
class StandardMLP(nn.Module):
    """A standard human-designed dense network."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

def train_manual(model, X_train, y_train, X_test, y_test, epochs):
    """Manual training loop to simulate standard workflow."""
    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.tensor(y_train).long().to(DEVICE)
    X_v = torch.FloatTensor(X_test).to(DEVICE)
    y_v = torch.tensor(y_test).long().to(DEVICE)
    
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()
    
    total_time = time.time() - start_time
    
    # Final Acc
    model.eval()
    with torch.no_grad():
        out = model(X_v)
        preds = torch.argmax(out, dim=1)
        acc = (preds == y_v).float().mean().item()
        
    return total_time, acc

def run_benchmark():
    X_train, y_train, X_test, y_test = get_data()
    
    print("====================================================")
    print(" 1. DISCOVERY PHASE (Finding the 'Right' Model)")
    print("====================================================")
    
    # Determine the 'Evoth' Model using our optimizer
    # We use a quick search for demonstration
    print("Running Evoth Optimization...")
    search_model = MatrixGGLEN(784, 10, hidden_dim=64, num_chains=2, chain_depth=3).to(DEVICE)
    optimizer = GAIOptimizer(search_model, patience=5, initial_temp=0.5, loss_fn=nn.CrossEntropyLoss())
    
    # Logic: fit() with optimize_efficiency=True will finalize self.model as the best efficient one
    try:
        optimizer.fit(X_train, y_train, X_test, y_test, epochs=10, optimize_efficiency=True)
    except Exception as e:
        print(f"Optimization Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # This is the "Right Model" found from first principles
    try:
        evoth_model_structure = optimizer.model.get_structure()
        evoth_hidden_dim = optimizer.model.hparams_dict['hidden_dim']
        print(f"Evoth Discovered Model: Hidden={evoth_hidden_dim}, Structure={evoth_model_structure}")
    except Exception as e:
        print(f"Failed to extract structure: {e}")
        return

    print("\n====================================================")
    print(" 2. HEAD-TO-HEAD TRAINING BENCHMARK")
    print("====================================================")
    
    # --- A. BASELINE (Standard Large Model) ---
    print(f"\n[A] Training Baseline StandardMLP (128x3)...")
    baseline_model = StandardMLP()
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    base_time, base_acc = train_manual(baseline_model, X_train, y_train, X_test, y_test, EPOCHS)
    print(f" -> Time: {base_time:.2f}s | Acc: {base_acc:.2%} | Params: {baseline_params}")

    # --- B. EVOTH (Discovered Efficient Model) ---
    # We create a fresh instance of the discovered model to train from scratch (fair comparison)
    # Using the exact same number of epochs
    print(f"\n[B] Training Evoth Efficient Model (H={evoth_hidden_dim})...")
    
    # Re-instantiate based on discovery
    evoth_final = MatrixGGLEN(
        784, 10, 
        hidden_dim=evoth_hidden_dim, 
        fixed_structure=evoth_model_structure
    )
    evoth_params = sum(p.numel() for p in evoth_final.parameters())
    evoth_time, evoth_acc = train_manual(evoth_final, X_train, y_train, X_test, y_test, EPOCHS)
    print(f" -> Time: {evoth_time:.2f}s | Acc: {evoth_acc:.2%} | Params: {evoth_params}")

    # --- RESULTS ---
    print("\n====================================================")
    print(" FINAL RESULTS")
    print("====================================================")
    
    time_reduction = (base_time - evoth_time) / base_time * 100
    param_reduction = (baseline_params - evoth_params) / baseline_params * 100
    
    print(f"Baseline Time: {base_time:.2f}s")
    print(f"Evoth Time:    {evoth_time:.2f}s")
    print(f"-> TIME REDUCTION: {time_reduction:.2f}%")
    print(f"-> PARAM REDUCTION: {param_reduction:.2f}%")
    
    # Plot
    labels = ['Baseline (Standard)', 'Evoth (Optimized)']
    times = [base_time, evoth_time]
    accs = [base_acc * 100, evoth_acc * 100]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Bar plot for Time
    color = 'tab:blue'
    ax1.set_ylabel('Training Time (s)', color=color)
    ax1.bar(labels, times, color=color, alpha=0.6, width=0.4, label='Time')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Line plot for Accuracy
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(labels, accs, color=color, marker='o', linewidth=2, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)
    
    plt.title(f"Cost Reduction Benchmark\n(Same {EPOCHS} Epochs Training)")
    plt.savefig('cost_reduction_benchmark.png')
    print("Saved chart to cost_reduction_benchmark.png")

if __name__ == "__main__":
    run_benchmark()
