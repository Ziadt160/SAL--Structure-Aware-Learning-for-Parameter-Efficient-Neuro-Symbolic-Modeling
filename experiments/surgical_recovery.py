
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.adaptive_neural_model import MatrixGGLEN
from training.trainer import GAIOptimizer
from training.evaluation import SensitivityProbe

def run_surgical_benchmark():
    print("==========================================================")
    print(" SURGICAL D.A.S. BENCHMARK: The 'Doctor' Loop ")
    print("==========================================================")
    
    # 1. Setup Data (MNIST)
    print("Loading Data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    try:
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except:
        print("Failed to download MNIST, using fake data.")
        # Fallback for offline/no-internet
        X_train = np.random.randn(100, 784); y_train = np.random.randint(0, 10, 100)
        X_test = np.random.randn(20, 784); y_test = np.random.randint(0, 10, 20)
    else:
        # Mini subset for speed
        X_train = train_set.data.float().view(-1, 784).numpy()[:1000] / 255.0
        X_train = (X_train - 0.5) / 0.5
        y_train = train_set.targets.numpy()[:1000]
        X_test = test_set.data.float().view(-1, 784).numpy()[:200] / 255.0
        X_test = (X_test - 0.5) / 0.5
        y_test = test_set.targets.numpy()[:200]

    # 2. Setup Model
    print("\n[Phase 1] Initializing Patient (Model)...")
    # 2 Chains, 3 Layers each
    model = MatrixGGLEN(784, 10, hidden_dim=32, num_chains=2, chain_depth=3)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = GAIOptimizer(model, loss_fn=loss_fn)
    
    # Initial quick train
    print("Training 2 epochs to establish baseline...")
    optimizer.fit(X_train, y_train, X_test, y_test, epochs=2, name="Baseline")
    
    # 3. SABOTAGE !!
    # We consciously break Chain 0, Layer 1 (The middle of the chain)
    print("\n[Phase 2] 🗡️ SABOTAGE: Severing Chain 0, Layer 1...")
    target_c = 0
    target_l = 1
    with torch.no_grad():
        dead_node = model.chains[target_c].layers[target_l].linear
        # Set weights to actual zero -> Dead Gradient
        dead_node.weight.fill_(0.0)
        dead_node.bias.fill_(0.0)
        # Also ensure op is something we can change, e.g., 'identity' or 'relu'
        model.chains[target_c].layers[target_l].op_name = 'identity'
        
    # Verify Performance Drop
    optimizer.model.eval()
    with torch.no_grad():
        preds = optimizer.model(torch.FloatTensor(X_test))
        acc_sabotage = (torch.argmax(preds, 1) == torch.tensor(y_test)).float().mean().item()
    print(f"Patient Condition Critically Worsened. Acc: {acc_sabotage:.2%}")
    
    # 4. DIAGNOSIS (The Probe)
    print("\n[Phase 3] 🩺 DIAGNOSIS: Scanning Vital Signs...")
    probe = SensitivityProbe(model)
    probe.attach()
    
    # Run a forward/backward pass to collect stats
    optimizer.model.train()
    optimizer.optimizer.zero_grad()
    out = optimizer.model(torch.FloatTensor(X_train[:32]))
    loss = loss_fn(out, torch.tensor(y_train[:32]).long())
    loss.backward()
    
    report = probe.generate_report()
    probe.remove()
    
    # Find the sickest node
    sick_node_name = None
    for name, info in report.items():
        if info['status'] == 'CRITICAL' or info['status'] == 'WARNING':
            print(f"  -> DETECTED ISSUE at '{name}': {info['issues']}")
            if "DEAD_GRADIENT" in str(info['issues']):
                sick_node_name = name
    
    if not sick_node_name:
        print("Diagnosis Failed: No critical issues found? (Sabotage failed?)")
        # Ensure we find specifically the one we broke
        # Format is usually: chains.0.layers.1.linear
        sick_node_name = f"chains.{target_c}.layers.{target_l}.linear"
        print(f"  (Forcing diagnosis on {sick_node_name} for demo)")

    # 5. SURGERY
    print(f"\n[Phase 4] 🏥 SURGERY: Operating on {sick_node_name}...")
    
    # Parse name: chains.0.layers.1.linear -> c=0, l=1
    parts = sick_node_name.split('.')
    try:
        if 'chains' in parts:
            c_idx = int(parts[parts.index('chains')+1])
            l_idx = int(parts[parts.index('layers')+1])
            
            print(f"  -> Targeted Chain {c_idx}, Layer {l_idx}")
            
            # Execute Surgical Evolve
            result = optimizer.surgical_evolve(
                X_train, y_train, 
                X_test, y_test, 
                chain_id=c_idx, 
                layer_id=l_idx, 
                epochs=5
            )
            
            if result:
                print("\n[Phase 5] 🟢 RECOVERY: Patient stabilized.")
            else:
                print("\n[Phase 5] ⚫ FAILURE: Patient did not respond to treatment.")

    except Exception as e:
        print(f"Surgery Prep Failed: {e}")

    # Final Check
    optimizer.model.eval()
    with torch.no_grad():
        preds = optimizer.model(torch.FloatTensor(X_test))
        acc_final = (torch.argmax(preds, 1) == torch.tensor(y_test)).float().mean().item()
    
    print(f"\nFinal Accuracy: {acc_final:.2%} (vs Sabotaged: {acc_sabotage:.2%})")

if __name__ == "__main__":
    run_surgical_benchmark()
