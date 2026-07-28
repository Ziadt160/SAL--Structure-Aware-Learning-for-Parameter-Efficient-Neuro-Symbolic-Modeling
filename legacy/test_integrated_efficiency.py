import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import os

# Ensure GAI modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from modules.core import MatrixGGLEN, GAIOptimizer

# --- CONFIGURATION ---
SEARCH_EPOCHS = 5 # Short for verification
INITIAL_HIDDEN = 64
DEPTH = 3
CHAINS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def run_integrated_test():
    X_train, y_train, X_test, y_test = get_data()
    
    print("\n=======================================")
    print(" TEST: Integrated Efficiency Optimization")
    print("=======================================")
    
    model = MatrixGGLEN(
        input_dim=784, 
        output_dim=10, 
        hidden_dim=INITIAL_HIDDEN, 
        num_chains=CHAINS,
        chain_depth=DEPTH
    ).to(DEVICE)
    
    optimizer = GAIOptimizer(
        model, 
        patience=3, 
        grace_period=2, 
        initial_temp=0.5, 
        loss_fn=nn.CrossEntropyLoss()
    )
    
    # Enable the new flag!
    print(f"Starting Training (optimize_efficiency=True)...")
    _, history = optimizer.fit(
        X_train, y_train, 
        X_test, y_test, 
        epochs=SEARCH_EPOCHS, 
        name="IntegratedTest",
        optimize_efficiency=True
    )
    
    final_hidden = optimizer.model.hparams_dict['hidden_dim']
    print(f"\nFinal Model Hidden Dim: {final_hidden}")
    print(f"Original Hidden Dim: {INITIAL_HIDDEN}")
    
    if final_hidden < INITIAL_HIDDEN:
        print("SUCCESS: Model was reduced!")
    else:
        print("RESULT: Model was NOT reduced (either performance dropped or original was already optimal).")

if __name__ == "__main__":
    run_integrated_test()
