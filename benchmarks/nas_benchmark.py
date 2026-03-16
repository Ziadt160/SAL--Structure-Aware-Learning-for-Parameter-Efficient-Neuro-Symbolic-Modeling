import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# We keep these same as GAI for fair comparison
HIDDEN_DIM = 32
N_BITS = 8  

ACTIVATIONS = {
    'identity': lambda x: x,
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'sin': lambda x: torch.sin(x * torch.pi),
    'gaussian': lambda x: torch.exp(-x**2),
    'square': lambda x: x**2,
}

class MatrixSymbolicNode(nn.Module):
    def __init__(self, in_dim, out_dim, op_name='identity'):
        super().__init__()
        self.op_name = op_name
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        z = self.linear(x)
        if self.op_name in ACTIVATIONS:
            act = ACTIVATIONS[self.op_name]
            out = act(z) if isinstance(act, nn.Module) else act(z)
        else:
            out = z 
        return out

    def mutate(self):
        # Random mutation (NAS/ES style) - No gradient guidance
        choices = set(ACTIVATIONS.keys()) - {self.op_name}
        if not choices: return False
        
        self.op_name = random.choice(list(choices))
        
        # Jitter weights
        with torch.no_grad():
            self.linear.weight += torch.randn_like(self.linear.weight) * 0.5
        return True

class MatrixNASNet(nn.Module):
    def __init__(self, input_dim=N_BITS, output_dim=1, num_chains=2, chain_depth=4):
        super().__init__()
        self.chains = nn.ModuleList([
            nn.Sequential(
                MatrixSymbolicNode(input_dim, HIDDEN_DIM, 'identity'),
                *[MatrixSymbolicNode(HIDDEN_DIM, HIDDEN_DIM, 'identity') for _ in range(chain_depth-1)]
            ) for _ in range(num_chains)
        ])
        self.final = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, x):
        chain_sum = 0
        for chain in self.chains:
            out = x
            for layer in chain:
                out = layer(out)
            chain_sum = chain_sum + out
        return self.final(chain_sum)

    def random_mutation(self):
        # Pick a random node to mutate
        c_idx = random.randint(0, len(self.chains) - 1)
        l_idx = random.randint(0, len(self.chains[c_idx]) - 1)
        
        node = self.chains[c_idx][l_idx]
        if isinstance(node, MatrixSymbolicNode):
            node.mutate()
            return c_idx, l_idx
        return None, None

def generate_parity_data(n_bits=8, n_samples=2000):
    X = np.random.randint(0, 2, size=(n_samples, n_bits))
    Y = np.sum(X, axis=1) % 2
    return torch.FloatTensor(X), torch.FloatTensor(Y).unsqueeze(1)

def train_parity_nas(n_bits=8):
    # Use same seed logic if needed, but we want variance
    # torch.manual_seed(42) 
    
    X, Y = generate_parity_data(n_bits)
    
    model = MatrixNASNet(input_dim=n_bits, output_dim=1, num_chains=2, chain_depth=4)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    
    best_loss = float('inf')
    epochs_since_imp = 0
    patience = 300 
    
    backup_state = None
    backup_chains = None
    
    TOTAL_EPOCHS = 8000
    
    print(f"Task: {n_bits}-Bit Parity (NAS Baseline)")
    print("Mode: Random Structural Search (Simulated Annealing without Gradients)")
    print("-" * 50)

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, Y)
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        history.append(curr_loss)
        
        acc = ((torch.sigmoid(preds) > 0.5) == Y).float().mean()
        
        if epoch % 1000 == 0:
            print(f"[NAS] Epoch {epoch}: Loss {curr_loss:.4f} | Acc {acc:.2%}")

        if curr_loss < best_loss:
            best_loss = curr_loss
            epochs_since_imp = 0
            backup_state = copy.deepcopy(model.state_dict())
            backup_chains = copy.deepcopy(model.chains)
        else:
            epochs_since_imp += 1

        # Mutation Strategy: Random "Blind" Search
        # If we stagnate, we just try to mutate a random node.
        if epochs_since_imp > patience:
            # Revert to best before Mutating (standard Hill Climbing / SA)
            if backup_state:
                model.load_state_dict(backup_state)
                model.chains = copy.deepcopy(backup_chains)
            
            # Mutate Randomly
            c, l = model.random_mutation()
            # print(f"  [NAS] Randomly mutated Chain {c} Layer {l}")
            
            # Reset optimizer momentum
            optimizer = optim.Adam(model.parameters(), lr=0.01) 
            epochs_since_imp = 0
        
        if acc > 0.99:
            print(f"[NAS] SOLVED! 100% Accuracy at Epoch {epoch}")
            # Fill remaining history with 0 or last loss for consistent plotting?
            # Better to just return what we have
            break

    return model, history

if __name__ == "__main__":
    model, history = train_parity_nas()
    plt.plot(history)
    plt.title("NAS Baseline Training")
    plt.show()
