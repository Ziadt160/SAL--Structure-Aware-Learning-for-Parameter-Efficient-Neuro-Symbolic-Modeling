import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

# --- CONFIGURATION ---
torch.manual_seed(42)
np.random.seed(42)
DEVICE = 'cpu'
HIDDEN_DIM = 64  # Increased width for MNIST

# --- 1. THE MATRIX FUNCTIONAL BASIS ---
ACTIVATIONS = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'identity': lambda x: x,
    'sin': lambda x: torch.sin(x) 
}

class MatrixSymbolicNode(nn.Module):
    def __init__(self, in_dim, out_dim, op_name='identity'):
        super().__init__()
        self.op_name = op_name
        self.linear = nn.Linear(in_dim, out_dim)
        # Proper initialization is crucial for MNIST
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.grad_accumulator = 0.0
        self.call_count = 0
        
    def forward(self, x):
        z = self.linear(x)
        if self.op_name in ACTIVATIONS:
            act = ACTIVATIONS[self.op_name]
            out = act(z) if isinstance(act, nn.Module) else act(z)
        else:
            out = z 

        if self.training and out.requires_grad:
            out.retain_grad()
            def hook(grad):
                self.grad_accumulator += grad.norm().item()
                self.call_count += 1
            out.register_hook(hook)
        return out

    def get_importance(self):
        if self.call_count == 0: return 0.0
        return self.grad_accumulator / self.call_count

    def reset_metrics(self):
        self.grad_accumulator = 0.0
        self.call_count = 0

    def mutate(self, forbidden_ops=None):
        if forbidden_ops is None: forbidden_ops = set()
        choices = set(ACTIVATIONS.keys()) - {self.op_name} - forbidden_ops
        if not choices: return False
        
        old_op = self.op_name
        self.op_name = random.choice(list(choices))
        print(f"    -> Mutating: {old_op.upper()} ===> {self.op_name.upper()}")
        
        # Slight jitter to break symmetry
        with torch.no_grad():
            self.linear.weight += torch.randn_like(self.linear.weight) * 0.02
        return True

# --- 2. THE CHAIN ---
class MatrixChain(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=3):
        super().__init__()
        self.layers = nn.ModuleList()
        # Input Layer (784 -> Hidden)
        self.layers.append(MatrixSymbolicNode(input_dim, hidden_dim, 'identity'))
        # Hidden Layers (Hidden -> Hidden)
        for _ in range(depth - 1):
            self.layers.append(MatrixSymbolicNode(hidden_dim, hidden_dim, 'identity'))
            
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

# --- 3. THE NETWORK ---
class MatrixGGLEN(nn.Module):
    def __init__(self, input_dim=784, output_dim=10, num_chains=2, chain_depth=3):
        super().__init__()
        self.chains = nn.ModuleList([
            MatrixChain(input_dim, HIDDEN_DIM, chain_depth) 
            for _ in range(num_chains)
        ])
        self.final = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, x):
        chain_sum = 0
        for chain in self.chains:
            chain_sum = chain_sum + chain(x)
        return self.final(chain_sum)

    def evolve_structure(self, history_tracker):
        candidates = []
        for c_idx, chain in enumerate(self.chains):
            for l_idx, node in enumerate(chain.layers):
                score = node.get_importance()
                # Bias evolution towards deeper layers first (l_idx)? No, pure gradients.
                candidates.append((score, c_idx, l_idx))
                node.reset_metrics()
        
        # Sort by lowest sensitivity (useless layers)
        candidates.sort(key=lambda x: x[0])
        
        for score, c_id, l_id in candidates:
            forbidden = history_tracker.get((c_id, l_id), set())
            print(f"  [Evolution] Targeting Chain {c_id} Layer {l_id} (Sens: {score:.4f})...")
            success = self.chains[c_id].layers[l_id].mutate(forbidden)
            if success: return c_id, l_id
        return None, None

# --- 4. DATA SETUP ---
def get_mnist_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Use a subset of 5000 images for fast evolution demo (60k is too slow for CPU demo)
    indices = torch.randperm(len(dataset))[:5000]
    subset = Subset(dataset, indices)
    
    return DataLoader(subset, batch_size=64, shuffle=True)

# --- 5. TRAINING LOOP ---
def train_mnist_gglen():
    train_loader = get_mnist_loader()
    
    # Model: 2 Chains, Depth 3
    model = MatrixGGLEN(input_dim=784, output_dim=10, num_chains=2, chain_depth=3).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    history_loss = []
    history_acc = []
    mutation_history = {}
    
    best_loss = float('inf')
    epochs_since_imp = 0
    patience = 5  # Epochs (since 1 epoch = many batches now)
    
    backup_state = None
    backup_chains = None
    just_mutated = False
    grace_period = 3 # Give it 3 epochs to adjust to new architecture
    
    print("Task: MNIST Digit Recognition")
    print("Start: Pure Linear Model (Expect ~90% accuracy ceiling)")
    print("-" * 60)

    for epoch in range(50): # Run for 50 epochs total
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.view(-1, 784).to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        history_loss.append(epoch_loss)
        history_acc.append(epoch_acc)
        
        print(f"Epoch {epoch}: Loss {epoch_loss:.4f} | Acc {epoch_acc:.2f}%")

        # --- EVOLUTION LOGIC ---
        
        # 1. Improvement Check
        if epoch_loss < best_loss * 0.98: # require 2% drop to count as "better" to avoid noise
            best_loss = epoch_loss
            epochs_since_imp = 0
            if just_mutated:
                print(f"  [Success] Architecture accepted. New Best: {best_loss:.4f}")
                just_mutated = False
        else:
            epochs_since_imp += 1
            
        # 2. Reversion
        if just_mutated and epochs_since_imp > grace_period:
            print(f"  [Failure] Mutation didn't help. Reverting...")
            c, l = active_node
            if (c,l) not in mutation_history: mutation_history[(c,l)] = set()
            mutation_history[(c,l)].add(active_op)
            
            model.load_state_dict(backup_state)
            model.chains = copy.deepcopy(backup_chains)
            optimizer = optim.Adam(model.parameters(), lr=0.002) # Reduce LR on revert
            
            just_mutated = False
            epochs_since_imp = 0
            
        # 3. Evolution Trigger
        if epochs_since_imp > patience and not just_mutated:
            print(f"\n--- Stagnation Detected (Acc: {epoch_acc:.2f}%) ---")
            
            backup_state = copy.deepcopy(model.state_dict())
            backup_chains = copy.deepcopy(model.chains)
            
            c_id, l_id = model.evolve_structure(mutation_history)
            if c_id is not None:
                active_node = (c_id, l_id)
                active_op = model.chains[c_id].layers[l_id].op_name
                just_mutated = True
                epochs_since_imp = 0
                optimizer = optim.Adam(model.parameters(), lr=0.005) # Reset LR for new structure
            else:
                print("Evolution Complete.")
                break
                
    return history_loss, history_acc

if __name__ == "__main__":
    loss_hist, acc_hist = train_mnist_gglen()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_hist)
    plt.title("Loss (Look for drops after evolution)")
    plt.xlabel("Epochs")
    
    plt.subplot(1, 2, 2)
    plt.plot(acc_hist)
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.show()