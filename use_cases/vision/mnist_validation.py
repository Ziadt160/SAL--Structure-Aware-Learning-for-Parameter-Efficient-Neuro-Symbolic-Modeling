import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sympy
import time
import sys
from typing import List, Tuple, Any

# Adjust path to ensure modules are found
sys.path.append("d:/Quantum Projects/GAI")

from models.adaptive_neural_model import GatedMatrixGGLEN, MatrixChain
from training.trainer import GAIOptimizer

def load_mnist(subset_size: int = 5000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase 1: MNIST Environment Setup
    Loads MNIST, flattens to 784 dim, and creates subsets for rapid validation.
    """
    print("\n[Phase 1] Setting up MNIST Environment...")
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    # We use the train set for training and test set for validation
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Subsample for speed if requested (GAIOptimizer is full-batch)
    if subset_size > 0:
        indices = torch.randperm(len(train_dataset))[:subset_size]
        data = train_dataset.data[indices].float() / 255.0
        # Re-apply normalization manually or just use the loader
        # Easiest is to use a loader to apply transforms then collect
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=subset_size, shuffle=True)
        # Just grab the first batch
        X_train, y_train = next(iter(train_loader))
        print(f"  -> Loaded Subset: {len(X_train)} training samples.")
    else:
        # Full loading (might create memory pressure)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=60000, shuffle=True)
        X_train, y_train = next(iter(train_loader))
        print(f"  -> Loaded Full Set: {len(X_train)} training samples.")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)
    X_test, y_test = next(iter(test_loader))
    print(f"  -> Loaded Test Set: {len(X_test)} samples.")
    
    return X_train, y_train, X_test, y_test

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    if preds.shape[1] > 1:
        # Classification
        pred_cls = preds.argmax(dim=1)
    else:
        # Binary/Regression (not applicable here really, but handling)
        pred_cls = (preds > 0.5).long()
    
    correct = (pred_cls == targets).sum().item()
    return (correct / len(targets)) * 100.0

def evaluate_symbolic_formula(formula_str: str, input_vars: List[str], X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    """
    Evaluates the raw symbolic string using SymPy lambdify.
    """
    print("  -> Compiling Symbolic Formula (this may take a moment)...")
    try:
        # 1. Parse string back to SymPy expression list
        # We need the context of sympy functions
        local_dict = {
            'tanh': sympy.tanh, 'relu': sympy.Max, 'flatten': sympy.flatten,
            'sigmoid': lambda x: 1/(1+sympy.exp(-x)), 'sin': sympy.sin,
            'cos': sympy.cos, 'exp': sympy.exp, 'pi': sympy.pi,
            'Max': sympy.Max, 'Piecewise': sympy.Piecewise, 
            'Symbol': sympy.Symbol
        }
        # Add input vars to context
        for v in input_vars:
            local_dict[v] = sympy.Symbol(v)
            
        # The string is "[expr1, expr2..]"
        expr_list = sympy.sympify(formula_str, locals=local_dict)
        
        # 2. Lambdify
        # Create a function that takes arguments x0, x1... and returns list of outputs
        sym_vars = [sympy.Symbol(v) for v in input_vars]
        func = sympy.lambdify(sym_vars, expr_list, modules=['numpy'])
        
        # 3. Evaluate Batch
        # Lambdify might expect unpacked arguments or specific structure
        # X_test is [N, 784]. We need to feed columns.
        X_np = X_test.detach().cpu().numpy()
        
        # We can pass *cols
        input_cols = [X_np[:, i] for i in range(X_np.shape[1])]
        
        start_t = time.time()
        # Returns shape (InputDim, Batch) or (Batch, OutputDim)? 
        # Usually lambdify on a list returns a list of arrays.
        outputs = func(*input_cols) 
        # outputs is list of length 10, each is array of shape (N,)
        
        # Stack to (N, 10)
        outputs_np = np.stack(outputs, axis=1)
        
        # Accuracy
        pred_cls = np.argmax(outputs_np, axis=1)
        targets_np = y_test.detach().cpu().numpy()
        correct = (pred_cls == targets_np).sum()
        acc = (correct / len(targets_np)) * 100.0
        
        print(f"  -> Symbolic eval time: {time.time() - start_t:.2f}s")
        return acc
        
    except Exception as e:
        print(f"  [Error] Failed to evaluate symbolic formula: {e}")
        return 0.0

def run_experiment():
    # --- Phase 1: Setup ---
    X_train, y_train, X_test, y_test = load_mnist(subset_size=5000) # 5000 for speed
    
    input_dim = 784
    output_dim = 10
    
    # --- Phase 2: Train "Heavy" Oracle ---
    print("\n[Phase 2] Training GatedMatrixGGLEN (Heavy Oracle)...")
    model = GatedMatrixGGLEN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=32,
        num_chains=4,
        chain_depth=3
    )
    
    # Custom wrapper to handle float targets in optimizer but long targets for loss
    def loss_wrapper(preds, targets):
        return nn.CrossEntropyLoss()(preds, targets.long())

    optimizer = GAIOptimizer(
        model,
        lr=0.005, # Slightly higher for fast convergence
        patience=10,
        initial_temp=1.0, # Encourage early exploration
        loss_fn=loss_wrapper
    )
    
    print(f"  -> Initial Params: {count_params(model)}")
    
    # Train
    start_time = time.time()
    optimizer.fit(X_train, y_train, X_test, y_test, epochs=50, name="GatedOracle")
    print(f"  -> Training Time: {time.time() - start_time:.2f}s")
    
    # Evaluate Full Model
    model.eval()
    with torch.no_grad():
        full_preds = model(X_test)
        full_acc = get_accuracy(full_preds, y_test)
        full_params = count_params(model)
    
    print(f"\n  [Result] Full Gated Model Accuracy: {full_acc:.2f}%")

    # --- Phase 3: Distillation & Pruning ---
    print("\n[Phase 3] Distillation & Pruning...")
    
    # 1. Gate Analysis: Find Winning Expert
    # Run test set through gate to see weights
    with torch.no_grad():
        # gate returns logits, we need softmax
        gate_logits = model.gate(X_test)
        gate_weights = F.softmax(gate_logits, dim=1) # [N, NumExperts]
        
        # Average weight per expert
        avg_weights = gate_weights.mean(dim=0)
        winning_idx = torch.argmax(avg_weights).item()
        
    print(f"  -> Gate Analysis: Expert {winning_idx} is the winner (Avg Weight: {avg_weights[winning_idx]:.4f})")
    
    # 2. Ablation / Isolation
    winning_chain = model.chains[winning_idx]
    
    # We need to add the final layer of the gated model or does the chain output directly?
    # GatedMatrixGGLEN sums chain outputs then passes through self.final (Linear)
    # The chain itself gives hidden_dim. 
    # To truly "Prune" to a single formula, we need to collapse the whole path: Chain -> Sum -> Final.
    # Since we isolated one chain, the sum is just that chain.
    # So the "Distilled Model" is Chain + Model.Final
    
    class DistilledExpert(nn.Module):
        def __init__(self, chain, final_layer):
            super().__init__()
            self.chain = chain
            self.final = final_layer
        def forward(self, x):
            return self.final(self.chain(x))
            
    distilled_model = DistilledExpert(winning_chain, model.final)
    
    # Evaluate Distilled Model
    with torch.no_grad():
        distilled_preds = distilled_model(X_test)
        distilled_acc = get_accuracy(distilled_preds, y_test)
        distilled_params = count_params(distilled_model)
        
    print(f"  [Result] Top Expert Chain (Distilled) Accuracy: {distilled_acc:.2f}%")

    # 3. Symbolic Conversion
    print("  -> Generating Symbolic Formula...")
    input_vars = [f"x{i}" for i in range(input_dim)]
    
    # The chain.export_formula gives us the output of the chain (hidden state).
    # We need to manually compose it with the final linear layer to get the logits formula.
    # This is tricky because export_formula is just on the chain.
    # We will export the chain formula, then symbolically multiply by the final layer weights.
    
    # A. Get Chain Formula
    # chain_formula_str is a string of a list "[expr_h1, expr_h2...]"
    chain_formula_str = winning_chain.export_formula(input_vars)
    # Parse back to sympy
    chain_exprs = sympy.sympify(chain_formula_str, locals={'tanh': sympy.tanh, 'relu': sympy.Max, 'flatten': sympy.flatten, 'sigmoid': lambda x: 1/(1+sympy.exp(-x)), 'sin': sympy.sin, 'cos': sympy.cos, 'exp': sympy.exp, 'pi': sympy.pi, 'Max': sympy.Max, 'Piecewise': sympy.Piecewise})
    
    # B. Add Final Linear Layer
    W_final = model.final.weight.detach().cpu().numpy() # [OutDim, HiddenDim]
    b_final = model.final.bias.detach().cpu().numpy()   # [OutDim]
    
    final_exprs = []
    for out_i in range(output_dim):
        term = b_final[out_i]
        for h_i in range(len(chain_exprs)):
            term += W_final[out_i, h_i] * chain_exprs[h_i]
        final_exprs.append(term)
        
    final_formula_str = str(final_exprs)
    formula_length = len(final_formula_str)
    
    print(f"  [Result] Symbolic Formula Length: {formula_length} chars")
    
    # Calculate symbolic accuracy
    symbolic_acc = evaluate_symbolic_formula(final_formula_str, input_vars, X_test, y_test)
    print(f"  [Result] Symbolic Formula Accuracy: {symbolic_acc:.2f}%")

    # --- Phase 4: Comparative Results ---
    
    efficiency_mult = full_params / (formula_length if formula_length > 0 else 1) # Crude proxy: Params vs Chars
    # Better proxy might be "Compressed Size"
    
    print("\n" + "="*60)
    print("FINAL MNIST VALIDATION REPORT")
    print("="*60)
    print(f"{'Model Variant':<25} | {'Accuracy':<10} | {'Complexity':<15}")
    print("-" * 60)
    print(f"{'Full Gated Model':<25} | {full_acc:<9.2f}% | {full_params:<15} params")
    print(f"{'Top Expert (Distilled)':<25} | {distilled_acc:<9.2f}% | {distilled_params:<15} params")
    print(f"{'Symbolic Formula':<25} | {symbolic_acc:<9.2f}% | {formula_length:<15} chars")
    print("-" * 60)
    print(f"Efficiency Multiplier (Params/Chars): {efficiency_mult:.2f}x")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
