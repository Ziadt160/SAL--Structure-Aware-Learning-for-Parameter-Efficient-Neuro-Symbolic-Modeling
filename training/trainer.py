import copy
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any

class GAIOptimizer:
    """
    Evolutionary Training Loop for GAI Models.
    Handles gradient descent, simulated annealing for structural changes, and custom objectives.
    """
    def __init__(self, 
                 model: nn.Module, 
                 lr: float = 0.001, 
                 patience: int = 30, 
                 grace_period: int = 10, 
                 initial_temp: float = 0.5, 
                 use_annealing: bool = True, 
                 mutation_strategy: str = 'importance', 
                 l1_lambda: float = 0.0, 
                 loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 observer: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        """
        Args:
            model: The GAI model (MatrixGGLEN or GatedMatrixGGLEN).
            lr: Learning rate for Adam.
            patience: Epochs to wait before triggering evolution.
            grace_period: Epochs to wait after mutation before judging it.
            initial_temp: Temperature for simulated annealing.
            use_annealing: Whether to use probabilistic acceptance.
            mutation_strategy: 'importance' or 'random'.
            l1_lambda: L1 regularization coefficient.
            loss_fn: Custom loss function. If None, defaults to MSELoss.
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Loss Function Setup
        self.criterion = loss_fn if loss_fn is not None else nn.MSELoss()
            
        self.patience = patience
        self.grace_period = grace_period
        self.initial_temp = initial_temp
        self.use_annealing = use_annealing
        self.mutation_strategy = mutation_strategy 
        self.l1_lambda = l1_lambda
        self.observer = observer
        
        # Training State
        self.best_score = -float('inf') 
        self.epochs_no_improve = 0
        
        self.backup_state = None
        self.backup_chains = None
        self.best_ever_state = None
        self.best_ever_chains = None
        
        self.just_mutated = False
        self.mutation_history = {} # type: ignore
        self.active_op = None
        self.active_old_op = None
        self.active_node: Optional[Tuple[int, int]] = None
        self.prev_score_at_mutation = -float('inf')

    def fit(self, 
            X_train, y_train, 
            X_val, y_val, 
            epochs: int = 1000, 
            name: str = "GAI",
            optimize_efficiency: bool = False) -> Tuple[np.ndarray, List[float]]:
        
        # Data Prep
        # We rely on user providing correct types or we infer standard tensors
        X_t = torch.FloatTensor(X_train).float()
        X_v = torch.FloatTensor(X_val).float()
        
        # Handle y: Preserve dtype (e.g. Long for CrossEntropy, Float for MSE)
        if hasattr(y_train, 'dtype') and np.issubdtype(y_train.dtype, np.integer):
             y_t = torch.tensor(y_train).long()
             y_v = torch.tensor(y_val).long()
        else:
             y_t = torch.tensor(y_train).float()
             y_v = torch.tensor(y_val).float()
        
        if len(y_t.shape) == 1 and isinstance(self.criterion, (nn.MSELoss, nn.L1Loss)):
             # Auto-unsqueeze for regression if needed
             y_t = y_t.unsqueeze(1)
             y_v = y_v.unsqueeze(1)
             # And ensure float
             if not y_t.is_floating_point(): y_t = y_t.float(); y_v = y_v.float()
        
        history = []
        
        # --- PHASE 1: Structural Evolution ---
        for epoch in range(epochs):
            progress = epoch / epochs
            current_temp = self.initial_temp * (1 - progress)
            
            # --- Training Step ---
            self.model.train()
            self.optimizer.zero_grad()
            preds = self.model(X_t)
            
            # Flexible Loss Handling
            try:
                loss = self.criterion(preds, y_t)
            except TypeError:
                # Fallback for losses that need Input X (e.g. Physics/Judge losses)
                loss = self.criterion(preds, y_t, X_t)
            
            # Sparse Regularization
            if self.l1_lambda > 0:
                l1_reg = torch.tensor(0., requires_grad=True)
                # Supports both standard and gated via attribute access
                if hasattr(self.model, 'chains'):
                    for chain in self.model.chains: # type: ignore
                        l1_reg = l1_reg + torch.norm(chain.layers[0].linear.weight, 1)
                    loss = loss + self.l1_lambda * l1_reg
                
            loss.backward()
            self.optimizer.step()
            
            # --- Validation ---
            with torch.no_grad():
                val_preds = self.model(X_v)
                
                # Universal Metric: Negative Loss (Maximize score)
                val_loss = self.criterion(val_preds, y_v).item()
                val_score = -val_loss
                metric_str = f"Loss {val_loss:.5f}"
            
            history.append(val_score)

            # Log
            if epoch % 500 == 0:
                print(f"[{name}] Epoch {epoch}: Val {metric_str} | Temp {current_temp:.4f}", flush=True)
            
            if self.observer and epoch % 10 == 0:
                 self.observer('epoch_end', {
                     'epoch': epoch,
                     'score': val_score,
                     'loss': val_loss,
                     'temp': current_temp
                 })
            
            current_metric = val_score 
            
            # --- Strategy Update ---
            if current_metric > self.best_score:
                self._update_best(current_metric)
            else:
                self.epochs_no_improve += 1
                
            # --- Annealing / Acceptance ---
            if self.just_mutated and self.epochs_no_improve > self.grace_period:
                self._check_mutation_acceptance(current_metric, current_temp, name)
            
            # --- Evolution Trigger ---
            if self.epochs_no_improve > self.patience:
                if not self._trigger_evolution(current_metric, name):
                    break # Stop if no mutations left
        
        # --- Finalize Main Search ---
        if self.best_ever_state:
            print(f"[{name}] Restoring Best Ever State (Score: {self.best_score:.4f})", flush=True)
            self.model.load_state_dict(self.best_ever_state)
            if hasattr(self.model, 'chains'):
                self.model.chains = copy.deepcopy(self.best_ever_chains) # type: ignore
        
        # --- PHASE 2: Efficiency Sweep (Optional) ---
        if optimize_efficiency and hasattr(self.model, 'resize'):
            print(f"\n[{name}] --- Starting Efficiency Optimization ---", flush=True)
            baseline_score = self.best_score
            best_efficient_model = self.model
            best_efficient_score = baseline_score
            min_params = sum(p.numel() for p in self.model.parameters())
            
            # Start from current hidden_dim, go down
            current_h = getattr(self.model, 'hparams_dict', {}).get('hidden_dim', 64)
            candidates = [h for h in [64, 48, 32, 24, 16, 12, 8] if h < current_h]
            if not candidates and current_h > 16: candidates = [int(current_h * 0.75), int(current_h 
* 0.5)]

            for h_dim in candidates:
                try:
                    # Create reduced model
                    reduced_model = self.model.resize(h_dim).to(X_t.device)
                    params = sum(p.numel() for p in reduced_model.parameters())
                    
                    print(f"  [{name}] Testing size H={h_dim} (Params: {params})...", flush=True)
                    
                    # Quick Train (No evolution, just weight tuning)
                    opt = optim.Adam(reduced_model.parameters(), lr=0.001)
                    reduced_best = -float('inf')
                    
                    # Short but sufficient training
                    sweep_epochs = 50 
                    sub_patience = 10
                    no_imp = 0
                    
                    for e in range(sweep_epochs):
                        reduced_model.train()
                        opt.zero_grad()
                        p = reduced_model(X_t)
                        # Re-use loss logic
                        try: l = self.criterion(p, y_t)
                        except: l = self.criterion(p, y_t, X_t)
                        l.backward()
                        opt.step()
                        
                        reduced_model.eval()
                        with torch.no_grad():
                            vp = reduced_model(X_v)
                            vl = self.criterion(vp, y_v).item()
                            vs = -vl
                        
                        if vs > reduced_best:
                            reduced_best = vs
                            no_imp = 0
                        else:
                            no_imp += 1
                        if no_imp > sub_patience: break
                    
                    print(f"    -> Score: {reduced_best:.4f} (Baseline: {baseline_score:.4f})", flush=True)
                    
                    # Acceptance criteria: Maintain ~95% performance (or allow small drop)
                    # Since scores are negative (loss), we check if difference is small.
                    # e.g. -0.5 vs -0.55. 
                    # Let's say we allow 10% relative degradation of LOSS.
                    # loss = -score. 
                    base_loss = -baseline_score
                    curr_loss = -reduced_best
                    
                    if curr_loss <= base_loss * 1.15: # Allow 15% increase in loss
                        print(f"    -> ACCEPTED (Efficient).", flush=True)
                        best_efficient_model = reduced_model
                        best_efficient_score = reduced_best
                        min_params = params
                    else:
                        print(f"    -> REJECTED (Too much degradation).", flush=True)
                        # If a larger size failed, smaller ones likely will too? 
                        # Not always, but typically. We continue to see if maybe 32 is magic.
                
                except Exception as e:
                    print(f"    -> Failed: {e}", flush=True)
            
            # Switch to efficient model
            if best_efficient_model is not self.model:
                 print(f"[{name}] Switched to Efficient Model (H={best_efficient_model.hparams_dict['hidden_dim']}, Params={min_params})", flush=True)
                 self.model = best_efficient_model
                 self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) # Rebind optimizer
            
        self.model.eval()
        with torch.no_grad():
            final_out = self.model(X_v)
            final_preds = final_out.numpy()
        
        return final_preds, history

    def _update_best(self, current_metric):
        self.best_score = current_metric
        self.epochs_no_improve = 0
        self.backup_state = copy.deepcopy(self.model.state_dict())
        if hasattr(self.model, 'chains'):
            self.backup_chains = copy.deepcopy(self.model.chains) # type: ignore
        
        self.best_ever_state = copy.deepcopy(self.model.state_dict())
        if hasattr(self.model, 'chains'):
            self.best_ever_chains = copy.deepcopy(self.model.chains) # type: ignore

    def _check_mutation_acceptance(self, current_metric, temp, name):
        delta = current_metric - self.prev_score_at_mutation
        
        accept = False
        if delta > 0: 
            accept = True
        elif temp > 0.001 and self.use_annealing:
            prob = math.exp(delta / temp)
            if random.random() < prob:
                 accept = True
        
        if accept:
            status = "ACCEPTED" + (" (RISKY)" if delta <= 0 else "")
            print(f"  [{name}] [Result] '{str(self.active_op).upper()}' {status}. Score: {current_metric:.4f} (Delta: {delta:.4f})", flush=True)
            self.just_mutated = False
            self.epochs_no_improve = 0
            self.prev_score_at_mutation = current_metric
            # Update backup to current accepted state
            self.backup_state = copy.deepcopy(self.model.state_dict())
            if hasattr(self.model, 'chains'):
                self.backup_chains = copy.deepcopy(self.model.chains) # type: ignore
        else:
            print(f"  [{name}] [Result] '{str(self.active_op).upper()}' REJECTED. Score: {current_metric:.4f} (Delta: {delta:.4f})", flush=True)
            if self.observer:
                self.observer("mutation", {
                    "old_op": self.active_old_op,
                    "new_op": self.active_op,
                    "score_delta": delta,
                    "accepted": False
                })
            self._revert(name)
        
        if accept and self.observer:
             self.observer("mutation", {
                "old_op": self.active_old_op,
                "new_op": self.active_op,
                "score_delta": delta,
                "accepted": True
            })

    def _trigger_evolution(self, current_metric, name) -> bool:
        print(f"\n[{name}] --- Stagnation at Score {self.best_score:.4f} ---", flush=True)
        
        # Checkpoint before mutation attempts
        self.backup_state = copy.deepcopy(self.model.state_dict())
        if hasattr(self.model, 'chains'):
            self.backup_chains = copy.deepcopy(self.model.chains) # type: ignore
        self.prev_score_at_mutation = current_metric

        if hasattr(self.model, 'evolve_structure'):
            c_id, l_id, old_op = self.model.evolve_structure(self.mutation_history, logger_prefix=name, strategy=self.mutation_strategy) # type: ignore
            
            if c_id is not None:
                self.active_node = (c_id, l_id)
                self.active_old_op = old_op
                # Access via chains
                if hasattr(self.model, 'chains'):
                    self.active_op = self.model.chains[c_id].layers[l_id].op_name # type: ignore
                    
                self.just_mutated = True
                self.epochs_no_improve = 0
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) 
                return True
        
        print(f"[{name}] No mutations possible or supported.", flush=True)
        return False

    def _revert(self, name):
        self.model.load_state_dict(self.backup_state) # type: ignore
        if hasattr(self.model, 'chains') and self.backup_chains:
            self.model.chains = copy.deepcopy(self.backup_chains) # type: ignore
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if self.active_node:
            c, l = self.active_node
            if (c,l) not in self.mutation_history: self.mutation_history[(c,l)] = set()
            self.mutation_history[(c,l)].add(self.active_op)
        
        self.just_mutated = False
        self.epochs_no_improve = self.patience + 1 
        print(f"  [{name}] Instant Retry: Triggering evolution again...", flush=True)

    def surgical_evolve(self, 
                        X_train, y_train, 
                        X_val, y_val, 
                        chain_id: int, 
                        layer_id: int, 
                        epochs: int = 5,
                        name: str = "GAI-Surgeon") -> bool:
        """
        Targeted evolution: Freezes the network, mutates ONE specific node, and fine-tunes it.
        Used for 'Surgical' fixes of dead/weak layers.
        """
        print(f"\n[{name}] 🏥 STARTING SURGERY on Chain {chain_id}, Layer {layer_id}...", flush=True)
        
        # 1. State Preservation
        original_state = copy.deepcopy(self.model.state_dict())
        original_chains = None
        if hasattr(self.model, 'chains'):
            original_chains = copy.deepcopy(self.model.chains)

        # 2. Get Baseline Score
        self.model.eval()
        with torch.no_grad():
            base_preds = self.model(torch.FloatTensor(X_val).float())
            try: base_loss = self.criterion(base_preds, torch.tensor(y_val).float()).item()
            except: base_loss = 999.0
            base_score = -base_loss
        
        print(f"  [{name}] Baseline Score: {base_score:.4f}", flush=True)

        # 3. Freeze Global, Unfreeze Local
        for param in self.model.parameters():
            param.requires_grad = False
            
        target_node = self.model.chains[chain_id].layers[layer_id]
        for param in target_node.parameters():
            param.requires_grad = True
            
        # 4. Mutation Step (Force a mutation on this node)
        # We need a set of forbidden ops to ensure we actually change it
        current_op = target_node.op_name
        forbidden = {current_op}
        
        success, old_op = target_node.mutate(forbidden, logger_prefix=name)
        if not success:
            print(f"  [{name}] Mutation failed (no options). Aborting surgery.", flush=True)
            self._restore_full_training(original_state, original_chains)
            return False

        # 5. Local Training Loop (Fine-tune the new op)
        # We use a higher LR because we are only training a small part
        surgeon_opt = optim.Adam(target_node.parameters(), lr=0.01)
        
        X_t = torch.FloatTensor(X_train).float()
        y_t = torch.tensor(y_train).float()
        if len(y_t.shape) == 1: y_t = y_t.unsqueeze(1) # Safety

        for e in range(epochs):
            self.model.train()
            surgeon_opt.zero_grad()
            out = self.model(X_t)
            try: loss = self.criterion(out, y_t)
            except: loss = self.criterion(out, y_t, X_t)
            loss.backward()
            surgeon_opt.step()
        
        # 6. Evaluation
        self.model.eval()
        with torch.no_grad():
            new_preds = self.model(torch.FloatTensor(X_val).float())
            try: new_loss = self.criterion(new_preds, torch.tensor(y_val).float()).item()
            except: new_loss = 999.0
            new_score = -new_loss
            
        delta = new_score - base_score
        print(f"  [{name}] Post-Op Score: {new_score:.4f} (Delta: {delta:.4f})", flush=True)
        
        # 7. Accept or Revert
        if delta > 0:
            print(f"  [{name}] Surgery SUCCESSFUL. Keeping change ({old_op} -> {target_node.op_name}).", flush=True)
            # Restore requires_grad for everything (Unfreeze)
            for param in self.model.parameters():
                param.requires_grad = True
            return True
        else:
            print(f"  [{name}] Surgery FAILED. Reverting to original state.", flush=True)
            self._restore_full_training(original_state, original_chains)
            return False

    def _restore_full_training(self, state, chains):
        self.model.load_state_dict(state)
        if chains and hasattr(self.model, 'chains'):
            self.model.chains = copy.deepcopy(chains)
        
        # Unfreeze all
        for param in self.model.parameters():
            param.requires_grad = True
        # Rebind main optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
