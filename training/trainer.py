import os
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
                 l1_scope: str = 'first',
                 mutate_reset_weights: bool = True,
                 legacy_sa: bool = False,
                 mutation_mode: str = 'homotopy',
                 tabu_tenure: int = 12,
                 revert_cooldown: Optional[int] = None,
                 min_rel_delta: float = 1e-3,
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
            l1_lambda: L1 penalty on the FIRST layer of each chain only, not on
                all weights -- input-feature sparsity, not global sparsity.
                See the note at the penalty site.
            loss_fn: Custom loss function. If None, defaults to MSELoss.
        """
        self.model = model
        # Remembered so every later rebind uses the caller's value. All three
        # rebind sites used to hardcode lr=0.001, silently discarding it after
        # the first mutation.
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Loss Function Setup
        self.criterion = loss_fn if loss_fn is not None else nn.MSELoss()
            
        self.patience = patience
        self.grace_period = grace_period
        self.initial_temp = initial_temp
        self.use_annealing = use_annealing
        self.mutation_strategy = mutation_strategy 
        self.l1_lambda = l1_lambda
        # 'first' penalises layers[0] of each chain only -- input-feature
        # sparsity, and the original behaviour. 'all' penalises every layer,
        # which is what the phrase "L1 regularization coefficient" implies but
        # is NOT what this code did. Default kept at 'first' because GAI-B and
        # GAI-C set l1_lambda=1e-05, so switching would silently change every
        # result those configs have produced.
        if l1_scope not in ('first', 'all'):
            raise ValueError(f"l1_scope must be 'first' or 'all', got {l1_scope!r}")
        self.l1_scope = l1_scope
        # False = carry the existing weights into the new operator instead of
        # re-initialising the node. Untested until now: the flag existed on
        # MatrixSymbolicNode.mutate but nothing ever passed it.
        self.mutate_reset_weights = mutate_reset_weights
        # legacy_sa=True restores the original acceptance/stagnation behaviour
        # exactly, for reproducing earlier results.
        self.legacy_sa = legacy_sa
        # 'homotopy' morphs g_old -> g_new continuously so the swap starts
        # function-preserving (Network Morphism P-activation, arXiv:1603.01670).
        # 'reset' installs the new operator directly; whether that node's weights
        # are re-initialised or carried over is then controlled by
        # `mutate_reset_weights`, NOT by this argument.
        #
        # There is deliberately no 'transfer' mode. An earlier comment claimed
        # one, and since only 'homotopy' is ever branched on, any other string
        # silently behaved as 'reset' -- so a caller asking for 'transfer' got
        # reset semantics with no error. Validated here because this exact class
        # of silent fallthrough already invalidated one experiment
        # (mutate_reset_weights is a no-op under 'homotopy', which made both arms
        # of experiments/mutation_mechanics.py identical).
        if mutation_mode not in ('homotopy', 'reset'):
            raise ValueError(
                f"mutation_mode must be 'homotopy' or 'reset', got "
                f"{mutation_mode!r}. To carry the mutated node's weights into "
                f"the new operator use mutation_mode='reset' with "
                f"mutate_reset_weights=False."
            )
        self.mutation_mode = mutation_mode
        # Finite tabu tenure. The original list was never cleared, so after
        # ~5 rejections per node every option was forbidden and training halted.
        self.tabu_tenure = tabu_tenure
        # Epochs of plain training after a rejection before proposing again.
        # The original set epochs_no_improve = patience+1, re-triggering on the
        # very next epoch with zero intervening optimisation.
        self.revert_cooldown = (grace_period if revert_cooldown is None
                                else revert_cooldown)
        self.min_rel_delta = min_rel_delta
        self.rolling_reference = -float('inf')
        self.grace_best = -float('inf')
        self.cooldown = 0
        self.mutation_enabled = True
        self.mutation_order = 0
        self.backup_optimizer = None
        self.swap_node = None
        self.swap_steps = 1
        self.swap_epoch = 0
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

        # Epoch-indexed record of every structural event: one dict per mutation
        # with {epoch, chain, layer, old_op, new_op, accepted}.
        #
        # Experiments used to recover these from the loss curve by looking for a
        # >3x single-step jump, on the assumption that re-initialising a node
        # always leaves a visible discontinuity. Under mutation_mode='homotopy'
        # that assumption is false BY CONSTRUCTION: a swap at t=0 is
        # bit-identical to no swap, so the curve is smooth and the detector sees
        # nothing. Any homotopy-mode mutation count or payoff rate derived that
        # way is computed over a biased subsample. Read this instead.
        self.mutation_log: list = []
        self.current_epoch = -1

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
        #
        # np.issubdtype cannot interpret a torch dtype ("Cannot interpret
        # 'torch.float32' as a data type"), so passing a torch tensor to a torch
        # trainer used to raise TypeError here. Every script in the repo happens
        # to pass numpy, which is why it went unnoticed.
        if torch.is_tensor(y_train):
            y_is_int = not torch.is_floating_point(y_train)
        elif hasattr(y_train, 'dtype'):
            y_is_int = np.issubdtype(y_train.dtype, np.integer)
        else:
            y_is_int = False
        if y_is_int:
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
            self.current_epoch = epoch
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
            #
            # NOTE: this penalises `layers[0]` of each chain ONLY, not the whole
            # network. At chain_depth=3 that is one third of the weights; the
            # deeper layers are entirely unregularised. Read as INPUT-FEATURE
            # sparsity it is a defensible design (layer 0 is what selects among
            # the inputs); read as the "L1 regularization coefficient" the
            # docstring advertises, it is not what it says.
            #
            # `l1_scope='all'` opts into penalising every layer; 'first' (the
            # default) preserves the original behaviour, because GAI-B and
            # GAI-C set l1_lambda=1e-05 and widening it silently would change
            # every result those configs have produced.
            if self.l1_lambda > 0:
                l1_reg = torch.tensor(0., requires_grad=True)
                # Supports both standard and gated via attribute access
                if hasattr(self.model, 'chains'):
                    for chain in self.model.chains: # type: ignore
                        targets = (chain.layers if self.l1_scope == 'all'
                                   else chain.layers[:1])
                        for layer in targets:
                            l1_reg = l1_reg + torch.norm(layer.linear.weight, 1)
                    loss = loss + self.l1_lambda * l1_reg
                
            loss.backward()
            self.optimizer.step()
            
            # --- Validation ---
            # eval() matters: `no_grad` disables gradients but does NOT disable
            # dropout. Validating while still in train() mode -- the previous
            # behaviour -- made the score both noisy and biased. Measured at
            # dropout=0.3 on a frozen model: five successive readings spanned
            # 1.2327-1.3092 where the true value is 1.1158, a 13% upward bias on
            # top of the noise. That score drives best-model tracking, stagnation
            # detection AND mutation accept/reject, so the noise propagates into
            # every structural decision. Harmless at the dropout=0.0 default used
            # by every experiment in this repo; not harmless for
            # use_cases/control/run_ctz_baseline.py (0.01) or any --dropout run.
            self.model.eval()
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
                # Only the legacy path measures stagnation against the global
                # best. In the rolling path the block below owns the counter --
                # letting _update_best zero it here is what made the rolling
                # reference inert (the counter could never exceed 1 while the
                # run was still setting new global bests).
                if self.legacy_sa:
                    self.epochs_no_improve = 0
            elif self.legacy_sa:
                self.epochs_no_improve += 1

            if not self.legacy_sa:
                # Stagnation measured against a ROLLING reference rather than
                # the all-time best. With the global best as reference, the
                # counter increments essentially every epoch once a run peaks,
                # so evolution fires every `patience` epochs for the remainder
                # of training no matter how well local optimisation is going.
                # That is the mechanism behind the observed behaviour where a
                # run finds its best early and then thrashes for the rest of
                # the budget without ever recovering.
                #
                # The reference must be SEEDED from the first observed score.
                # Starting it at -inf makes the threshold below
                # `-inf + delta*abs(-inf)` = `-inf + inf` = nan, and `x > nan`
                # is False for every x and every delta -- so the reference could
                # never leave -inf and the else-branch fired unconditionally.
                if not math.isfinite(self.rolling_reference):
                    self.rolling_reference = current_metric
                    self.epochs_no_improve = 0
                elif current_metric > self.rolling_reference + \
                        self.min_rel_delta * abs(self.rolling_reference):
                    self.rolling_reference = current_metric
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                if self.just_mutated:
                    self.grace_best = max(self.grace_best, current_metric)
                    self._advance_swap()
                if self.cooldown > 0:
                    self.cooldown -= 1

            # --- Annealing / Acceptance ---
            if self.just_mutated and self.epochs_no_improve > self.grace_period:
                # Judge on the BEST score seen during the grace window, not a
                # single epoch's reading -- same cost, far lower variance.
                judged = (self.grace_best if not self.legacy_sa
                          else current_metric)
                self._check_mutation_acceptance(judged, current_temp, name)

            # --- Evolution Trigger ---
            ready = self.epochs_no_improve > self.patience and self.cooldown <= 0
            if ready and self.mutation_enabled:
                if not self._trigger_evolution(current_metric, name):
                    if self.legacy_sa:
                        break  # Stop if no mutations left
                    # Exhausting the tabu list should not end training. The
                    # original `break` cut the run short exactly when the
                    # rejection rate was high, which silently gave a
                    # no-mutation control arm more epochs than the mutating
                    # arm in any paired comparison.
                    print(f"[{name}] No mutations left; continuing to train "
                          f"with the structure frozen.", flush=True)
                    self.mutation_enabled = False

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
                 self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr) # Rebind optimizer
            
        self.model.eval()
        with torch.no_grad():
            final_out = self.model(X_v)
            final_preds = final_out.numpy()
        
        return final_preds, history

    def _cancel_swap(self):
        node = self._swap_node_ref()
        if node is not None and getattr(node, 'swapping', False):
            node.cancel_swap()
        self.swap_node = None

    def _swap_node_ref(self):
        nd = getattr(self, 'swap_node', None)
        if nd is None or not hasattr(self.model, 'chains'):
            return None
        c, l = nd
        try:
            return self.model.chains[c].layers[l]   # type: ignore
        except (IndexError, AttributeError):
            return None

    def _advance_swap(self):
        """Step the homotopy one epoch closer to the new operator."""
        node = self._swap_node_ref()
        if node is None or not getattr(node, 'swapping', False):
            return
        self.swap_epoch += 1
        node.set_swap_t(self.swap_epoch / float(self.swap_steps))

    def _age_tabu(self):
        """Drop the oldest forbidden operators so the tabu list stays finite.

        The original list was append-only and never cleared, so after roughly
        five rejections per node every alternative was forbidden, evolve_structure
        returned None, and training stopped early. Standard tabu search uses a
        finite tenure for exactly this reason.
        """
        total = sum(len(v) for v in self.mutation_history.values())
        if total <= self.tabu_tenure:
            return
        for key in list(self.mutation_history):
            if self.mutation_history[key]:
                self.mutation_history[key].pop()
                total -= 1
            if not self.mutation_history[key]:
                del self.mutation_history[key]
            if total <= self.tabu_tenure:
                return

    def _snapshot_optimizer(self):
        """Adam's moment estimates, so a mutation does not restart its warm-up.

        Both _trigger_evolution and _revert used to rebind Adam outright,
        discarding first and second moments every time -- against a measured
        4-6 epoch recovery horizon, that is a large fraction of the recovery.
        """
        return copy.deepcopy(self.optimizer.state_dict())

    def _restore_optimizer(self, state):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        if state is not None and not self.legacy_sa:
            try:
                self.optimizer.load_state_dict(state)
            except (ValueError, KeyError):
                pass   # parameter set changed (topology move); fresh Adam is right

    def _update_best(self, current_metric):
        # Deliberately does NOT touch epochs_no_improve: which signal owns the
        # stagnation counter depends on legacy_sa, so the caller decides. This
        # used to zero it here unconditionally, which silently overrode the
        # rolling reference and pinned the counter at 1 for any run that was
        # still setting new global bests.
        self.best_score = current_metric
        self.backup_state = copy.deepcopy(self.model.state_dict())
        if hasattr(self.model, 'chains'):
            self.backup_chains = copy.deepcopy(self.model.chains) # type: ignore
        
        self.best_ever_state = copy.deepcopy(self.model.state_dict())
        if hasattr(self.model, 'chains'):
            self.best_ever_chains = copy.deepcopy(self.model.chains) # type: ignore

    def _check_mutation_acceptance(self, current_metric, temp, name):
        delta = current_metric - self.prev_score_at_mutation

        accept = False
        _pending = self.mutation_log[-1] if self.mutation_log else None
        if delta > 0:
            accept = True
        elif temp > 0.001 and self.use_annealing:
            if self.legacy_sa:
                prob = math.exp(delta / temp)
            else:
                # Scale-invariant acceptance. `delta` is in absolute loss units,
                # so with a fixed temperature the same setting accepts almost
                # everything on a task whose MSE is ~1e-5 and almost nothing
                # where the loss is ~1. Normalising by the pre-mutation score
                # makes the temperature mean the same thing on every task.
                scale = max(abs(self.prev_score_at_mutation), 1e-12)
                prob = math.exp((delta / scale) / temp)
            if random.random() < prob:
                accept = True
        
        if _pending is not None and _pending.get('accepted') is None:
            _pending['accepted'] = accept
            _pending['delta'] = float(delta)
            _pending['judged_epoch'] = self.current_epoch

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
        self.backup_optimizer = self._snapshot_optimizer()
        self.prev_score_at_mutation = current_metric
        self.grace_best = current_metric

        if hasattr(self.model, 'evolve_structure'):
            c_id, l_id, old_op = self.model.evolve_structure(
                self.mutation_history, logger_prefix=name,
                strategy=self.mutation_strategy,
                reset_weights=self.mutate_reset_weights)  # type: ignore
            
            if c_id is not None:
                self.active_node = (c_id, l_id)
                self.active_old_op = old_op
                # Access via chains
                if hasattr(self.model, 'chains'):
                    self.active_op = self.model.chains[c_id].layers[l_id].op_name # type: ignore
                    
                node = self.model.chains[c_id].layers[l_id]  # type: ignore
                if (not self.legacy_sa and self.mutation_mode == 'homotopy'
                        and hasattr(node, 'begin_swap')):
                    # evolve_structure already installed the new operator and
                    # (optionally) reset the weights. Undo both, then re-enter
                    # the change as a function-preserving morph from the old
                    # operator: at t=0 the model is untouched, so the grace
                    # window measures the OPERATOR rather than the damage from
                    # re-initialising the node.
                    self.model.load_state_dict(self.backup_state)
                    if hasattr(self.model, 'chains') and self.backup_chains:
                        self.model.chains = copy.deepcopy(self.backup_chains)
                    node = self.model.chains[c_id].layers[l_id]
                    node.begin_swap(self.active_op)
                    self.swap_node = (c_id, l_id)
                    self.swap_steps = max(1, self.grace_period)
                    self.swap_epoch = 0
                self.just_mutated = True
                self.epochs_no_improve = 0
                self._restore_optimizer(self.backup_optimizer)
                self.mutation_log.append({
                    'epoch': self.current_epoch,
                    'chain': c_id, 'layer': l_id,
                    'old_op': old_op, 'new_op': self.active_op,
                    'accepted': None,      # filled in by _check_mutation_acceptance
                })
                return True
        
        print(f"[{name}] No mutations possible or supported.", flush=True)
        return False

    def _revert(self, name):
        self._cancel_swap()
        self.model.load_state_dict(self.backup_state) # type: ignore
        if hasattr(self.model, 'chains') and self.backup_chains:
            self.model.chains = copy.deepcopy(self.backup_chains) # type: ignore

        self._restore_optimizer(getattr(self, 'backup_optimizer', None))
        
        if self.active_node:
            c, l = self.active_node
            if (c, l) not in self.mutation_history:
                self.mutation_history[(c, l)] = set()
            self.mutation_history[(c, l)].add(self.active_op)
            if not self.legacy_sa and self.tabu_tenure > 0:
                self._age_tabu()

        self.just_mutated = False
        self.grace_best = -float('inf')
        if self.legacy_sa:
            self.epochs_no_improve = self.patience + 1
        else:
            # Train for a cooldown before proposing again, instead of
            # re-triggering on the very next epoch.
            self.epochs_no_improve = 0
            # -inf is a RESEED sentinel, not a value: the next epoch sees a
            # non-finite reference and re-anchors it to the restored model's
            # score, which is the right baseline to measure stagnation from
            # after a revert. (Before the seeding fix in fit(), -inf was
            # absorbing: the reference could never leave it again.)
            self.rolling_reference = -float('inf')
            self.cooldown = self.revert_cooldown
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
        X_v = torch.FloatTensor(X_val).float()
        y_v = self._prepare_targets(y_val)
        base_score = -self._eval_loss(X_v, y_v)
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
        y_t = self._prepare_targets(y_train)

        for e in range(epochs):
            self.model.train()
            surgeon_opt.zero_grad()
            out = self.model(X_t)
            try:
                loss = self.criterion(out, y_t)
            except TypeError:
                loss = self.criterion(out, y_t, X_t)
            loss.backward()
            surgeon_opt.step()

        # 6. Evaluation
        new_score = -self._eval_loss(X_v, y_v)

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

    def save_best(self, path: str, extra: Optional[Dict[str, Any]] = None):
        """Persist the best-on-validation model found during fit().

        In-memory selection was already correct -- `_update_best` deep-copies
        both `state_dict()` and `chains`, and `fit()` restores them at the end.
        But nothing was ever written out, so every discovered architecture died
        with the process. That matters more here than in an ordinary trainer:
        `op_name` is a plain Python string and therefore absent from
        `state_dict()`, so a bare `torch.save(model.state_dict())` silently
        loses the structure -- the one thing the search produces. `checkpoint()`
        stores weights and structure together.
        """
        model = self.model
        if not hasattr(model, 'checkpoint'):
            raise TypeError(f"{type(model).__name__} has no checkpoint(); "
                            f"use a MatrixGGLEN/GatedMatrixGGLEN")
        payload = model.checkpoint()
        payload['best_val_score'] = self.best_score
        payload['best_val_loss'] = -self.best_score
        payload['lr'] = self.lr
        payload['mutation_strategy'] = self.mutation_strategy
        payload['initial_temp'] = self.initial_temp
        payload['use_annealing'] = self.use_annealing
        payload['l1_lambda'] = self.l1_lambda
        if extra:
            payload.update(extra)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(payload, path)
        return path

    def _prepare_targets(self, y) -> torch.Tensor:
        """Shape and type targets to match self.criterion.

        The old code did `torch.tensor(y_val).float()` with no unsqueeze. For
        CrossEntropyLoss that raises (a float target must be class
        probabilities shaped [N, C]), and the bare `except` turned the failure
        into a sentinel loss of 999.0 for BOTH the baseline and the post-op
        score -- so delta was always exactly 0.0 and surgery could never be
        reported as successful. For MSELoss it silently broadcast [N] against
        [N, 1] into an [N, N] comparison and produced a meaningless number.
        """
        y_t = torch.as_tensor(np.asarray(y))
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            return y_t.long()
        y_t = y_t.float()
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
        return y_t

    def _eval_loss(self, X, y) -> float:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X)
            try:
                return float(self.criterion(preds, y).item())
            except TypeError:
                return float(self.criterion(preds, y, X).item())

    def _restore_full_training(self, state, chains):
        self.model.load_state_dict(state)
        if chains and hasattr(self.model, 'chains'):
            self.model.chains = copy.deepcopy(chains)
        
        # Unfreeze all
        for param in self.model.parameters():
            param.requires_grad = True
        # Rebind main optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


# --------------------------------------------------------------------------
def fit_with_restarts(model_factory: Callable[[int], nn.Module],
                      X_train, y_train, X_val, y_val,
                      epochs: int,
                      restarts: int = 4,
                      seed0: int = 0,
                      name: str = "GAI",
                      **optimizer_kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """Run `restarts` independent searches on a SPLIT budget, keep the best.

    Why this exists. Measured on `y = sin(pi*x1) + x2^2` at hidden_dim=1, all
    arms given an identical 6000-epoch budget, 28 seeds:

        random operators  + 8 restarts   ->   0/28 exact recoveries
        one long search   (no restarts)  ->   3/28
        correct operators + 8 restarts   ->  28/28

    Restarts are worth 1/8 -> 8/8 once the operators are right (p = 0.0014) and
    the search forfeits that entirely by spending its whole budget on a single
    trajectory. This gives it back.

    IT IS NOT A FREE WIN, and the split matters. Eight 750-epoch searches
    improve the MEDIAN 4.7x but cut exact recoveries to 1/8, because restarts
    and operator search compete for the same epochs: 750 is ample to fit weights
    once operators are known and far too few to discover them. Prefer few long
    restarts (2 x 3000) over many short ones (8 x 750) when exact recovery is
    the goal; invert that when typical-case error is the goal.

    `model_factory(seed)` must return a FRESH model -- reusing one across
    restarts would defeat the point.

    Selection is on validation. The test split is never touched here.
    """
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}")
    per = max(1, epochs // restarts)
    crit = optimizer_kwargs.get('loss_fn') or nn.MSELoss()

    best_model, best_val, runs = None, float('inf'), []
    for r in range(restarts):
        seed = seed0 * 1000 + r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = model_factory(seed)
        opt = GAIOptimizer(model, **optimizer_kwargs)
        _, history = opt.fit(X_train, y_train, X_val, y_val,
                             epochs=per, name=f"{name}/r{r}")
        opt.model.eval()
        with torch.no_grad():
            xv = torch.as_tensor(X_val).float()
            yv = torch.as_tensor(y_val).float()
            if yv.dim() == 1:
                yv = yv.unsqueeze(1)
            val = float(crit(opt.model(xv), yv))
        runs.append({'restart': r, 'seed': seed, 'val': val,
                     'mutations': len(opt.mutation_log),
                     'structure': opt.model.get_structure()
                     if hasattr(opt.model, 'get_structure') else None})
        print(f"[{name}] restart {r}: val {val:.6e} "
              f"({len(opt.mutation_log)} mutations)", flush=True)
        if val < best_val:
            best_val, best_model = val, opt.model

    return best_model, {'best_val': best_val, 'epochs_per_restart': per,
                        'restarts': restarts, 'runs': runs}
