
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class SensitivityProbe:
    """
    A Diagnostic Tool for Neural Architectures.
    It attaches to a model and measures 'Vital Signs':
    1. Gradient Flow (Is the layer learning?)
    2. Activation Variance (Is the layer doing anything?)
    3. Output Magnitude (Is the layer exploding?)
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.stats = defaultdict(lambda: {"grad_norm": [], "act_var": [], "act_mean": []})
        
        # Identify "Probe-able" layers (currently targeting Linear and Custom Nodes)
        self.target_modules = {}
        for name, mod in model.named_modules():
            # We target Linear layers as proxies for the "Node" they belong to
            if isinstance(mod, nn.Linear):
                self.target_modules[name] = mod

    def attach(self):
        """Registers forward and backward hooks."""
        self.remove() # Clean up existing
        
        for name, mod in self.target_modules.items():
            # Forward Hook for Activations
            self.hooks.append(mod.register_forward_hook(self._get_fwd_hook(name)))
            # Backward Hook for Gradients
            self.hooks.append(mod.register_full_backward_hook(self._get_bwd_hook(name)))
            
    def remove(self):
        """Removes all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _get_fwd_hook(self, name):
        def hook(module, input, output):
            # Output is [Batch, OutDim]
            if isinstance(output, torch.Tensor):
                # Calculate Information Entropy / Variance 
                # (Low variance = Dead/Constant output)
                var = torch.var(output, dim=0).mean().item()
                mean = torch.mean(output).item()
                self.stats[name]["act_var"].append(var)
                self.stats[name]["act_mean"].append(mean)
        return hook

    def _get_bwd_hook(self, name):
        def hook(module, grad_input, grad_output):
            # grad_output is tuple, usually (tensor,)
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                g = grad_output[0]
                norm = torch.norm(g).item()
                self.stats[name]["grad_norm"].append(norm)
        return hook

    def clear_stats(self):
        self.stats = defaultdict(lambda: {"grad_norm": [], "act_var": [], "act_mean": []})

    def generate_report(self) -> Dict[str, Any]:
        """
        Analyzes the collected stats and returns a health report.
        """
        report = {}
        
        for name, data in self.stats.items():
            if not data["grad_norm"]: continue
            
            avg_grad = np.mean(data["grad_norm"])
            avg_var = np.mean(data["act_var"])
            
            status = "HEALTHY"
            issues = []
            
            # --- DIAGNOSIS LOGIC ---
            
            # 1. Vanishing Gradient / Dead Unit
            if avg_grad < 1e-6:
                status = "CRITICAL"
                issues.append("DEAD_GRADIENT (Vanishing or Disconnected)")
                
            # 2. Saturation / Uselessness
            if avg_var < 1e-5:
                status = "WARNING"
                issues.append("LOW_VARIANCE (Saturated Activation or Constant Output)")
            
            # 3. Exploding Gradient
            if avg_grad > 100.0:
                status = "CRITICAL"
                issues.append("EXPLODING_GRADIENT (Unstable)")

            report[name] = {
                "status": status,
                "issues": issues,
                "metrics": {
                    "gradient_flow": f"{avg_grad:.2e}",
                    "activation_entropy": f"{avg_var:.2e}"
                }
            }
            
        return report

    def print_summary(self):
        print("\n=== SYSTEM HEALTH DIAGNOSIS ===")
        report = self.generate_report()
        if not report:
            print("No data collected yet. (Run a forward/backward pass)")
            return
            
        for name, info in report.items():
            status_icon = "🟢"
            if info['status'] == 'WARNING': status_icon = "ww"
            if info['status'] == 'CRITICAL': status_icon = "ww"
            
            print(f"{status_icon} [{name}] {info['status']}")
            if info['issues']:
                for issue in info['issues']:
                    print(f"    - {issue}")
            else:
                 print(f"    - Flow: {info['metrics']['gradient_flow']} | Var: {info['metrics']['activation_entropy']}")
