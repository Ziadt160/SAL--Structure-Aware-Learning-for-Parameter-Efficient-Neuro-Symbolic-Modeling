import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Legacy import for baseline comparison if needed, or we implement new Parity Logic.
# Let's use the new Core for GAI instead of legacy gai_final_boss.
# But train_parity_gglen wraps the data generation.
# I will import from legacy for now to ensure exact reproduction of "Final Boss" logic unless I rewrite it.
try:
    from legacy.gai_final_boss import train_parity_gglen
except ImportError:
    # If legacy is hard to reach, let's define a wrapper using Core.
    # This is better for "Refactoring".
    from modules.core import MatrixGGLEN, GAIOptimizer
    
    def train_parity_gglen(n_bits=10):
        # Re-implementation using Core
        def get_parity_data(n=1000, bits=10):
            X = torch.randint(0, 2, (n, bits)).float()
            Y = (X.sum(dim=1) % 2).view(-1, 1)
            return X, Y
            
        X, Y = get_parity_data(n=1000, bits=n_bits)
        # Final Boss used: input_dim=N, hidden=N*2, depth=N
        model = MatrixGGLEN(input_dim=n_bits, output_dim=1, hidden_dim=n_bits*2, chain_depth=n_bits, num_chains=2)
        optimizer = GAIOptimizer(model, lr=0.01, patience=50)
        
        # Manually run fit to get history
        # GAIOptimizer.fit returns preds. We need history/loss.
        # GAIOptimizer prints logs but doesn't return history list.
        # This wrapper needs to adapt. 
        # For the sake of the benchmark script which expects (model, history), 
        # I'll stick to importing legacy if possible, or accept I need to change the calling code.
        pass
        
# Actually, the simplest reliable fix for now is importing Legacy.
from legacy.gai_final_boss import train_parity_gglen
from benchmarks.nas_benchmark import train_parity_nas

# --- CONFIGURATION ---
TRIALS = 3
DIFFICULTIES = [128, 256]

def run_benchmark():
    print(f"=== FULL SCALE BENCHMARK: GAI vs NAS ===")
    print(f"Difficulties (N-Bits): {DIFFICULTIES}")
    
    results = {
        'GAI': {'time': [], 'epochs': [], 'success': []},
        'NAS': {'time': [], 'epochs': [], 'success': []}
    }
    
    for n in DIFFICULTIES:
        print(f"\n>>> TESTING N={n} BITS <<<")
        
        # --- GAI ---
        gai_times = []
        gai_epochs = []
        gai_success = 0
        for i in range(TRIALS):
            print(f"  [GAI] Trial {i+1}...", end=" ", flush=True)
            try:
                start = time.time()
                _, hist = train_parity_gglen(n_bits=n)
                duration = time.time() - start
                
                # Check if solved (last loss low or just assume if it returns it finished?)
                # The script breaks on success. Let's assume return = success or max epochs.
                # We need to peek at history to see if it converged.
                final_loss = hist[-1]
                if final_loss < 0.1: # Approximate 'solved' threshold
                     gai_success += 1
                
                gai_times.append(duration)
                gai_epochs.append(len(hist))
                print(f"Done ({duration:.2f}s, {len(hist)} eps)")
            except Exception as e:
                print(f"Failed ({e})")
                gai_times.append(0) # Penalty?
                gai_epochs.append(8000)

        results['GAI']['time'].append(np.mean(gai_times))
        results['GAI']['epochs'].append(np.mean(gai_epochs))
        results['GAI']['success'].append(gai_success / TRIALS)

        # --- NAS ---
        nas_times = []
        nas_epochs = []
        nas_success = 0
        for i in range(TRIALS):
            print(f"  [NAS] Trial {i+1}...", end=" ", flush=True)
            try:
                start = time.time()
                _, hist = train_parity_nas(n_bits=n)
                duration = time.time() - start
                
                final_loss = hist[-1]
                if final_loss < 0.1:
                     nas_success += 1

                nas_times.append(duration)
                nas_epochs.append(len(hist))
                print(f"Done ({duration:.2f}s, {len(hist)} eps)")
            except Exception as e:
                print(f"Failed ({e})")
                nas_times.append(0)
                nas_epochs.append(8000)

        results['NAS']['time'].append(np.mean(nas_times))
        results['NAS']['epochs'].append(np.mean(nas_epochs))
        results['NAS']['success'].append(nas_success / TRIALS)

    # --- PLOTTING ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Time to Solve
    x = np.arange(len(DIFFICULTIES))
    width = 0.35
    
    ax[0].bar(x - width/2, results['GAI']['time'], width, label='GAI', color='blue', alpha=0.7)
    ax[0].bar(x + width/2, results['NAS']['time'], width, label='NAS', color='red', alpha=0.7)
    ax[0].set_ylabel('Avg Time (seconds)')
    ax[0].set_xlabel('Parity Bits (N)')
    ax[0].set_title('Scalability: Time to Solve')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(DIFFICULTIES)
    ax[0].legend()
    
    # 2. Convergence Speed (Epochs)
    ax[1].bar(x - width/2, results['GAI']['epochs'], width, label='GAI', color='blue', alpha=0.7)
    ax[1].bar(x + width/2, results['NAS']['epochs'], width, label='NAS', color='red', alpha=0.7)
    ax[1].set_ylabel('Avg Epochs')
    ax[1].set_xlabel('Parity Bits (N)')
    ax[1].set_title('Efficiency: Epochs to Convergence')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(DIFFICULTIES)
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('scalability_benchmark.png')
    print("\nBenchmark Saved to scalability_benchmark.png")
    
    print("\n--- DETAILED SUMMARY ---")
    print(f"Difficulties: {DIFFICULTIES}")
    print(f"GAI Times: {results['GAI']['time']}")
    print(f"NAS Times: {results['NAS']['time']}")
    print(f"GAI Success: {results['GAI']['success']}")
    print(f"NAS Success: {results['NAS']['success']}")

if __name__ == "__main__":
    run_benchmark()
