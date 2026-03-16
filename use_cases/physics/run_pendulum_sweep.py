
import subprocess
import itertools
import pandas as pd
import os
import re

# --- CONFIGURATION SPACE ---
# Strategies: Importance (GAI) vs Random (Baseline)
strategies = ['importance', 'random'] # limiting to importance for speed, user asked for 'different mixture'
chains_list = [2, 4, 6]
dropouts = [0.0, 0.05, 0.1]
epochs = 2000 

# If "all available" was strictly interpreted, we might add mutation strategies.
# But for now let's do a reasonable grid.

configs = list(itertools.product(strategies, chains_list, dropouts))

results = []

print(f"Starting Sweep over {len(configs)} configurations...")
print("="*60)

for strategy, chains, dropout in configs:
    print(f"\n>>> Running Config: Strategy={strategy}, Chains={chains}, Dropout={dropout}")
    
    cmd = [
        "python", "double_pendulum_benchmark.py",
        "--epochs", str(epochs),
        "--chains", str(chains),
        "--dropout", str(dropout),
        "--strategy", strategy
    ]
    
    # Run process
    try:
        # We want to capture the output to parse the Energy MSE
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout
        print(output) # Print to main log as well
        
        # Parse Metric
        # Look for [METRIC] Energy MSE: 0.123456
        match = re.search(r"Energy MSE: ([0-9.]+)", output)
        mse = float(match.group(1)) if match else None
        
        results.append({
            'Strategy': strategy,
            'Chains': chains,
            'Dropout': dropout,
            'Energy MSE': mse,
            'Status': 'Success'
        })
        
    except subprocess.CalledProcessError as e:
        print(f"!!! Error in config {strategy}-{chains}-{dropout}")
        print(e.stderr)
        results.append({
            'Strategy': strategy,
            'Chains': chains,
            'Dropout': dropout,
            'Energy MSE': None,
            'Status': 'Failed'
        })

print("\n" + "="*60)
print("SWEEP RESULTS")
print("="*60)
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Save simplified results
df.to_csv('pendulum_sweep_results.csv', index=False)
