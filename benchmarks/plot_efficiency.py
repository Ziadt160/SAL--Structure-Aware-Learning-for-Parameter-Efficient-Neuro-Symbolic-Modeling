import matplotlib.pyplot as plt

# Data collected from the experiment output
results = [
    {'hidden_dim': 128, 'params': 268298, 'accuracy': 0.9090},
    {'hidden_dim': 96, 'params': 188938, 'accuracy': 0.9021},
    {'hidden_dim': 64, 'params': 117770, 'accuracy': 0.8904},
    {'hidden_dim': 48, 'params': 85258, 'accuracy': 0.8438},
    {'hidden_dim': 32, 'params': 54794, 'accuracy': 0.6883},
    {'hidden_dim': 24, 'params': 40330, 'accuracy': 0.7191},
    {'hidden_dim': 16, 'params': 26378, 'accuracy': 0.4670},
    {'hidden_dim': 12, 'params': 19594, 'accuracy': 0.4470},
]

best_structure = [['relu', 'relu', 'gaussian'], ['relu', 'relu', 'sigmoid']]

res_dicts = sorted(results, key=lambda x: x['params'])
params_x = [r['params'] for r in res_dicts]
acc_y = [r['accuracy'] for r in res_dicts]

plt.figure(figsize=(10, 6))
plt.plot(params_x, acc_y, marker='o', linestyle='-', color='b')
plt.title(f"Parameter Efficiency: Best GAI Structure\nOps: {best_structure}")
plt.xlabel("Number of Parameters")
plt.ylabel("Test Accuracy")
plt.grid(True, alpha=0.3)
plt.xscale('log') 

# Annotate points
for i, txt in enumerate([r['hidden_dim'] for r in res_dicts]):
    plt.annotate(f"H={txt}", (params_x[i], acc_y[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig('nas_efficiency_sweep.png')
print("Saved plot to nas_efficiency_sweep.png")
