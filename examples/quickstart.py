import numpy as np
from pqblocks_anneal_sa import simulated_annealing

J = np.array([[0, 1, -1, 0], [1, 0, 1, -1], [-1, 1, 0, 1], [0, -1, 1, 0]])
h = np.array([0.5, -0.3, 0.2, -0.1])
res = simulated_annealing(J, h, num_sweeps=100, num_reads=5, seed=42)
best_idx = np.argmin(res['energies'])
print(f'Best energy: {res["energies"][best_idx]:.6f}')
