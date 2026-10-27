# Use Cases & Integration Patterns

> Practical applications and integration scenarios for `pqblocks-anneal-sa`.

---

## 1. Zero-Tuning Combinatorial Optimization

Traditional simulated annealing solvers require extensive manual experimentation to select starting temperatures, ending temperatures, and decay constants. If $\beta_0$ is too high, the solver freezes immediately in suboptimal local minima; if $\beta_f$ is too low, the system continues fluctuating without settling into a ground state.

`pqblocks-anneal-sa` solves this by automatically analyzing the coefficient distribution:

```python
import numpy as np
from pqblocks_anneal_sa import simulated_annealing

# Coupling matrix for graph partitioning or portfolio selection
J = np.random.randn(20, 20)
J = (J + J.T) / 2
np.fill_diagonal(J, 0.0)
h = np.random.randn(20) * 0.1

# Auto-derives beta_0 and beta_f from coefficient scales
result = simulated_annealing(J, h, num_sweeps=1000, num_reads=20, seed=123)
print(f"Optimal energy: {min(result['energies']):.4f}")
```

---

## 2. Comparing Annealing Schedules

Different optimization landscapes benefit from distinct cooling dynamics:
- **Geometric (`geometric`)**: Default schedule ($T_{k+1} = \alpha T_k$), ideal for rugged spin-glass landscapes with many local barriers.
- **Linear (`linear`)**: Linear temperature reduction, effective for smooth energy landscapes.
- **Logarithmic (`logarithmic`)**: Slower cooling rate providing theoretical asymptotic convergence guarantees on small instances.

```python
# Select cooling schedule explicitly
result = simulated_annealing(J, h, cooling_schedule="geometric")
```

---

## 3. Shell Pipeline and Batch Ingestion

`pqblocks-anneal-sa` includes a command-line interface that allows DevOps pipelines and shell scripts to evaluate optimization problems without writing Python wrappers:

```bash
# Output JSON for automated ingestion in downstream CI/CD pipelines
pqblocks-anneal-sa --j-matrix "[[0, 1], [1, 0]]" --h-vector "[0.5, -0.3]" --output json > solution.json
```
