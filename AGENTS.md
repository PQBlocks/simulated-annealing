# Autonomous Agent Integration Guide

> Operational instructions and best practices for AI and autonomous software agents integrating `pqblocks-anneal-sa`.

---

## 1. Package Purpose

`pqblocks-anneal-sa` provides a classical Simulated Annealing solver for Ising spin-glass models and Quadratic Unconstrained Binary Optimization (QUBO) problems.

---

## 2. Key APIs

```python
from pqblocks_anneal_sa import simulated_annealing

result = simulated_annealing(
    J=J_matrix,              # np.ndarray: (N, N) coupling matrix with zero diagonal
    h=h_vector,              # np.ndarray: (N,) linear bias vector
    num_sweeps=1000,         # int: sweeps per read (default: max(1000, 10*N))
    num_reads=10,            # int: independent annealing trajectories (default: 10)
    cooling_schedule="geometric", # str: "geometric", "linear", or "logarithmic"
    seed=42                  # int: optional deterministic random seed
)

# Output structure
samples = result["samples"]                  # np.ndarray: shape (num_reads, N) with values in {-1, +1}
energies = result["energies"]                # np.ndarray: shape (num_reads,)
info = result["info"]                        # dict: {"beta_0": float, "beta_f": float, ...}
```

---

## 3. Best Practices for Agents

1. **Matrix Formatting**: Coupling matrices must be 2D square symmetric arrays ($N \times N$) with zero diagonals ($J_{ii} = 0$). While the solver automatically symmetrizes asymmetric inputs, providing clean symmetric matrices avoids diagnostic warnings.
2. **Deterministic Reproducibility**: Always provide an integer seed (`seed=42`) when running tests or benchmarks to ensure bit-identical results.
3. **Auto-Derived Temperatures**: Omit `beta_0` and `beta_f` unless reproducing a specific academic schedule; the auto-derivation engine computes optimal boundary temperatures directly from coefficient scales.
4. **Ising vs QUBO Variables**: Native inputs assume spin variables $s_i \in \{-1, +1\}$. For binary variables $x_i \in \{0, 1\}$, convert problems using `pqblocks-qubo`.
