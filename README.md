# pqblocks-anneal-sa

> Classical Simulated Annealing heuristic solver for Ising spin-glass and QUBO optimization problems with automatic parameter derivation.

[![CI](https://github.com/PQBlocks/simulated-annealing/actions/workflows/ci.yml/badge.svg)](https://github.com/PQBlocks/simulated-annealing/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## Overview

`pqblocks-anneal-sa` is a classical Simulated Annealing solver for Quadratic Unconstrained Binary Optimization (QUBO) and Ising optimization problems.

Inspired by thermal annealing in statistical physics, the solver progressively cools a spin system according to configurable temperature schedules (geometric, linear, or logarithmic). It features automatic parameter derivation to compute initial and final inverse temperatures ($\beta_0$, $\beta_f$) directly from the coupling matrix $J$ and linear bias vector $h$, eliminating manual hyperparameter tuning.

### Key Highlights

- **Automatic Temperature Derivation**: Derives high-temperature initialization ($\beta_0 \approx \ln(2) / \Delta E_{\text{mean}}$) and low-temperature freezing ($\beta_f \approx \ln(10000) / \Delta E_{\text{min}}$) automatically from problem coefficients.
- **Multiple Cooling Schedules**: Supports geometric, linear, and logarithmic temperature decay schedules.
- **Optimized Kernel Execution**: Fast incremental local-field updates ($\Delta E_i = 2 s_i h_i + 2 s_i \sum_j J_{ij} s_j$) with parallel reads.
- **Command-Line Interface**: Built-in CLI for batch solving and JSON output pipelines.

---

## Installation

```bash
pip install pqblocks-anneal-sa
```

### Dependencies

Requires Python 3.11+, `numpy`, `scipy`, and `dimod`.

---

## Quickstart

```python
import numpy as np
from pqblocks_anneal_sa import simulated_annealing

# 1. Define 4-spin Ising problem (couplings J and linear biases h)
J = np.array([
    [0.0, 1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0, -1.0],
    [-1.0, 1.0, 0.0, 1.0],
    [0.0, -1.0, 1.0, 0.0],
])
h = np.array([0.5, -0.3, 0.2, -0.1])

# 2. Solve with auto-derived parameters
result = simulated_annealing(J, h, num_sweeps=100, num_reads=10, seed=42)

# 3. Retrieve best solution
best_idx = np.argmin(result["energies"])
print(f"Best energy: {result['energies'][best_idx]:.6f}")
print(f"Sample: {result['samples'][best_idx].tolist()}")
```

---

## Command-Line Usage

```bash
# Solve directly from the shell with JSON output
pqblocks-anneal-sa --j-matrix "[[0, 1], [1, 0]]" --h-vector "[0.5, -0.3]" --output json
```

---

## Fair Benchmarking Disclaimer

When benchmarking simulated annealing against exact or heuristic solvers (such as Google OR-Tools CP-SAT, D-Wave Neal, or OpenJij):
- **Work Normalization**: State total sweeps and read counts for all annealing solvers.
- **Latency vs Quality**: Solvers with single-read execution report lower raw latency but may exhibit lower success probabilities than multi-read ensembles.
- **Problem Size**: Classical SA excels on small-to-medium dense problems ($N \le 50$); exact solvers like CP-SAT may prove faster on structured sparse problems.

---

## Verification & Testing

Tested across Python 3.11, 3.12, 3.13, and 3.14.

```bash
pytest
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
