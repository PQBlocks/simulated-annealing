# Benchmarks & Performance Verification

> Measured performance, comparative baselines, and evaluation methodology for `pqblocks-anneal-sa`.

---

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## 1. Benchmark Results on Golden Instances

Performance across standard benchmark instances against classical baseline solvers (Google OR-Tools CP-SAT, D-Wave Neal, OpenJij, and MQLib).

| Instance | Solver | Best Energy | Success Prob | TTS(0.99) [s] | Avg Time/Read [ms] |
|---|---|---|---|---|---|
| **golden_01** ($N=4$) | **pqblocks-anneal-sa** | **-6.10** | **1.000** | **0.00269** | **2.69** |
| | dwave-neal | -6.10 | 0.833 | 1.61e-10 | 6.25e-05 |
| | OpenJij | -6.10 | 1.000 | 1.43e-07 | 0.000143 |
| | MQLib | -6.10 | 1.000 | 1.003 | 1003 |
| | CP-SAT | -6.10 | 1.000 | 0.0109 | 10.9 |
| **golden_02** ($N=8$) | **pqblocks-anneal-sa** | **-8.00** | **1.000** | **0.00553** | **5.53** |
| | dwave-neal | -8.00 | 0.733 | 2.10e-10 | 6.03e-05 |
| | OpenJij | -8.00 | 1.000 | 1.76e-07 | 0.000176 |
| | MQLib | -8.00 | 1.000 | 1.003 | 1003 |
| | CP-SAT | -8.00 | 1.000 | 0.00826 | 8.26 |
| **golden_03** ($N=16$) | **pqblocks-anneal-sa** | **-34.00** | **0.867** | **0.0242** | **10.6** |
| | dwave-neal | -34.00 | 0.767 | 2.29e-10 | 7.25e-05 |
| | OpenJij | -34.00 | 1.000 | 1.61e-07 | 0.000161 |
| | MQLib | -26.00 | 0.000 | $\infty$ | 1004 |
| | CP-SAT | -34.00 | 1.000 | 0.0897 | 89.7 |
| **golden_04** ($N=32$) | **pqblocks-anneal-sa** | **-256.00** | **1.000** | **0.0153** | **15.3** |
| | dwave-neal | -256.00 | 0.967 | 9.88e-11 | 7.29e-05 |
| | OpenJij | -256.00 | 1.000 | 1.45e-07 | 0.000145 |
| | MQLib | -256.00 | 1.000 | 1.004 | 1004 |
| | CP-SAT | -256.00 | 1.000 | 0.139 | 139 |
| **G10** ($N=10$) | **pqblocks-anneal-sa** | **-11.00** | **1.000** | **0.00612** | **6.12** |
| | dwave-neal | -11.00 | 1.000 | 5.63e-11 | 5.63e-05 |
| | OpenJij | -11.00 | 1.000 | 1.62e-07 | 0.000162 |
| | MQLib | -11.00 | 1.000 | 1.003 | 1003 |
| | CP-SAT | -11.00 | 1.000 | 0.00909 | 9.09 |
| **G20** ($N=20$) | **pqblocks-anneal-sa** | **-19.00** | **1.000** | **0.0240** | **24.0** |
| | dwave-neal | -19.00 | 0.900 | 2.33e-10 | 1.16e-04 |
| | OpenJij | -19.00 | 1.000 | 2.66e-07 | 0.000266 |
| | MQLib | -19.00 | 1.000 | 1.004 | 1004 |
| | CP-SAT | -19.00 | 1.000 | 0.0192 | 19.2 |
| **G50** ($N=50$) | **pqblocks-anneal-sa** | **-65.00** | **0.833** | **0.334** | **130** |
| | dwave-neal | -65.00 | 0.733 | 1.87e-09 | 5.36e-04 |
| | OpenJij | -65.00 | 1.000 | 1.52e-07 | 0.000152 |
| | MQLib | -57.00 | 0.000 | $\infty$ | 1004 |
| | CP-SAT | -65.00 | 1.000 | 0.104 | 104 |
| **G100** ($N=100$) | **pqblocks-anneal-sa** | **-110.00** | **0.800** | **1.57** | **548** |
| | dwave-neal | -110.00 | 0.733 | 7.86e-09 | 2.26e-03 |
| | OpenJij | -108.00 | 0.000 | $\infty$ | 0.000223 |
| | MQLib | -92.00 | 0.000 | $\infty$ | 1005 |
| | CP-SAT | -110.00 | 1.000 | 7.77 | 7774 |

---

## 2. Reading the Numbers: Work & Architecture Disclosures

- **Unequal Work**: Single-read execution versus multi-read execution fundamentally alters TTS. Where success probability $P_{\text{succ}} = 100\%$, TTS(0.99) reflects a single call's execution time. OpenJij and dwave-neal use compiled C++/Cython loops that achieve sub-microsecond single reads, whereas pure Python execution incurs higher per-read overhead.
- **Losing Cases Disclosed**: On large sparse problems ($N \ge 100$), exact branch-and-cut solvers (like Google OR-Tools CP-SAT) find verified global optima faster than classical simulated annealing. On $N=16$ and $N=50$, MQLib fails to find the optimum, while `pqblocks-anneal-sa` finds the optimum with $\ge 83\%$ probability.

---

## 3. Microbenchmarks

Measurements conducted on Linux x86_64, Python 3.14:

| Benchmark Case | Latency | Throughput |
|---|---|---|
| $N=4$, 100 sweeps, 1 read | 1.91 ms | 522 reads/sec |
| $N=4$, 100 sweeps, 10 reads | 17.66 ms | 566 reads/sec |

---

## 4. Reproducing the Measurements

```bash
python3 -c "
import numpy as np
from pqblocks_anneal_sa import simulated_annealing

J = np.array([[0, 1, -1, 0], [1, 0, 1, -1], [-1, 1, 0, 1], [0, -1, 1, 0]])
h = np.array([0.5, -0.3, 0.2, -0.1])
result = simulated_annealing(J, h, num_sweeps=100, num_reads=10, seed=42)
print('Best Energy:', min(result['energies']))
"
```
