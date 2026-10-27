# Security Notes & Posture

> Security scope, numerical safeguards, and execution boundaries for `pqblocks-anneal-sa`.

---

## 1. Non-Cryptographic Classification

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

`pqblocks-anneal-sa` is a heuristic numerical solver for combinatorial optimization and physics-inspired simulated annealing. It does not implement, expose, or validate cryptographic primitives, key encapsulation mechanisms, digital signatures, or confidential data protection mechanisms.

---

## 2. Numerical Safeguards & Input Validation

The solver implements defensive validation to prevent numerical instabilities and crashes:
- **Exponential Overflow Guarding**: When evaluating the Metropolis acceptance criterion $\alpha = \exp(-\Delta E \cdot \beta)$, large positive energy jumps or high inverse temperatures could trigger IEEE 754 floating-point overflow. The kernel clips $\Delta E \cdot \beta$ to prevent mathematical overflow.
- **Coefficient Validation**: Input coupling matrices $J$ and linear vectors $h$ are strictly checked for non-finite values (NaN and $\pm\infty$), non-zero diagonals ($J_{ii} \ne 0$), and dimension mismatches.
- **Symmetry Enforcement**: Asymmetric coupling matrices are automatically detected and symmetrized ($J_{\text{sym}} = \frac{1}{2}(J + J^T)$) to maintain consistent physical Hamiltonian formulations.

---

## 3. Random Number Generation

The internal sampler utilizes a standard pseudo-random number generator (PRNG) parameterized by an explicit integer seed (`seed=42`) to enable deterministic replication of benchmark runs. This generator is intentionally non-cryptographic and must not be used for key derivation or cryptographic nonce generation.
