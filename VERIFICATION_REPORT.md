# Verification Report: pqblocks-anneal-sa

> Formal release verification report for `pqblocks-anneal-sa`.

---

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Product** | `pqblocks-anneal-sa` |
| **Package Name** | `pqblocks_anneal_sa` |
| **Version** | `0.1.0` |
| **Classification** | Quantum-inspired classical optimisation: not cryptography, no security claim |
| **Verification Status** | **PASSED** |
| **Test Matrix** | Python 3.11, 3.12, 3.13, 3.14 |
| **Test Results** | 69 passed, 0 failed across all supported versions |
| **Leak Gate** | 0 hits (23 files scanned) |

---

## 2. Verification Gate Results

| Check ID | Verification Item | Target | Result | Status |
|---|---|---|---|---|
| **CHK-01** | Test Suite Completeness | 69 passing tests | 69 passed, 0 failed | **PASS** |
| **CHK-02** | Python Compatibility | Python 3.11–3.14 | Clean execution on all 4 versions | **PASS** |
| **CHK-03** | Source & Wheel Build | PEP 517 / PEP 621 | Valid sdist and wheel generated | **PASS** |
| **CHK-04** | Package Integrity | `twine check --strict` | Passed without warnings | **PASS** |
| **CHK-05** | Public Leak Gate | 0 internal leaks | 0 hits across all files | **PASS** |
| **CHK-06** | Example Verification | $\ge 2$ functioning examples | `quickstart.py` & `02_simulated_annealing_schedules.py` passed | **PASS** |
| **CHK-07** | CLI Entrypoint | Entrypoint `pqblocks-anneal-sa` | Validated via `test_cli.py` and subprocess | **PASS** |

---

## 3. Algorithmic Correctness & Temperature Derivation

The implementation verifies:
- Mathematical correctness of automatic temperature bounds:
  $$\beta_0 = \frac{\ln(2)}{\Delta E_{\text{mean}}}$$
  $$\beta_f = \frac{\ln(10000)}{\Delta E_{\text{min}}}$$
- Energy conservation and delta energy updates across spin flips:
  $$\Delta E_i = 2 s_i h_i + 2 s_i \sum_{j} J_{ij} s_j$$
- Complete recovery of verified ground state configurations on canonical test instances.

---

## 4. Conclusion

The package `pqblocks-anneal-sa` has satisfied all release verification requirements and is verified for release.
