# Testing & Test Suite Documentation

> Test suite coverage, execution procedures, and verification breakdown for `pqblocks-anneal-sa`.

---

## 1. Test Suite Summary

The test suite contains **69 tests** covering all aspects of the simulated annealing algorithm, input validation, temperature schedules, and CLI execution.

- **Status**: 69 passed, 0 failed
- **Runtime**: ~3.5 seconds
- **Statement Coverage**: 72% across 356 statements (100% in `__init__`, 98% in `reference`, 93% in `kernel`, 90% in `validation`, 89% in `solver`)
- **Python Matrix**: Fully verified on Python 3.11, 3.12, 3.13, and 3.14

---

## 2. Test File Breakdown

| Test File | Tests | Focus Area |
|---|---|---|
| `test_cli.py` | 8 | CLI argument parsing, valid runs, edge cases (missing flags, malformed matrices) |
| `test_coverage.py` | 8 | Guard conditions, temperature derivations, and edge parameter validations |
| `test_kernel.py` | 6 | Numba kernel deterministic execution, energy consistency, and schedule variants |
| `test_reference.py` | 3 | Worked example ground state, incremental field updates vs recomputation |
| `test_sa.py` | 28 | Comprehensive solver API, parameter auto-derivation, edge case handling, cooling schedules |
| `test_validation.py` | 16 | Dimension mismatch, non-finite values, asymmetric coupling symmetrization, overflow guards |

---

## 3. Running the Tests

To run the complete test suite:

```bash
pytest -v
```

To run with coverage reporting:

```bash
pytest --cov=pqblocks_anneal_sa --cov-report=term
```
