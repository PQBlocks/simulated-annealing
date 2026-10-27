# Changelog

All notable changes to `pqblocks-anneal-sa` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-26

### Added
- Simulated Annealing heuristic solver for Ising and QUBO optimization.
- Automatic inverse temperature derivation ($\beta_0$, $\beta_f$) based on coefficient distributions.
- Multiple cooling schedule implementations: geometric, linear, and logarithmic.
- Optimized Numba/NumPy kernel with fast incremental local field updates.
- Defensive validation rejecting NaN/inf coefficients, non-zero diagonals, and dimension mismatches.
- Command-line interface `pqblocks-anneal-sa` supporting JSON and summary output formats.
- Comprehensive test suite with 69 passing tests across Python 3.11, 3.12, 3.13, and 3.14.
- Quickstart and schedule evaluation examples.
