# Verification Evidence & Traceability

> Detailed execution evidence, command outputs, and reproducibility artifacts for `pqblocks-anneal-sa`.

---

## 1. Clean-Room Test Execution Evidence

Full test execution output in a clean virtual environment running Python 3.14.8:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: /tmp/pqblocks-gate-sa/tests
configfile: pyproject.toml
testpaths: tests
plugins: hypothesis-6.168.0, cov-7.1.0, anyio-4.15.1
collected 69 items

tests/test_cli.py ........                                               [ 11%]
tests/test_coverage.py ........                                          [ 23%]
tests/test_kernel.py ......                                              [ 31%]
tests/test_reference.py ...                                              [ 36%]
tests/test_sa.py ............................                            [ 76%]
tests/test_validation.py ................                                [100%]

============================== 69 passed in 3.49s ==============================
```

---

## 2. Command-Line Interface (CLI) Execution Evidence

```text
$ pqblocks-anneal-sa --help
usage: pqblocks-anneal-sa [-h] --j-matrix J_MATRIX --h-vector H_VECTOR [--sweeps SWEEPS] [--reads READS] [--seed SEED] [--output {json,summary}]

Simulated Annealing solver for Ising models

options:
  -h, --help            show this help message and exit
  --j-matrix J_MATRIX   Coupling matrix J as a string, e.g., "[[0, 1], [1, 0]]"
  --h-vector H_VECTOR   Linear biases h as a string, e.g., "[0.5, -0.3]"
  --sweeps SWEEPS       Number of sweeps per read (default: max(1000, 10*N))
  --reads READS         Number of independent reads (default: 10)
  --seed SEED           Random seed for reproducibility
  --output {json,summary}
                        Output format: json for machine-readable, summary for human-readable (default: json)
```

### Running CLI with Worked Example:

```bash
$ pqblocks-anneal-sa --j-matrix "[[0, 1, -1, 0], [1, 0, 1, -1], [-1, 1, 0, 1], [0, -1, 1, 0]]" \
                     --h-vector "[0.5, -0.3, 0.2, -0.1]" --sweeps 100 --reads 5 --seed 42 --output json
{
  "samples": [
    [-1, 1, -1, 1],
    [-1, 1, -1, 1],
    [-1, 1, -1, 1],
    [-1, 1, -1, 1],
    [-1, 1, -1, 1]
  ],
  "energies": [
    -6.1,
    -6.1,
    -6.1,
    -6.1,
    -6.1
  ],
  "num_occurrences": [1, 1, 1, 1, 1],
  "info": {
    "beta_0": 0.12487449514757393,
    "beta_f": 46.051701859880916,
    "cooling_schedule": "geometric",
    "num_sweeps": 100,
    "num_reads": 5,
    "seed": 42
  }
}
```

---

## 3. Example Execution Evidence

### Quickstart Execution:

```bash
$ python3 examples/quickstart.py
Best energy: -6.100000
```

### Multiple Schedule Evaluation:

```bash
$ python3 examples/02_simulated_annealing_schedules.py
--- Simulated Annealing Schedule Comparison ---

Cooling Schedule: geometric
  Auto-derived Beta_0: 0.1249
  Auto-derived Beta_f: 46.0517
  Best Energy Found:   -6.100000
  Ground State Config: [-1, 1, -1, 1]

Cooling Schedule: linear
  Auto-derived Beta_0: 0.1249
  Auto-derived Beta_f: 46.0517
  Best Energy Found:   -6.100000
  Ground State Config: [-1, 1, -1, 1]

Cooling Schedule: logarithmic
  Auto-derived Beta_0: 0.1249
  Auto-derived Beta_f: 46.0517
  Best Energy Found:   -6.100000
  Ground State Config: [-1, 1, -1, 1]

Schedule evaluation completed successfully.
```
