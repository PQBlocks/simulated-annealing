"""Example 02: Evaluating different annealing cooling schedules and auto-derived temperatures."""

import numpy as np
from pqblocks_anneal_sa import simulated_annealing


def main():
    # 4-spin frustrated Ising model
    J = np.array([
        [0.0, 1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0, -1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0, 0.0],
    ])
    h = np.array([0.5, -0.3, 0.2, -0.1])

    print("--- Simulated Annealing Schedule Comparison ---")
    schedules = ["geometric", "linear", "logarithmic"]

    for sched in schedules:
        res = simulated_annealing(
            J=J,
            h=h,
            num_sweeps=200,
            num_reads=10,
            cooling_schedule=sched,
            seed=42,
        )
        best_energy = float(np.min(res["energies"]))
        best_idx = int(np.argmin(res["energies"]))
        best_sample = res["samples"][best_idx].tolist()

        info = res["info"]
        print(f"\nCooling Schedule: {sched}")
        print(f"  Auto-derived Beta_0: {info['beta_0']:.4f}")
        print(f"  Auto-derived Beta_f: {info['beta_f']:.4f}")
        print(f"  Best Energy Found:   {best_energy:.6f}")
        print(f"  Ground State Config: {best_sample}")

    print("\nSchedule evaluation completed successfully.")


if __name__ == "__main__":
    main()
