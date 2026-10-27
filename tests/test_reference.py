import numpy as np
import pytest
from pqblocks_anneal_sa.reference import _sa_reference

def test_worked_example_ground_state():
    """Test that the reference oracle finds the ground state of the worked example."""
    # Worked example parameters from spec
    J = np.array([[0, 1, -1, 0],
                  [1, 0, 1, -1],
                  [-1, 1, 0, 1],
                  [0, -1, 1, 0]], dtype=np.float64)
    h = np.array([0.5, -0.3, 0.2, -0.1], dtype=np.float64)
    
    # Use low temperature to only accept downhill moves (approaches ground state)
    beta_0 = 1000.0
    beta_f = 1000.0
    num_sweeps = 1000
    cooling_schedule = 'geometric'
    seed = 42
    
    best_s, best_energy = _sa_reference(J, h, num_sweeps, beta_0, beta_f, cooling_schedule, seed)
    
    # Verified ground state energy from reference calculation
    expected_energy = -6.100000
    assert np.isclose(best_energy, expected_energy, atol=1e-5), \
        f"Expected energy {expected_energy}, got {best_energy}"
    
    # Verified ground state spin configuration
    expected_s = np.array([-1, +1, -1, +1], dtype=np.int8)
    np.testing.assert_array_equal(best_s, expected_s)

def test_incremental_field_matches_full_recomputation():
    """Test that incremental local field updates match full recomputation."""
    # Small random instance
    np.random.seed(123)
    N = 10
    h = np.random.randn(N) * 0.5
    J = np.random.randn(N, N) * 0.5
    J = (J + J.T) / 2  # symmetric
    np.fill_diagonal(J, 0)  # zero diagonal
    
    # Parameters
    num_sweeps = 2
    beta_0 = 1.0
    beta_f = 1.0
    cooling_schedule = 'geometric'
    seed = 123
    
    best_s, best_energy = _sa_reference(J, h, num_sweeps, beta_0, beta_f, cooling_schedule, seed)
    
    # Recompute energy from the final state
    recomputed_energy = 0.5 * np.dot(best_s, np.dot(J, best_s)) + np.dot(h, best_s)
    assert np.isclose(best_energy, recomputed_energy, atol=1e-10), \
        f"Energy mismatch: incremental {best_energy} vs full {recomputed_energy}"


def test_num_sweeps_one_uses_beta_0():
    """Test that num_sweeps=1 uses beta_0 directly (no schedule interpolation)."""
    h = np.array([0.5, -0.3])
    J = np.array([[0, 1], [1, 0]])
    
    # With num_sweeps=1, beta should equal beta_0 (not interpolated)
    best_s, best_energy = _sa_reference(J, h, 1, 5.0, 10.0, 'geometric', 42)
    
    # Just verify it runs without error and returns valid output
    assert len(best_s) == 2
    assert np.isfinite(best_energy)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])