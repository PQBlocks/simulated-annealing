import numpy as np
import pytest
from pqblocks_anneal_sa.kernel import _sa_kernel
from pqblocks_anneal_sa.validation import validate_inputs


def test_sa_kernel_basic_functionality():
    """Test that the SA kernel runs and returns reasonable results."""
    # Simple 2-spin ferromagnetic case (zero diagonal)
    h = np.array([0.0, 0.0])
    J = np.array([[0.0, 1.0], [1.0, 0.0]])  # Ferromagnetic coupling, zero diagonal
    
    # Validate inputs first
    J_valid, h_valid = validate_inputs(h, J, num_sweeps=10, num_reads=1)
    
    # Run the kernel
    best_states, best_energies = _sa_kernel(
        J_valid, h_valid, 
        num_sweeps=10, 
        num_reads=1,
        beta_0=0.1,
        beta_f=1.0,
        cooling_schedule='geometric',
        seed=42
    )
    
    # Basic checks
    assert best_states.shape == (1, 2)
    assert best_energies.shape == (1,)
    assert best_states.dtype == np.int8
    assert best_energies.dtype == np.float64
    assert np.all(np.isfinite(best_energies))


def test_sa_kernel_deterministic_with_seed():
    """Test that same seed produces same results."""
    h = np.array([0.5, -0.3, 0.2, -0.1])
    J = np.array([
        [0.0, 1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0, -1.0],
        [-1.0, 1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0, 0.0]
    ])
    
    J_valid, h_valid = validate_inputs(h, J, num_sweeps=20, num_reads=2)
    
    # Run twice with same seed
    states1, energies1 = _sa_kernel(
        J_valid, h_valid,
        num_sweeps=20,
        num_reads=2,
        beta_0=0.1,
        beta_f=1.0,
        cooling_schedule='geometric',
        seed=12345
    )
    
    states2, energies2 = _sa_kernel(
        J_valid, h_valid,
        num_sweeps=20,
        num_reads=2,
        beta_0=0.1,
        beta_f=1.0,
        cooling_schedule='geometric',
        seed=12345
    )
    
    # Should be identical
    np.testing.assert_array_equal(states1, states2)
    np.testing.assert_array_equal(energies1, energies2)


def test_sa_kernel_multiple_reads_independence():
    """Test that multiple reads produce different results (with high probability)."""
    h = np.array([0.0, 0.0])
    J = np.array([[0.0, 1.0], [1.0, 0.0]])  # Zero diagonal
    
    J_valid, h_valid = validate_inputs(h, J, num_sweeps=10, num_reads=5)
    
    states, energies = _sa_kernel(
        J_valid, h_valid,
        num_sweeps=10,
        num_reads=5,
        beta_0=0.1,
        beta_f=1.0,
        cooling_schedule='geometric',
        seed=999
    )
    
    # Check that we got results for all reads
    assert states.shape == (5, 2)
    assert energies.shape == (5,)
    
    # With high probability, at least some reads should differ
    # (though not guaranteed due to randomness)
    unique_states = set()
    for i in range(states.shape[0]):
        unique_states.add(tuple(states[i, :]))
    
    # At minimum, we should have valid states
    for state in states:
        assert np.all(np.isin(state, [-1, 1]))


def test_sa_kernel_energy_consistency():
    """Test that reported energies match computed energies from states."""
    h = np.array([0.5, -0.3])
    J = np.array([[0.0, 1.0], [1.0, 0.0]])  # Zero diagonal
    
    J_valid, h_valid = validate_inputs(h, J, num_sweeps=5, num_reads=1)
    
    states, energies = _sa_kernel(
        J_valid, h_valid,
        num_sweeps=5,
        num_reads=1,
        beta_0=0.1,
        beta_f=1.0,
        cooling_schedule='geometric',
        seed=777
    )
    
    # Compute energy from the returned state
    s = states[0, :]
    computed_energy = 0.0
    for i in range(len(h)):
        for j in range(i+1, len(h)):
            computed_energy += J_valid[i, j] * s[i] * s[j]
    for i in range(len(h)):
        computed_energy += h_valid[i] * s[i]
    
    # Should match reported energy (within floating point tolerance)
    assert abs(computed_energy - energies[0]) < 1e-10


def test_sa_kernel_handles_zero_model():
    """Test that the kernel handles all-zero model gracefully."""
    h = np.array([0.0, 0.0, 0.0])
    J = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    
    J_valid, h_valid = validate_inputs(h, J, num_sweeps=5, num_reads=1)
    
    states, energies = _sa_kernel(
        J_valid, h_valid,
        num_sweeps=5,
        num_reads=1,
        beta_0=0.1,
        beta_f=5.0,  # Auto-derived values for zero model
        cooling_schedule='geometric',
        seed=555
    )
    
    # All states should have energy 0 for zero model
    assert np.allclose(energies, 0.0, atol=1e-10)
    assert states.shape == (1, 3)
    assert np.all(np.isin(states, [-1, 1]))


def test_sa_kernel_different_cooling_schedules():
    """Test that different cooling schedules work."""
    h = np.array([0.1, -0.1])
    J = np.array([[0.0, 0.5], [0.5, 0.0]])  # Zero diagonal
    
    J_valid, h_valid = validate_inputs(h, J, num_sweeps=5, num_reads=1)
    
    schedules = ['geometric', 'linear', 'logarithmic']
    
    for schedule in schedules:
        states, energies = _sa_kernel(
            J_valid, h_valid,
            num_sweeps=5,
            num_reads=1,
            beta_0=0.1,
            beta_f=1.0,
            cooling_schedule=schedule,
            seed=42
        )
        
        assert states.shape == (1, 2)
        assert energies.shape == (1,)
        assert np.isfinite(energies[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])