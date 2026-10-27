import numpy as np
import pytest
from pqblocks_anneal_sa.validation import validate_inputs, BQMAdapter

def test_edge_empty_model_n0():
    """E01: Empty model (N=0) raises ValueError"""
    with pytest.raises(ValueError, match="Empty model: N=0 is not supported."):
        validate_inputs(np.array([]), np.array([]).reshape(0, 0))

def test_edge_single_spin_n1():
    """E02: Single spin (N=1) raises ValueError"""
    with pytest.raises(ValueError, match="Model must have at least 2 spins: N=1 is not supported."):
        validate_inputs(np.array([0.5]), np.array([[0.0]]))

def test_edge_dimension_mismatch():
    """E03: Dimension mismatch between h and J raises ValueError"""
    with pytest.raises(ValueError, match="Dimension mismatch between h and J: len\\(h\\)=2 vs J.shape=\\(3, 3\\)."):
        validate_inputs(np.array([0.5, 0.3]), np.array([[0, 1, -1], [1, 0, 1], [-1, 1, 0]]))

def test_edge_nan_inf_h():
    """E04: Non-finite / NaN in h raises ValueError"""
    with pytest.raises(ValueError, match="Linear coefficients h contain non-finite values \\(NaN or Inf\\)."):
        validate_inputs(np.array([0.5, np.nan]), np.array([[0, 1], [1, 0]]))

def test_edge_nan_inf_j():
    """E05: Non-finite / NaN in J raises ValueError"""
    with pytest.raises(ValueError, match="Coupling matrix J contains non-finite values \\(NaN or Inf\\)."):
        validate_inputs(np.array([0.5, 0.3]), np.array([[0, np.inf], [1, 0]]))

def test_edge_asymmetric_j_rejected():
    """E06: Asymmetric J (symmetrize=False) raises ValueError"""
    with pytest.raises(ValueError, match="Coupling matrix J must be symmetric."):
        validate_inputs(np.array([0.5, 0.3]), np.array([[0, 1], [2, 0]]), symmetrize=False)

def test_edge_asymmetric_j_symmetrized():
    """E07: Asymmetric J (symmetrize=True) gets canonical symmetrization"""
    J_input = np.array([[0, 3], [1, 0]])  # Not symmetric
    J_expected = np.array([[0, 2], [2, 0]])  # (J + J.T)/2
    h = np.array([0.5, 0.3])
    
    J_validated, h_validated = validate_inputs(h, J_input, symmetrize=True)
    
    np.testing.assert_array_equal(J_validated, J_expected)
    np.testing.assert_array_equal(h_validated, h)

def test_edge_nonzero_diagonal_j():
    """E08: Non-zero diagonal in J raises ValueError"""
    with pytest.raises(ValueError, match="Coupling matrix J must have zero diagonal elements."):
        validate_inputs(np.array([0.5, 0.3]), np.array([[1, 0], [0, 1]]))

def test_edge_nonpositive_sweeps():
    """E09: Non-positive sweeps raises ValueError"""
    with pytest.raises(ValueError, match="num_sweeps must be a positive integer."):
        validate_inputs(np.array([0.5, 0.3]), np.array([[0, 1], [1, 0]]), num_sweeps=0)
    
    with pytest.raises(ValueError, match="num_sweeps must be a positive integer."):
        validate_inputs(np.array([0.5, 0.3]), np.array([[0, 1], [1, 0]]), num_sweeps=-5)

def test_edge_nonpositive_reads():
    """E10: Non-positive reads raises ValueError"""
    with pytest.raises(ValueError, match="num_reads must be a positive integer."):
        validate_inputs(np.array([0.5, 0.3]), np.array([[0, 1], [1, 0]]), num_reads=0)
    
    with pytest.raises(ValueError, match="num_reads must be a positive integer."):
        validate_inputs(np.array([0.5, 0.3]), np.array([[0, 1], [1, 0]]), num_reads=-3)

def test_edge_disconnected_spins():
    """E11: Disconnected spins handled gracefully"""
    # Spin 0 is disconnected (no couplings)
    h = np.array([0.5, -0.3])
    J = np.array([[0, 0], [0, 0]])  # No couplings
    
    J_validated, h_validated = validate_inputs(h, J)
    
    np.testing.assert_array_equal(J_validated, J)
    np.testing.assert_array_equal(h_validated, h)

def test_edge_all_zero_model():
    """E12: All-zero model handled gracefully"""
    h = np.array([0.0, 0.0])
    J = np.array([[0, 0], [0, 0]])
    
    J_validated, h_validated = validate_inputs(h, J)
    
    np.testing.assert_array_equal(J_validated, J)
    np.testing.assert_array_equal(h_validated, h)
    # Should return valid sample (not raise exception)

def test_edge_exp_overflow_guard():
    """E13: exp() numerical overflow guard - tested implicitly through validation"""
    # This is more about the algorithm implementation, but we can verify
    # that large (but within bounds) values don't cause issues in validation itself
    # Using values near but under the 1e12 threshold from E14
    large_h = np.array([1e11, 1e11])
    large_J = np.array([[0, 1e11], [1e11, 0]])
    
    # Should not raise during validation (overflow handling is in algorithm)
    J_validated, h_validated = validate_inputs(large_h, large_J)
    np.testing.assert_array_equal(J_validated, large_J)
    np.testing.assert_array_equal(h_validated, large_h)

def test_edge_coefficient_overflow():
    """E14: Coefficient bounds exceeded raises OverflowError"""
    with pytest.raises(OverflowError, match="Coefficient magnitude exceeds 1e12."):
        validate_inputs(np.array([0.0, 0.0]), np.array([[0, 1e13], [1e13, 0]]))

def test_edge_nonpositive_temperatures():
    """E16: Non-positive inverse temperatures raise ValueError"""
    from pqblocks_anneal_sa.solver import simulated_annealing
    
    h = np.array([0.5, -0.3])
    J = np.array([[0, 1], [1, 0]])
    
    with pytest.raises(ValueError, match="Inverse temperatures beta_0 and beta_f must be positive floats."):
        simulated_annealing(J, h, num_sweeps=100, num_reads=10, beta_0=-1.0, beta_f=2.0)
    
    with pytest.raises(ValueError, match="Inverse temperatures beta_0 and beta_f must be positive floats."):
        simulated_annealing(J, h, num_sweeps=100, num_reads=10, beta_0=1.0, beta_f=-0.5)
    
    with pytest.raises(ValueError, match="Inverse temperatures beta_0 and beta_f must be positive floats."):
        simulated_annealing(J, h, num_sweeps=100, num_reads=10, beta_0=0.0, beta_f=0.0)

    with pytest.raises(ValueError, match="Inverse temperatures beta_0 and beta_f must be positive floats."):
        simulated_annealing(J, h, num_sweeps=100, num_reads=10, beta_0=float("nan"), beta_f=2.0)

    with pytest.raises(ValueError, match="Inverse temperatures beta_0 and beta_f must be positive floats."):
        simulated_annealing(J, h, num_sweeps=100, num_reads=10, beta_0=1.0, beta_f=float("inf"))

    with pytest.raises(ValueError, match="Inverse temperatures beta_0 and beta_f must be positive floats."):
        simulated_annealing(J, h, num_sweeps=100, num_reads=10, beta_0=complex(1.0, 0.0), beta_f=2.0)

def test_bqm_adapter():
    """Test BQM adapter raises NotImplementedError for unimplemented methods"""
    adapter = BQMAdapter()
    assert adapter is not None
    
    with pytest.raises(NotImplementedError, match="BQMAdapter.to_ising not implemented in this version"):
        adapter.to_ising(None)
    
    with pytest.raises(NotImplementedError, match="BQMAdapter.to_qubo not implemented in this version"):
        adapter.to_qubo(None)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])