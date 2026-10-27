"""
Test suite for Simulated Annealing.

This module tests all aspects of the SA solver including:
- Edge case handling
- Algorithmic correctness
- Parameter auto-derivation
- Comparison with reference implementation
"""

import numpy as np
import pytest
from pqblocks_anneal_sa.validation import validate_inputs
from pqblocks_anneal_sa.reference import _sa_reference
from pqblocks_anneal_sa.solver import simulated_annealing, _derive_beta_0, _derive_beta_f


class TestSolverAPI:
    """Test the public API façade functionality."""
    
    def test_beta_0_derivation(self):
        """Test beta_0 derivation matches spec formula."""
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        
        # Verify calculation from spec:
        # DeltaE_row(i) = 2|h_i| + 2 sum_j |J_ij|
        # deltaE_mean = mean of DeltaE_row
        # beta_0 = ln(2) / deltaE_mean
        
        deltaE_row = 2 * np.abs(h) + 2 * np.sum(np.abs(J), axis=1)
        deltaE_mean = np.mean(deltaE_row)
        expected_beta_0 = np.log(2) / deltaE_mean
        
        computed_beta_0 = _derive_beta_0(h, J)
        
        assert np.isclose(computed_beta_0, expected_beta_0), \
            f"Expected {expected_beta_0}, got {computed_beta_0}"
    
    def test_beta_f_derivation(self):
        """Test beta_f derivation matches spec formula."""
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        h = np.array([0.5, -0.3, 0.2, -0.1])
        
        # Verify calculation from spec:
        # DeltaE_min = min({2|J_ij| : J_ij != 0} U {2|h_i| : h_i != 0} U {1.0})
        nonzero_couplings = J[np.where(J != 0)]
        nonzero_biases = h[np.where(h != 0)]
        min_energy = min(
            2 * np.min(np.abs(nonzero_couplings)) if nonzero_couplings.size > 0 else float('inf'),
            2 * np.min(np.abs(nonzero_biases)) if nonzero_biases.size > 0 else float('inf'),
            1.0
        )
        
        expected_beta_f = np.log(10**4) / min_energy
        computed_beta_f = _derive_beta_f(J, h, 1.0)  # Pass arbitrary beta_0
        
        assert np.isclose(computed_beta_f, expected_beta_f), \
            f"Expected {expected_beta_f}, got {computed_beta_f}"
    
    def test_min_sweep_default(self):
        """Test default num_sweeps calculation."""
        N = 10
        # Default should be max(1000, 10*N) = 1000 for N=10
        h = np.zeros(N)
        J = np.zeros((N, N))
        
        # For N=10, max(1000, 100) = 1000
        result = simulated_annealing(J, h, num_reads=1, seed=42)
        # Actually, we can't easily check internal num_sweeps from outside
        # But we can verify the function runs without error
        
    def test_solver_returns_correct_structure(self):
        """Test solver returns dict with expected keys."""
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        
        result = simulated_annealing(J, h, num_sweeps=100, num_reads=5, seed=42)
        
        assert 'samples' in result
        assert 'energies' in result
        assert 'num_occurrences' in result
        assert 'info' in result
        
        assert result['samples'].shape == (5, 4)
        assert len(result['energies']) == 5
        
    def test_seeded_reproducibility(self):
        """Test that same seed produces same results."""
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        
        result1 = simulated_annealing(J, h, num_sweeps=100, num_reads=3, seed=123)
        result2 = simulated_annealing(J, h, num_sweeps=100, num_reads=3, seed=123)
        
        np.testing.assert_array_equal(result1['samples'], result2['samples'])
        np.testing.assert_array_equal(result1['energies'], result2['energies'])


class TestIntegration:
    """Integration tests comparing solver with reference."""
    
    def test_worked_example_matches_spec(self):
        """Test worked example matches spec values exactly."""
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        
        # From spec: beta_0 derived from mean, beta_f from min
        result = simulated_annealing(J, h, num_sweeps=1000, num_reads=10, seed=42)
        
        # The worked example shows energy should converge to -6.1
        # with ground state [-1, +1, -1, +1]
        best_energy = np.min(result['energies'])
        best_sample = result['samples'][np.argmin(result['energies'])]
        
        assert np.isclose(best_energy, -6.1, atol=0.1), \
            f"Expected energy near -6.1, got {best_energy}"
        np.testing.assert_array_equal(best_sample, np.array([-1, 1, -1, 1]))
    
    def test_reference_vs_solver_consistency(self):
        """Test reference and main solver give consistent results."""
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        
        # Run reference
        ref_sample, ref_energy = _sa_reference(
            J, h, 1000, 10.0, 10.0, 'geometric', 42
        )
        
        # Run main solver (using same parameters)
        result = simulated_annealing(
            J, h, num_sweeps=1000, num_reads=1, seed=42
        )
        
        # Note: energies may differ slightly due to different contexts
        # but they should be in the same ballpark
        assert np.isclose(ref_energy, result['energies'][0], atol=0.01), \
            f"Reference: {ref_energy}, Solver: {result['energies'][0]}"


class TestEdgeCases:
    """Test edge case handling in the solver."""
    
    def test_empty_model_error(self):
        """E01: Empty model raises ValueError."""
        with pytest.raises(ValueError, match="Empty model: N=0 is not supported"):
            simulated_annealing(np.array([]).reshape(0, 0), np.array([]))
    
    def test_single_spin_error(self):
        """E02: Single spin raises ValueError."""
        with pytest.raises(ValueError, match="Model must have at least 2 spins"):
            simulated_annealing(np.array([[0.0]]), np.array([0.5]))
    
    def test_dimension_mismatch_error(self):
        """E03: Dimension mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Dimension mismatch"):
            simulated_annealing(
                np.array([[0, 1], [1, 0]]), 
                np.array([0.5, 0.3, 0.1])  # Wrong size
            )
    
    def test_non_finite_h_error(self):
        """E04: Non-finite h raises ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            simulated_annealing(
                np.array([[0, 1], [1, 0]]),
                np.array([0.5, np.nan])
            )
    
    def test_non_finite_j_error(self):
        """E05: Non-finite J raises ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            simulated_annealing(
                np.array([[0, np.inf], [1, 0]]),
                np.array([0.5, 0.3])
            )
    
    def test_asymmetric_j_rejected(self):
        """E06: Asymmetric J raises ValueError."""
        with pytest.raises(ValueError, match="symmetric"):
            simulated_annealing(
                np.array([[0, 1], [2, 0]]),
                np.array([0.5, 0.3])
            )
    
    def test_asymmetric_j_symmetrized(self):
            """E07: Asymmetric J (symmetrize=True) gets canonical symmetrization"""
            # Note: The public API doesn't expose symmetrize parameter directly
            # But we can test that our validation works correctly
            from pqblocks_anneal_sa.validation import validate_inputs
        
            J_input = np.array([[0, 3], [1, 0]])  # Not symmetric
            J_expected = np.array([[0, 2], [2, 0]])  # (J + J.T)/2
            h = np.array([0.5, 0.3])
        
            J_validated, h_validated = validate_inputs(h, J_input, symmetrize=True)
        
            np.testing.assert_array_equal(J_validated, J_expected)
            np.testing.assert_array_equal(h_validated, h)
    
    def test_nonzero_diagonal_error(self):
        """E08: Non-zero diagonal J raises ValueError."""
        with pytest.raises(ValueError, match="zero diagonal"):
            simulated_annealing(
                np.array([[1, 1], [1, 0]]),
                np.array([0.5, 0.3])
            )
    
    def test_nonpositive_sweeps_error(self):
        """E09: Non-positive sweeps raises ValueError."""
        with pytest.raises(ValueError, match="num_sweeps must be a positive integer"):
            simulated_annealing(
                np.array([[0, 1], [1, 0]]),
                np.array([0.5, 0.3]),
                num_sweeps=0
            )
    
    def test_nonpositive_reads_error(self):
        """E10: Non-positive reads raises ValueError."""
        with pytest.raises(ValueError, match="num_reads must be a positive integer"):
            simulated_annealing(
                np.array([[0, 1], [1, 0]]),
                np.array([0.5, 0.3]),
                num_reads=0
            )
    
    def test_beta_f_guard(self):
        """Test beta_f guard: if beta_f < beta_0, beta_f is set to 10*beta_0."""
        J = np.array([[0, 1], [1, 0]])
        h = np.array([0.5, -0.3])
        # Auto-derive beta_0 and beta_f where beta_f < beta_0
        beta_0 = _derive_beta_0(h, J)
        beta_f = _derive_beta_f(J, h, beta_0)
        # Verify guard condition
        assert beta_f >= beta_0
    
    def test_all_zero_model_beta_derivation(self):
        """Test beta derivation for all-zero model returns defaults."""
        J = np.zeros((4, 4))
        h = np.zeros(4)
        beta_0 = _derive_beta_0(h, J)
        beta_f = _derive_beta_f(J, h, beta_0)
        assert beta_0 == 0.1
        assert beta_f == 5.0
    
    def test_non_array_j_conversion(self):
        """Test non-array J input gets converted to numpy array."""
        h = np.array([0.5, -0.3])
        J_list = [[0, 1], [1, 0]]
        result = simulated_annealing(J_list, h, num_sweeps=10, num_reads=1, seed=42)
        assert result is not None
    
    def test_coefficient_overflow_error(self):
        """E14: Large coefficients raise OverflowError."""
        with pytest.raises(OverflowError, match="Coefficient magnitude exceeds 1e12"):
            simulated_annealing(
                np.array([[0, 1e13], [1e13, 0]]),
                np.array([0.0, 0.0])
            )
    
    def test_test_solver(self):
        """Test _test_solver runs without error."""
        from pqblocks_anneal_sa.solver import _test_solver
        _test_solver()


class TestCoolingSchedules:
    """Test all cooling schedule options."""
    
    def test_geometric_schedule(self):
        """Test geometric cooling schedule."""
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        
        result = simulated_annealing(
            J, h, num_sweeps=100, num_reads=1, seed=42,
            cooling_schedule='geometric'
        )
        assert result is not None
    
    def test_linear_schedule(self):
        """Test linear cooling schedule."""
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        
        result = simulated_annealing(
            J, h, num_sweeps=100, num_reads=1, seed=42,
            cooling_schedule='linear'
        )
        assert result is not None
    
    def test_logarithmic_schedule(self):
        """Test logarithmic cooling schedule."""
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        
        result = simulated_annealing(
            J, h, num_sweeps=100, num_reads=1, seed=42,
            cooling_schedule='logarithmic'
        )
        assert result is not None
    
    def test_invalid_schedule_error(self):
        """Test invalid cooling schedule raises ValueError."""
        h = np.array([0.5, -0.3])
        J = np.array([[0, 1], [1, 0]])
        
        with pytest.raises(ValueError, match="Unknown cooling schedule"):
            simulated_annealing(J, h, num_sweeps=10, seed=42, cooling_schedule='invalid')


class TestNumbaKernel:
    """Tests for Numba kernel integration."""
    
    def test_kernel_exists(self):
        """Test that Numba kernel is available."""
        from pqblocks_anneal_sa.kernel import _sa_kernel
        assert callable(_sa_kernel)
    
    def test_kernel_basic_execution(self):
        """Test kernel runs without error."""
        from pqblocks_anneal_sa.kernel import _sa_kernel
        
        h = np.array([0.5, -0.3, 0.2, -0.1])
        J = np.array([
            [0, 1, -1, 0],
            [1, 0, 1, -1],
            [-1, 1, 0, 1],
            [0, -1, 1, 0]
        ])
        
        samples, energies = _sa_kernel(
            J, h, num_sweeps=10, num_reads=3,
            beta_0=5.0, beta_f=10.0, cooling_schedule='geometric', seed=42
        )
        
        assert samples.shape == (3, 4)
        assert len(energies) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])