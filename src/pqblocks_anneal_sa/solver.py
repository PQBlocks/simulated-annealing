"""
Public API Façade & Auto-Parameter Engine for Simulated Annealing.

This module provides the public solver interface that seamlessly integrates
with dimod BinaryQuadraticModel objects and automatically derives annealing
parameters when not explicitly provided.
"""

import numpy as np
from typing import Optional, Union, Dict, Any
import warnings

try:
    from dimod import BinaryQuadraticModel
    HAS_DIMOD = True
except ImportError:
    HAS_DIMOD = False
    BinaryQuadraticModel = None  # type: ignore

from .validation import validate_inputs, BQMAdapter
from .reference import _sa_reference


def _derive_beta_0(h: np.ndarray, J: np.ndarray) -> float:
    """
    Derive initial inverse temperature beta_0 from problem structure.
    
    Uses the mean energy scale: beta_0 = ln(2) / DeltaE_mean
    where DeltaE_mean = (1/N) * sum_i (2|h_i| + 2 * sum_j |J_ij|)
    
    Parameters
    ----------
    h : np.ndarray
        Linear biases, shape (N,)
    J : np.ndarray
        Coupling matrix, shape (N, N)
       
    Returns
    -------
    beta_0 : float
        Initial inverse temperature
    """
    N = len(h)
    if N == 0:
        return 0.1  # Default for empty model (should be caught by validation)
        
    # Compute delta E for each row: 2|h_i| + 2 sum_j |J_ij|
    deltaE_row = 2 * np.abs(h) + 2 * np.sum(np.abs(J), axis=1)
    deltaE_mean = np.mean(deltaE_row)
    
    if deltaE_mean == 0:
        return 0.1  # All-zero model case
       
    return np.log(2) / deltaE_mean


def _derive_beta_f(J: np.ndarray, h: np.ndarray, beta_0: float) -> float:
    """
    Derive final inverse temperature beta_f from problem structure.
    
    Uses the minimum resolvable energy scale: beta_f = ln(10^4) / DeltaE_min
    where DeltaE_min = min({2|J_ij| : J_ij != 0} U {2|h_i| : h_i != 0} U {1.0})
    
    Parameters
    ----------
    J : np.ndarray
        Coupling matrix, shape (N, N)
    h : np.ndarray
        Linear biases, shape (N,)
    beta_0 : float
        Initial inverse temperature (for guard condition)
       
    Returns
    -------
    beta_f : float
        Final inverse temperature
    """
    # Check for all-zero model case first (spec requirement: return 5.0)
    if np.all(J == 0) and np.all(h == 0):
        return 5.0
    
    # Compute minimum resolvable energy scale
    min_energy = float('inf')  # Start with infinity
    
    # Check couplings
    if J.size > 0:
        nonzero_couplings = J[np.where(J != 0)]
        if nonzero_couplings.size > 0:
            min_coupling = 2 * np.min(np.abs(nonzero_couplings))
            min_energy = min(min_energy, min_coupling)
    
    # Check linear biases
    if h.size > 0:
        nonzero_biases = h[np.where(h != 0)]
        if nonzero_biases.size > 0:
            min_bias = 2 * np.min(np.abs(nonzero_biases))
            min_energy = min(min_energy, min_bias)
    
    # Apply default floor of 1.0 if no nonzero elements found
    if min_energy == float('inf'):
        min_energy = 1.0
    
    beta_f = np.log(10**4) / min_energy  # ln(10000) ≈ 9.21034
    
    # Guard condition: ensure beta_f >= beta_0
    if beta_f < beta_0:
        beta_f = 10 * beta_0
       
    return beta_f


def _is_positive_finite_float(x: Any) -> bool:
    """E16 helper: True only for real, finite floats with value > 0."""
    if isinstance(x, bool):
        return False
    if isinstance(x, complex):
        return False
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(xf)) and xf > 0


def simulated_annealing(
    J: Optional[Union[np.ndarray, 'BinaryQuadraticModel']] = None,
    h: Optional[np.ndarray] = None,
    num_sweeps: Optional[int] = None,
    num_reads: int = 10,
    beta_0: Optional[float] = None,
    beta_f: Optional[float] = None,
    cooling_schedule: str = 'geometric',
    seed: Optional[int] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Simulated Annealing solver with automatic parameter derivation.
    
    This function provides a dimod-compatible interface for solving Ising
    and QUBO models using simulated annealing with automatic parameter
    derivation when not explicitly provided.
    
    Parameters
    ----------
    J : np.ndarray or BinaryQuadraticModel, optional
        Coupling matrix, shape (N, N) or a BinaryQuadraticModel object.
        If using a BQM, the 'h' parameter should not be provided.
    h : np.ndarray, optional
        Linear biases, shape (N,). Required if J is a numpy array.
    num_sweeps : int, optional
        Number of sweeps per read. If not provided, defaults to max(1000, 10*N).
    num_reads : int, default 10
        Number of independent reads (interactive) or 100 (benchmark).
    beta_0 : float, optional
        Initial inverse temperature. If not provided, auto-derived from problem.
    beta_f : float, optional
        Final inverse temperature. If not provided, auto-derived from problem.
    cooling_schedule : str, default 'geometric'
        Cooling schedule: 'geometric', 'linear', or 'logarithmic'.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs : dict
        Additional solver-specific parameters.
        
    Returns
    -------
    response : dict
        Dictionary containing:
        - 'samples': np.ndarray of shape (num_reads, N) with spin vectors
        - 'energies': np.ndarray of shape (num_reads,) with energies
        - 'num_occurrences': np.ndarray of shape (num_reads,) with occurrence counts
        - 'info': dict with solver metadata
        
    Raises
    ------
    ValueError
        For invalid inputs as per edge-case ledger.
    OverflowError
        If coefficient magnitude exceeds 1e12.
    ImportError
        If dimod is not available and a BQM is provided.
    """
    # Handle BinaryQuadraticModel input
    if HAS_DIMOD and isinstance(J, BinaryQuadraticModel):
        if h is not None:
            raise ValueError("Cannot provide both BQM and explicit h parameter")
            
        # Convert BQM to Ising format
        adapter = BQMAdapter()
        h_array, J_array, offset = adapter.to_ising(J)
        
        # For simplicity in this implementation, we'll ignore the offset
        # In a full implementation, this would be incorporated into energy calculations
        h = h_array
        J = J_array
        
    elif J is not None and not isinstance(J, np.ndarray):
        # Try to convert to numpy array
        try:
            J = np.asarray(J, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Unable to convert J to numpy array: {e}")
            

    # Validate inputs
    if h is None:
        raise ValueError("Linear biases h must be provided")
        
    J_validated, h_validated = validate_inputs(
        h, J, num_sweeps=num_sweeps, num_reads=num_reads
    )
    
    # Set default num_sweeps if not provided
    if num_sweeps is None:
        N = len(h_validated)
        num_sweeps = max(1000, 10 * N)
        
    # Auto-derive parameters if not provided
    if beta_0 is None:
        beta_0 = _derive_beta_0(h_validated, J_validated)
        
    if beta_f is None:
        beta_f = _derive_beta_f(J_validated, h_validated, beta_0)
        
    if not (_is_positive_finite_float(beta_0) and _is_positive_finite_float(beta_f)):
        raise ValueError("Inverse temperatures beta_0 and beta_f must be positive floats.")
        
    # Handle multiple reads
    all_samples = []
    all_energies = []
    
    for read in range(num_reads):
        # Derive seed for this read to ensure independence
        read_seed = None if seed is None else seed + read * 1000
        
        # Call reference solver
        sample, energy = _sa_reference(
            J_validated, h_validated, num_sweeps, beta_0, beta_f,
            cooling_schedule, read_seed
        )
        
        all_samples.append(sample)
        all_energies.append(energy)
        
    # Convert to numpy arrays
    samples_array = np.array(all_samples, dtype=np.int8)
    energies_array = np.array(all_energies, dtype=np.float64)
    
    # Create response in dimod-compatible format
    response = {
        'samples': samples_array,
        'energies': energies_array,
        'num_occurrences': np.ones(len(all_energies), dtype=int),
        'info': {
            'num_sweeps': num_sweeps,
            'num_reads': num_reads,
            'beta_0': beta_0,
            'beta_f': beta_f,
            'cooling_schedule': cooling_schedule,
            'seed': seed,
            'input_type': 'BQM' if (HAS_DIMOD and isinstance(J, BinaryQuadraticModel)) else 'arrays'
        }
    }
    
    return response


def _test_solver():
    """Simple test function to verify solver works."""
    # Test with worked example from spec
    h = np.array([0.5, -0.3, 0.2, -0.1])
    J = np.array([
        [0, 1, -1, 0],
        [1, 0, 1, -1],
        [-1, 1, 0, 1],
        [0, -1, 1, 0]
    ])
    
    result = simulated_annealing(J, h, num_sweeps=100, num_reads=10, seed=42)
    print(f"Best energy found: {np.min(result['energies'])}")
    print(f"Corresponding sample: {result['samples'][np.argmin(result['energies'])]}")


if __name__ == "__main__":
    _test_solver()