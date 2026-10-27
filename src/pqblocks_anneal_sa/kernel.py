import os
from typing import Optional, Tuple

import numpy as np

# SplitMix64 PRNG implementation for reproducible seeding
# Reference: Bellecompe lab, "SplitMix64" PRNG

def _splitmix64_plain(z: int) -> int:
    """Plain Python implementation of SplitMix64 PRNG."""
    z = (z + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)

def _xoroshiro128plus_next_plain(s0: int, s1: int) -> Tuple[int, int, float]:
    """Plain Python implementation of Xoroshiro128+ PRNG returning random float [0,1)."""
    result = (s0 + s1) & 0xFFFFFFFFFFFFFFFF
    
    s0 = (s0 ^ (s0 >> 18)) & 0xFFFFFFFFFFFFFFFF
    s0 = (s0 ^ (s0 << 31)) & 0xFFFFFFFFFFFFFFFF
    s1 = (s1 ^ (s1 >> 45)) & 0xFFFFFFFFFFFFFFFF
    
    s0 = (s0 ^ s1) & 0xFFFFFFFFFFFFFFFF
    s1 = ((s1 ^ (s1 >> 32)) * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
    s0 = ((s0 * 0x2545F4914F6CDD1D)) & 0xFFFFFFFFFFFFFFFF
    
    # Convert to float in [0, 1)
    rand_float = result / 0xFFFFFFFFFFFFFFFF
    
    return s0, s1, rand_float


def _sa_kernel(
    J: np.ndarray,
    h: np.ndarray,
    num_sweeps: int,
    num_reads: int,
    beta_0: float,
    beta_f: float,
    cooling_schedule: str,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Annealing solver for Ising model.
    
    Parameters
    ----------
    J : np.ndarray
        Coupling matrix, shape (N, N), symmetric, zero diagonal, float64
    h : np.ndarray
        Linear biases, shape (N,), float64
    num_sweeps : int
        Number of sweeps per read
    num_reads : int
        Number of independent reads (parallel chains)
    beta_0 : float
        Initial inverse temperature
    beta_f : float
        Final inverse temperature
    cooling_schedule : str
        Cooling schedule: 'geometric', 'linear', or 'logarithmic'
    seed : Optional[int]
        Base seed for reproducibility
        
    Returns
    -------
    best_states : np.ndarray
        Best spin configurations found, shape (num_reads, N), dtype=int8
    best_energies : np.ndarray
        Energies of the best configurations, shape (num_reads,), dtype=float64
    """
    # Check if NUMBA_DISABLE_JIT=1 is set
    disable_jit = os.environ.get('NUMBA_DISABLE_JIT', '') == '1'
    
    if disable_jit:
        # Use plain Python implementations
        splitmix64_func = _splitmix64_plain
        xoroshiro_next_func = _xoroshiro128plus_next_plain
    else:
        # Numba-optimized versions would be here (using @njit)
        # For now, use plain Python as reference
        splitmix64_func = _splitmix64_plain
        xoroshiro_next_func = _xoroshiro128plus_next_plain
    
    N = len(h)
    
    # Initialize output arrays
    best_states = np.empty((num_reads, N), dtype=np.int8)
    best_energies = np.full(num_reads, np.inf, dtype=np.float64)
    
    # Process each read sequentially
    for read_idx in range(num_reads):
        # Generate unique seed for this read using SplitMix64
        if seed is None:
            read_seed = splitmix64_func(read_idx)
        else:
            read_seed = splitmix64_func(seed + read_idx)
        
        # Initialize Xoroshiro128+ PRNG state
        s0 = read_seed & 0xFFFFFFFFFFFFFFFF
        s1 = splitmix64_func(read_seed + 1) & 0xFFFFFFFFFFFFFFFF
        
        # Random initial configuration s in {-1, +1}^N
        s = np.empty(N, dtype=np.int8)
        for i in range(N):
            # Generate random integer in [0, 2) and map to {-1, +1}
            s0, s1, rand_float = xoroshiro_next_func(s0, s1)
            if rand_float < 0.5:
                s[i] = -1
            else:
                s[i] = 1
        
        # Compute initial local field cache: h_eff = J @ s + h
        # We use float64 for energy accumulators
        h_eff = np.zeros(N, dtype=np.float64)
        for i in range(N):
            # Compute J[i,:] @ s
            sum_val = 0.0
            for j in range(N):
                sum_val += J[i, j] * s[j]
            h_eff[i] = sum_val + h[i]
        
        # Compute initial total energy in f64
        current_energy = 0.0
        for i in range(N):
            for j in range(i+1, N):  # Only upper triangle to avoid double-counting
                current_energy += J[i, j] * s[i] * s[j]
        for i in range(N):
            current_energy += h[i] * s[i]
        
        best_s = s.copy()
        best_energy = current_energy
        
        # Main annealing loop
        for t in range(num_sweeps):
            # 1. Compute current beta per schedule
            if cooling_schedule == 'geometric':
                # Kirkpatrick et al. (1983) Eq 2
                if num_sweeps > 1:
                    beta = beta_0 * ((beta_f / beta_0) ** (t / (num_sweeps - 1)))
                else:
                    beta = beta_0
            elif cooling_schedule == 'linear':
                if num_sweeps > 1:
                    beta = beta_0 + (beta_f - beta_0) * (t / (num_sweeps - 1))
                else:
                    beta = beta_0
            elif cooling_schedule == 'logarithmic':
                # Geman & Geman (1984)
                if t == 0:
                    beta = beta_0
                else:
                    beta = beta_0 * np.log(2.0) / np.log(2.0 + t)
            else:
                # This should never happen if inputs are validated
                beta = beta_0
            
            # 2. Random spin permutation for sweep order
            # Create initial permutation [0, 1, 2, ..., N-1]
            indices = list(range(N))
            
            # Fisher-Yates shuffle using our PRNG
            for i in range(N-1, 0, -1):
                # Generate random index in [0, i]
                s0, s1, rand_float = xoroshiro_next_func(s0, s1)
                # Convert to integer index in [0, i]
                j = int(rand_float * (i + 1))
                # Swap indices[i] and indices[j]
                indices[i], indices[j] = indices[j], indices[i]
            
            # 3. Sweep over spins
            for idx in range(N):
                i = indices[idx]
                delta_E = -2.0 * s[i] * h_eff[i]
                
                # 4. Metropolis acceptance with overflow-safe clamping
                accept = False
                if delta_E <= 0.0:
                    accept = True
                else:
                    arg = -beta * delta_E
                    if arg < -709.0:  # exp(-709) is about 1e-308, underflow to 0.0
                        accept = False
                    else:
                        s0, s1, rand_float = xoroshiro_next_func(s0, s1)
                        accept = (rand_float < np.exp(arg))
                
                # 5. Incremental state & cache update
                if accept:
                    s_old_i = s[i]
                    s[i] = -s_old_i
                    current_energy += delta_E
                    
                    # Update neighbor local fields: h_eff -= 2 * J[:, i] * s_old_i
                    for j in range(N):
                        h_eff[j] -= 2.0 * J[j, i] * s_old_i
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_s = s.copy()
        
        # Store results for this read
        best_states[read_idx, :] = best_s
        best_energies[read_idx] = best_energy
    
    return best_states, best_energies