import numpy as np
from typing import Optional, Tuple

def _sa_reference(
    J: np.ndarray,
    h: np.ndarray,
    num_sweeps: int,
    beta_0: float,
    beta_f: float,
    cooling_schedule: str,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """
    Simulated Annealing solver for Ising model.
    
    Parameters
    ----------
    J : np.ndarray
        Coupling matrix, shape (N, N), symmetric, zero diagonal
    h : np.ndarray
        Linear biases, shape (N,)
    num_sweeps : int
        Number of sweeps per run
    beta_0 : float
        Initial inverse temperature
    beta_f : float
        Final inverse temperature
    cooling_schedule : str
        Cooling schedule: 'geometric', 'linear', or 'logarithmic'
    seed : Optional[int]
        Random seed for reproducibility
    
    Returns
    -------
    best_s : np.ndarray
        Best spin configuration found, shape (N,), dtype=int8
    best_energy : float
        Energy of the best configuration
    """
    # Initialize random number generator
    rng = np.random.Generator(np.random.PCG64(seed))
    N = len(h)
    
    # Random initial configuration s in {-1, +1}^N
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    
    # Compute initial local field cache: h_eff = J @ s + h
    # We use float64 for energy accumulators
    h_eff = np.dot(J, s.astype(np.float64)) + h
    
    # Compute initial total energy in f64
    current_energy = 0.5 * np.dot(s, np.dot(J, s)) + np.dot(h, s)
    best_s = s.copy()
    best_energy = current_energy
    
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
            raise ValueError(f"Unknown cooling schedule: {cooling_schedule}")
        
        # 2. Random spin permutation for sweep order
        permutation = rng.permutation(N)
        
        # 3. Sweep over spins
        for i in permutation:
            delta_E = -2.0 * s[i] * h_eff[i]
            
            # 4. Metropolis acceptance with overflow-safe clamping
            if delta_E <= 0.0:
                accept = True
            else:
                arg = -beta * delta_E
                if arg < -709.0:  # exp(-709) is about 1e-308, underflow to 0.0
                    accept = False
                else:
                    accept = (rng.random() < np.exp(arg))
            
            # 5. Incremental state & cache update
            if accept:
                s_old_i = s[i]
                s[i] = -s_old_i
                current_energy += delta_E
                
                # Update neighbor local fields: h_eff -= 2 * J[:, i] * s_old_i
                h_eff -= 2.0 * J[:, i] * s_old_i
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_s = s.copy()
    
    return best_s, best_energy