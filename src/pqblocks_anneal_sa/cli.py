#!/usr/bin/env python3
"""
Command-line interface for the Simulated Annealing solver.

This module provides a CLI for solving Ising models using simulated annealing
with automatic parameter derivation when not explicitly provided.
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path

# Add the project root to pathlib for proper imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from src package using the same style as tests
try:
    from pqblocks_anneal_sa.solver import simulated_annealing
except ImportError:
    from .solver import simulated_annealing


def parse_matrix_string(matrix_str):
    """Parse a string representation of a matrix into a numpy array.
    
    Expected format: "[[0, 1, -1], [1, 0, 1], [-1, 1, 0]]" or similar.
    """
    try:
        # Remove outer brackets and split by rows
        matrix_str = matrix_str.strip()
        if matrix_str.startswith('[') and matrix_str.endswith(']'):
            matrix_str = matrix_str[1:-1]
        
        # Split by '], [' to get rows
        rows = matrix_str.split('], [')
        # Clean up first and last elements
        rows[0] = rows[0].lstrip('[')
        rows[-1] = rows[-1].rstrip(']')
        
        # Parse each row
        matrix = []
        for row in rows:
            # Split by comma and convert to float
            elements = [float(x.strip()) for x in row.split(',')]
            matrix.append(elements)
        
        return np.array(matrix, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Invalid matrix format: {matrix_str}. Error: {str(e)}")


def parse_vector_string(vector_str):
    """Parse a string representation of a vector into a numpy array.
    
    Expected format: "[0.5, -0.3, 0.2, -0.1]" or similar.
    """
    try:
        # Remove outer brackets
        vector_str = vector_str.strip()
        if vector_str.startswith('[') and vector_str.endswith(']'):
            vector_str = vector_str[1:-1]
        
        # Split by comma and convert to float
        elements = [float(x.strip()) for x in vector_str.split(',')]
        return np.array(elements, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Invalid vector format: {vector_str}. Error: {str(e)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simulated Annealing solver for Ising models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve a 4-spin system from the worked example
  %(prog)s --j-matrix "[[0, 1, -1, 0], [1, 0, 1, -1], [-1, 1, 0, 1], [0, -1, 1, 0]]" \\
           --h-vector "[0.5, -0.3, 0.2, -0.1]" \\
           --sweeps 1000 --reads 10 --seed 42 --output json
       
  # Solve with default parameters
  %(prog)s --j-matrix "[[0, 1], [1, 0]]" --h-vector "[0.5, -0.3]"
       
  # Get human-readable summary
  %(prog)s --j-matrix "[[0, 1], [1, 0]]" --h-vector "[0.5, -0.3]" --output summary
        """
    )
    
    parser.add_argument(
        '--j-matrix',
        type=str,
        required=True,
        help='Coupling matrix J as a string, e.g., "[[0, 1], [1, 0]]"'
    )
    
    parser.add_argument(
        '--h-vector',
        type=str,
        required=True,
        help='Linear biases h as a string, e.g., "[0.5, -0.3]"'
    )
    
    parser.add_argument(
        '--sweeps',
        type=int,
        help='Number of sweeps per read (default: max(1000, 10*N))'
    )
    
    parser.add_argument(
        '--reads',
        type=int,
        default=10,
        help='Number of independent reads (default: 10)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--output',
        choices=['json', 'summary'],
        default='json',
        help='Output format: json for machine-readable, summary for human-readable (default: json)'
    )
    
    args = parser.parse_args()
    
    try:
        # Parse inputs
        J = parse_matrix_string(args.j_matrix)
        h = parse_vector_string(args.h_vector)
        
        # Validate basic constraints
        if J.shape[0] != J.shape[1]:
            raise ValueError("Coupling matrix J must be square")
        
        if len(h) != J.shape[0]:
            raise ValueError(f"Linear biases length ({len(h)}) must match matrix dimension ({J.shape[0]})")
        
        # Run the solver
        result = simulated_annealing(
            J=J,
            h=h,
            num_sweeps=args.sweeps,
            num_reads=args.reads,
            seed=args.seed
        )
        
        # Produce output
        if args.output == 'json':
            # Convert numpy arrays to lists for JSON serialization
            output = {
                'samples': result['samples'].tolist(),
                'energies': result['energies'].tolist(),
                'num_occurrences': result['num_occurrences'].tolist(),
                'info': result['info']
            }
            print(json.dumps(output, indent=2))
        else:  # summary
            best_energy_idx = np.argmin(result['energies'])
            best_energy = result['energies'][best_energy_idx]
            best_sample = result['samples'][best_energy_idx]
            
            print(f"Best energy found: {best_energy:.6f}")
            print(f"Corresponding sample: {best_sample}")
            print(f"Number of reads: {len(result['energies'])}")
            print(f"Beta_0: {result['info']['beta_0']:.6f}")
            print(f"Beta_f: {result['info']['beta_f']:.6f}")
            print(f"Cooling schedule: {result['info']['cooling_schedule']}")
            
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()