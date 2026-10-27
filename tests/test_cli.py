"""
Tests for the CLI entry point (Slice 5: Public API Facade & Auto-Parameter Engine & CLI).

These tests verify the CLI functionality including:
- Argument parsing
- Edge case validation from the spec's edge-case ledger (E01-E16)
- Output format handling
"""

import subprocess
import json
import pytest
import os

def run_cli(args, expect_success=True):
    """Run the CLI with given arguments and return output."""
    import shutil, sys
    cli_bin = shutil.which('pqblocks-anneal-sa')
    if cli_bin:
        cmd = [cli_bin] + args
    else:
        cmd = [sys.executable, '-m', 'pqblocks_anneal_sa.cli'] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if expect_success and result.returncode != 0:
        raise AssertionError(f"CLI failed with args {args}: {result.stderr}")
    
    return result.stdout, result.stderr, result.returncode


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""
    
    def test_help_flag(self):
        """Test that --help works."""
        stdout, stderr, code = run_cli(['--help'], expect_success=True)
        assert 'Simulated Annealing solver' in stdout
        assert '--j-matrix' in stdout
        assert '--h-vector' in stdout


class TestCLIValidInputs:
    """Test CLI with valid inputs."""
    
    def test_basic_run_with_summary(self):
        """Test basic run with summary output."""
        stdout, stderr, code = run_cli([
            '--j-matrix', '[[0, 1], [1, 0]]',
            '--h-vector', '[0.5, -0.3]',
            '--sweeps', '100',
            '--reads', '5',
            '--seed', '42',
            '--output', 'summary'
        ])
        assert 'Best energy found' in stdout
        assert 'Beta_0' in stdout
        assert 'Beta_f' in stdout
    
    def test_basic_run_with_json(self):
        """Test basic run with JSON output."""
        stdout, stderr, code = run_cli([
            '--j-matrix', '[[0, 1, -1, 0], [1, 0, 1, -1], [-1, 1, 0, 1], [0, -1, 1, 0]]',
            '--h-vector', '[0.5, -0.3, 0.2, -0.1]',
            '--sweeps', '1000',
            '--reads', '10',
            '--seed', '42',
            '--output', 'json'
        ])
        data = json.loads(stdout)
        assert 'samples' in data
        assert 'energies' in data
        assert 'info' in data
        assert len(data['samples']) == 10
        assert len(data['energies']) == 10


class TestCLIEdgeCases:
    """Test CLI edge cases from spec's edge-case ledger."""
    
    def test_missing_j_matrix(self):
        """E01: Missing --j-matrix should raise error."""
        stdout, stderr, code = run_cli(['--h-vector', '[0.5]'], expect_success=False)
        assert code != 0
    
    def test_missing_h_vector(self):
        """Missing --h-vector should raise error."""
        stdout, stderr, code = run_cli(['--j-matrix', '[[0, 1], [1, 0]]'], expect_success=False)
        assert code != 0
    
    def test_invalid_matrix_format(self):
        """Invalid matrix format should raise error."""
        stdout, stderr, code = run_cli([
            '--j-matrix', 'not a matrix',
            '--h-vector', '[0.5, -0.3]'
        ], expect_success=False)
        assert code != 0
    
    def test_dimension_mismatch(self):
        """E03: Dimension mismatch between h and J should raise ValueError."""
        stdout, stderr, code = run_cli([
            '--j-matrix', '[[0, 1], [1, 0]]',
            '--h-vector', '[0.5, -0.3, 0.2]'  # Wrong length
        ], expect_success=False)
        assert code != 0
        assert 'dimension' in stderr.lower() or 'dimension' in stdout.lower()
    
    def test_non_finite_j(self):
        """E04/E05: Non-finite values in J or h should raise ValueError."""
        stdout, stderr, code = run_cli([
            '--j-matrix', '[[0, nan], [1, 0]]',
            '--h-vector', '[0.5, -0.3]'
        ], expect_success=False)
        assert code != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])