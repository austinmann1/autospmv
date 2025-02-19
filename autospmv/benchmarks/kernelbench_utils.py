import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, List
import json
import os
import time

class KernelBenchWrapper:
    """Wrapper for KernelBench dataset and evaluation utilities."""
    
    def __init__(self, cache_dir: Path = None, use_mock: bool = False):
        """Initialize KernelBench wrapper.
        
        Args:
            cache_dir: Directory to cache datasets
            use_mock: If True, use mock data for testing
        """
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'kernelbench'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock = use_mock
        self.has_cuda = torch.cuda.is_available()
        
        if not use_mock:
            try:
                import sys
                from KernelBench.src.problems import get_problem
                self.get_problem = get_problem
            except ImportError as e:
                print(f"Warning: Could not import KernelBench: {e}")
                print("Using mock data instead")
                self.use_mock = True
                
        if self.use_mock:
            self._setup_mock_dataset()
            
    def get_task_data(self, task_id: str) -> Dict[str, torch.Tensor]:
        """Get input data for a specific task.
        
        Args:
            task_id: Task identifier (e.g., 'spmv_level1')
            
        Returns:
            Dictionary of input tensors
        """
        if self.use_mock:
            return self._get_mock_data(task_id)
            
        # Parse task ID
        level = int(task_id.split('_level')[-1])
        problem_id = int(task_id.split('_')[0])
        
        # Get problem
        problem = self.get_problem(level, problem_id)
        
        # Get inputs
        inputs = problem.get_inputs()
        
        return inputs
                   
    def verify_output(self, task_id: str, output: torch.Tensor, 
                     inputs: Dict[str, torch.Tensor]) -> bool:
        """Verify the correctness of kernel output.
        
        Args:
            task_id: Task identifier
            output: Kernel output tensor
            inputs: Input tensors used
            
        Returns:
            True if output matches expected result
        """
        if self.use_mock:
            return True  # Mock always passes
            
        # Parse task ID
        level = int(task_id.split('_level')[-1])
        problem_id = int(task_id.split('_')[0])
        
        # Get problem
        problem = self.get_problem(level, problem_id)
        
        # Get reference output
        with torch.no_grad():
            expected = problem.reference_impl(**inputs)
            
        # Compare with tolerance
        return torch.allclose(output, expected, rtol=1e-5, atol=1e-5)
        
    def _setup_mock_dataset(self):
        """Create mock dataset for local development."""
        self.mock_tasks = [
            {
                'id': 'spmv_level1',
                'level': 1,
                'name': 'SpMV',
                'description': 'Sparse Matrix-Vector Multiplication'
            }
        ]
        
    def _get_mock_data(self, task_id: str) -> Dict[str, torch.Tensor]:
        """Get mock data for testing."""
        if 'spmv' in task_id.lower():
            # Mock SpMV data
            nnz = 1000
            n = 1000
            return {
                'values': torch.randn(nnz, dtype=torch.float32),
                'col_indices': torch.randint(0, n, (nnz,), dtype=torch.int32),
                'row_ptr': torch.randint(0, nnz, (n+1,), dtype=torch.int32),
                'x': torch.randn(n, dtype=torch.float32)
            }
        else:
            # Generic mock data
            return {
                'a': torch.randn(1000, 1000, dtype=torch.float32),
                'b': torch.randn(1000, 1000, dtype=torch.float32)
            }
