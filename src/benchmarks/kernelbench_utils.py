import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, List
import json
import os
import time
from kernelbench import Model, get_inputs, get_init_inputs

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
            # Load KernelBench models
            self.models = {}
            for level in range(1, 4):  # Levels 1-3
                try:
                    self.models[f'level{level}'] = Model(level)
                except Exception as e:
                    print(f"Warning: Could not load level {level}: {e}")
        else:
            self._setup_mock_dataset()
            
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
        
    def get_level1_tasks(self) -> List[Dict[str, Any]]:
        """Get list of Level-1 tasks."""
        if self.use_mock:
            return [task for task in self.mock_tasks if task['level'] == 1]
        else:
            return [task for task in self.models['level1'].tasks if task['level'] == 1]
    
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
        model = self.models[f'level{level}']
        
        # Get inputs
        inputs = get_inputs(model)
        init_inputs = get_init_inputs(model)
        
        return {
            **inputs,
            **init_inputs
        }
                   
    def get_task_inputs(self, task_id: str) -> Tuple[torch.Tensor, ...]:
        """Generate inputs for a specific task."""
        if self.use_mock:
            task = next(t for t in self.mock_tasks if t['id'] == task_id)
        else:
            task = next(t for t in self.models[f'level{int(task_id.split("_level")[-1])}'].tasks if t['id'] == task_id)
            
        input_gen = eval(task['get_inputs'])  # Safe eval as this is from trusted source
        return input_gen()
    
    def run_pytorch_baseline(self, task_id: str, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
        """Run PyTorch eager implementation and collect performance metrics."""
        if self.use_mock:
            task = next(t for t in self.mock_tasks if t['id'] == task_id)
        else:
            task = next(t for t in self.models[f'level{int(task_id.split("_level")[-1])}'].tasks if t['id'] == task_id)
            
        # Create a namespace for the function
        namespace = {'torch': torch}
        exec(task['reference_impl'], namespace)
        pytorch_impl = namespace['ref_impl']
        
        # Warmup
        for _ in range(5):
            _ = pytorch_impl(*inputs)
        
        # Benchmark
        if self.has_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            
            starter.record()
            output = pytorch_impl(*inputs)
            ender.record()
            
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender)
        else:
            # Use time.perf_counter for CPU timing
            start_time = time.perf_counter()
            output = pytorch_impl(*inputs)
            end_time = time.perf_counter()
            elapsed_time = (end_time - start_time) * 1000  # Convert to ms
            
            # For mock environment, add some randomness
            if self.use_mock:
                elapsed_time = elapsed_time * np.random.uniform(0.8, 1.2)
        
        return {
            'output': output,
            'runtime_ms': elapsed_time,
            'implementation': 'pytorch_eager'
        }
    
    def run_cuda_kernel(self, task_id: str, kernel_binary: Path, 
                       collect_metrics: bool = True) -> Dict[str, Any]:
        """Run a CUDA kernel and collect performance metrics.
        
        Args:
            task_id: Task ID to run
            kernel_binary: Path to compiled CUDA kernel
            collect_metrics: Whether to collect detailed GPU metrics
            
        Returns:
            Dict containing performance metrics
        """
        if self.use_mock:
            # Simulate kernel execution with mock metrics
            runtime = np.random.uniform(0.5, 2.0)
            metrics = {
                'runtime_ms': runtime,
                'achieved_occupancy': np.random.uniform(0.6, 0.9),
                'sm_efficiency': np.random.uniform(70, 95),
                'dram_read_throughput': np.random.uniform(200, 400),
                'dram_write_throughput': np.random.uniform(100, 300)
            }
            
            # Add some randomness to metrics based on kernel binary name
            iteration = int(kernel_binary.stem.split('_')[-1]) if '_' in kernel_binary.stem else 0
            improvement = min(0.2, iteration * 0.05)  # Max 20% improvement
            metrics['runtime_ms'] *= (1.0 - improvement)
            
            return metrics
        else:
            # TODO: Implement real CUDA kernel execution and profiling
            raise NotImplementedError("Real CUDA execution not implemented yet")
    
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
            
        # Get reference output
        level = int(task_id.split('_level')[-1])
        model = self.models[f'level{level}']
        
        with torch.no_grad():
            expected = model(**inputs)
            
        # Compare with tolerance
        return torch.allclose(output, expected, rtol=1e-5, atol=1e-5)
        
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
    
    def generate_perf_report(self, task_id: str, baseline_metrics: Dict[str, Any],
                            candidate_metrics: Dict[str, Any]) -> str:
        """Generate a performance comparison report."""
        speedup = baseline_metrics['runtime_ms'] / candidate_metrics['runtime_ms']
        
        report = {
            'task_id': task_id,
            'baseline_runtime_ms': baseline_metrics['runtime_ms'],
            'candidate_runtime_ms': candidate_metrics['runtime_ms'],
            'speedup': speedup,
            'numerically_correct': self.verify_output(
                task_id,
                candidate_metrics['output'].cpu(),
                self.get_task_data(task_id)
            )
        }
        
        return json.dumps(report, indent=2)
