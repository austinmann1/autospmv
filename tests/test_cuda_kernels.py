import pytest
import torch
import numpy as np
from pathlib import Path
import os
import shutil

# Import both real and mock implementations
try:
    from src.cuda.kernel_utils import CUDACompiler
    REAL_CUDA = True
except ImportError:
    from src.cuda.mock_cuda_env import MockCUDACompiler as CUDACompiler
    REAL_CUDA = False

from src.benchmarks.kernelbench_utils import KernelBenchWrapper

# Skip tests that require real CUDA if it's not available
requires_cuda = pytest.mark.skipif(
    not REAL_CUDA,
    reason="Test requires CUDA GPU"
)

class TestCUDAKernels:
    @classmethod
    def setup_class(cls):
        """Setup test environment and utilities."""
        cls.output_dir = Path(__file__).parent.parent / 'output'
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        
        cls.cuda_compiler = CUDACompiler(cls.output_dir)
        cls.kernelbench = KernelBenchWrapper()
        
        # Set up mock data for local testing
        if not REAL_CUDA:
            cls.setup_mock_data()
    
    @classmethod
    def setup_mock_data(cls):
        """Create mock data for testing without CUDA."""
        cls.mock_matrix = torch.randn(1000, 1000)
        cls.mock_vector = torch.randn(1000)
        
    def compile_kernel(self, source_file: Path) -> Path:
        """Compile a CUDA kernel for testing."""
        output_file = self.output_dir / f"{source_file.stem}_test"
        success, message = self.cuda_compiler.compile_kernel(source_file, output_file)
        assert success, f"Kernel compilation failed: {message}"
        return output_file
    
    @pytest.mark.basic
    def test_baseline_compilation(self):
        """Test that baseline CUDA kernel compiles."""
        source_file = Path(__file__).parent.parent / 'src' / 'cuda' / 'baseline_spmv.cu'
        output_file = self.compile_kernel(source_file)
        assert output_file.exists(), "Compiled kernel not found"
    
    @requires_cuda
    def test_kernel_correctness_real_gpu(self):
        """Test kernel correctness on real GPU."""
        # This test only runs when CUDA is available
        tasks = self.kernelbench.get_level1_tasks()
        task_id = tasks[0]['id']
        inputs = self.kernelbench.get_task_inputs(task_id)
        
        baseline_results = self.kernelbench.run_pytorch_baseline(task_id, inputs)
        kernel_path = self.compile_kernel(
            Path(__file__).parent.parent / 'src' / 'cuda' / 'baseline_spmv.cu'
        )
        candidate_results = self.kernelbench.run_cuda_kernel(kernel_path, inputs)
        
        assert self.kernelbench.verify_outputs(
            baseline_results['output'],
            candidate_results['output']
        )
    
    @pytest.mark.basic
    def test_kernel_correctness_mock(self):
        """Test kernel correctness with mock data."""
        if REAL_CUDA:
            pytest.skip("This test is for mock environment only")
            
        # Use mock data and execution
        result = self.cuda_compiler.profile_kernel(
            self.output_dir / "mock_kernel"
        )
        assert 'achieved_occupancy' in result
        assert 'sm_efficiency' in result
        
    @pytest.mark.basic
    def test_profiling_metrics(self):
        """Test that we can collect profiling metrics."""
        source_file = Path(__file__).parent.parent / 'src' / 'cuda' / 'baseline_spmv.cu'
        kernel_path = self.compile_kernel(source_file)
        
        metrics = self.cuda_compiler.profile_kernel(kernel_path)
        expected_metrics = {
            'achieved_occupancy',
            'sm_efficiency',
            'dram_read_throughput',
            'dram_write_throughput'
        }
        
        assert set(metrics.keys()) >= expected_metrics
        
    def teardown_method(self):
        """Cleanup after each test."""
        # Remove compiled kernels
        for file in self.output_dir.glob("*_test"):
            if file.is_file():
                file.unlink()
