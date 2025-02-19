import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import time
import random

class MockCUDACompiler:
    """Mock CUDA compiler for local development without NVIDIA GPU."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def compile_kernel(self, source_file: Path, output_file: Path,
                      arch: str = "sm_80") -> Tuple[bool, str]:
        """Simulate CUDA kernel compilation."""
        try:
            # Read the source file to check basic syntax
            with open(source_file, 'r') as f:
                source = f.read()
                
            # Basic syntax checking
            if not "__global__" in source:
                return False, "Error: No CUDA kernel found (missing __global__)"
                
            # Create a mock binary file
            with open(output_file, 'w') as f:
                f.write("MOCK_CUDA_BINARY")
                
            return True, "Mock compilation successful"
        except Exception as e:
            return False, f"Mock compilation failed: {str(e)}"

    def profile_kernel(self, binary: Path) -> Dict[str, float]:
        """Simulate CUDA kernel profiling with realistic mock metrics."""
        # Generate realistic-looking mock metrics
        return {
            'achieved_occupancy': random.uniform(0.6, 0.9),
            'sm_efficiency': random.uniform(70.0, 95.0),
            'dram_read_throughput': random.uniform(100.0, 300.0),
            'dram_write_throughput': random.uniform(80.0, 250.0)
        }

class MockCUDAKernel:
    """Mock CUDA kernel execution environment."""
    
    def __init__(self, kernel_source: str):
        self.kernel_source = kernel_source
        self.execution_time = 0.0
        
    def run(self, *args, **kwargs) -> Tuple[float, Dict[str, float]]:
        """Simulate kernel execution with realistic timing and metrics."""
        # Simulate execution time based on input size
        input_size = sum(arg.size if hasattr(arg, 'size') else 1 for arg in args)
        self.execution_time = random.uniform(0.1, 2.0) * (input_size / 1000)
        
        # Simulate kernel metrics
        metrics = {
            'threads_launched': input_size,
            'blocks_used': (input_size + 255) // 256,
            'shared_memory_used': random.randint(0, 48) * 1024  # 0-48KB
        }
        
        return self.execution_time, metrics

def is_cuda_available() -> bool:
    """Check if CUDA is available (always returns False in mock environment)."""
    return False

def get_mock_device_properties() -> Dict[str, any]:
    """Return mock CUDA device properties."""
    return {
        'name': 'Mock NVIDIA GPU',
        'compute_capability': '8.0',
        'total_memory': 16384,  # MB
        'max_threads_per_block': 1024,
        'max_shared_memory_per_block': 49152  # bytes
    }
