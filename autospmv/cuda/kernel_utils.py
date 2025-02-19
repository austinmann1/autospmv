import subprocess
import re
from pathlib import Path
from typing import Dict, Tuple, Optional
import shutil

class CUDACompiler:
    """Handles CUDA kernel compilation and profiling."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_cuda_available()
        
    def _check_cuda_available(self):
        """Check if CUDA toolkit is available."""
        self.has_cuda = shutil.which('nvcc') is not None
        if not self.has_cuda:
            print("CUDA not found, using mock environment")
        
    def compile_kernel(self, source_file: Path, output_file: Path,
                      arch: str = "sm_80") -> Tuple[bool, str]:
        """Compile a CUDA source file using nvcc."""
        if not self.has_cuda:
            # Mock compilation
            try:
                # Read the source file to check basic syntax
                with open(source_file, 'r') as f:
                    source = f.read()
                    
                if not "__global__" in source:
                    return False, "Error: No CUDA kernel found (missing __global__)"
                    
                # Create a mock binary file
                with open(output_file, 'w') as f:
                    f.write("MOCK_CUDA_BINARY")
                    
                return True, "Mock compilation successful"
            except Exception as e:
                return False, f"Mock compilation failed: {str(e)}"
        
        # Real compilation
        cmd = [
            "nvcc",
            f"-arch={arch}",
            "-O3",
            "--compiler-options",
            "'-fPIC'",
            "-o", str(output_file),
            str(source_file)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr

    def profile_kernel(self, binary: Path) -> Dict[str, float]:
        """Profile a CUDA binary using nvprof."""
        if not self.has_cuda:
            # Return mock metrics
            import random
            return {
                'achieved_occupancy': random.uniform(0.6, 0.9),
                'sm_efficiency': random.uniform(70.0, 95.0),
                'dram_read_throughput': random.uniform(100.0, 300.0),
                'dram_write_throughput': random.uniform(80.0, 250.0)
            }
            
        metrics = [
            "achieved_occupancy",
            "sm_efficiency",
            "dram_read_throughput",
            "dram_write_throughput"
        ]
        
        cmd = [
            "nvprof",
            "--metrics", ",".join(metrics),
            str(binary)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            return self._parse_nvprof_output(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Profiling failed: {e.stderr}")
            return {}

    def _parse_nvprof_output(self, output: str) -> Dict[str, float]:
        """Parse nvprof output to extract metrics."""
        metrics = {}
        
        # Regular expressions for different metric patterns
        patterns = {
            "achieved_occupancy": r"achieved_occupancy\s+(\d+\.\d+)",
            "sm_efficiency": r"sm_efficiency\s+(\d+\.\d+)",
            "dram_read_throughput": r"dram_read_throughput\s+(\d+\.\d+)",
            "dram_write_throughput": r"dram_write_throughput\s+(\d+\.\d+)"
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[metric] = float(match.group(1))
                
        return metrics

    def get_optimization_feedback(self, metrics: Dict[str, float]) -> str:
        """Generate optimization feedback based on profiling metrics."""
        feedback = []
        
        if metrics.get("achieved_occupancy", 1.0) < 0.7:
            feedback.append("Low GPU occupancy. Consider adjusting block size or reducing register usage.")
            
        if metrics.get("sm_efficiency", 100.0) < 80.0:
            feedback.append("Low SM efficiency. Check for thread divergence or load imbalance.")
            
        read_tp = metrics.get("dram_read_throughput", 0.0)
        write_tp = metrics.get("dram_write_throughput", 0.0)
        if read_tp + write_tp < 100.0:  # GB/s, adjust threshold as needed
            feedback.append("Low memory throughput. Consider using shared memory or improving memory access patterns.")
            
        return " ".join(feedback) if feedback else "Performance metrics look good."
