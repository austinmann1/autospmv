from pathlib import Path
import json
import time
import os
from typing import Dict, Any, List, Tuple, Optional
import torch

from .kernel_utils import CUDACompiler
from ..benchmarks.kernelbench_utils import KernelBenchWrapper
from ..llm.gpu_optimizer import GPUOptimizer

class AutoCUDA:
    """Main orchestrator for GPU kernel optimization."""
    
    def __init__(self, output_dir: Path, task_id: str = "mock_spmv_1"):
        """Initialize the GPU kernel optimization system.
        
        Args:
            output_dir: Directory for outputs (compiled kernels, logs)
            task_id: KernelBench task ID to optimize for
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.compiler = CUDACompiler(self.output_dir)
        self.kernelbench = KernelBenchWrapper(use_mock=True)
        self.task_id = task_id
        
        # Initialize LLM optimizer
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Warning: OPENROUTER_API_KEY not set, using mock LLM responses")
            self.llm = None
        else:
            print(f"Initializing LLM with API key: {api_key[:8]}...")
            self.llm = GPUOptimizer(api_key)
        
        # Setup logging
        self.log_file = self.output_dir / "optimization_log.json"
        self.optimization_history: List[Dict[str, Any]] = []
        
    def _load_baseline(self) -> Tuple[Path, Dict[str, Any]]:
        """Load and compile baseline kernel."""
        baseline_file = Path(__file__).parent / "baseline_spmv.cu"
        baseline_binary = self.output_dir / "baseline.cubin"
        
        success, message = self.compiler.compile_kernel(baseline_file, baseline_binary)
        if not success:
            raise RuntimeError(f"Failed to compile baseline: {message}")
            
        # Get baseline performance
        inputs = self.kernelbench.get_task_inputs(self.task_id)
        metrics = self.kernelbench.run_pytorch_baseline(self.task_id, inputs)
        
        return baseline_binary, metrics
        
    def _evaluate_candidate(self, candidate_file: Path) -> Optional[Dict[str, Any]]:
        """Compile and evaluate a candidate kernel."""
        candidate_binary = self.output_dir / f"{candidate_file.stem}.cubin"
        
        # Compile candidate
        success, message = self.compiler.compile_kernel(candidate_file, candidate_binary)
        if not success:
            print(f"Compilation failed: {message}")
            self.optimization_history.append({
                "iteration": len(self.optimization_history),
                "error": f"Compilation error: {message}",
                "candidate_file": str(candidate_file)
            })
            return None
            
        # Run performance evaluation
        try:
            metrics = self.kernelbench.run_cuda_kernel(
                self.task_id,
                candidate_binary,
                collect_metrics=True
            )
            
            # Extract key changes from the candidate code
            candidate_code = candidate_file.read_text()
            key_changes = self._extract_key_changes(candidate_code)
            
            # Add to history with key changes
            self.optimization_history.append({
                "iteration": len(self.optimization_history),
                "metrics": metrics,
                "candidate_file": str(candidate_file),
                "key_changes": key_changes
            })
            
            return metrics
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            self.optimization_history.append({
                "iteration": len(self.optimization_history),
                "error": f"Runtime error: {str(e)}",
                "candidate_file": str(candidate_file)
            })
            return None
            
    def _extract_key_changes(self, candidate_code: str) -> List[str]:
        """Extract key changes from candidate code based on comments."""
        key_changes = []
        for line in candidate_code.split('\n'):
            if '// OPTIMIZATION:' in line:
                change = line.split('// OPTIMIZATION:')[1].strip()
                key_changes.append(change)
        return key_changes
        
    def _generate_llm_prompt(self, baseline_code: str, 
                           current_candidate: Optional[str] = None,
                           metrics: Optional[Dict[str, Any]] = None) -> str:
        """Generate prompt for the LLM with GPU-specific context."""
        prompt = f"""Here is the baseline CUDA SpMV kernel implementation:

{baseline_code}

Past Optimization Attempts:
"""
        # Add history of past attempts
        if self.optimization_history:
            for attempt in self.optimization_history[-3:]:  # Show last 3 attempts
                prompt += f"\nAttempt {attempt['iteration'] + 1}:\n"
                if 'error' in attempt:
                    prompt += f"Failed: {attempt['error']}\n"
                else:
                    prompt += f"Runtime: {attempt['metrics']['runtime_ms']:.2f} ms\n"
                    if 'achieved_occupancy' in attempt['metrics']:
                        prompt += f"Occupancy: {attempt['metrics']['achieved_occupancy']:.2%}\n"
                    prompt += "Key Changes:\n"
                    if 'key_changes' in attempt:
                        for change in attempt['key_changes']:
                            prompt += f"- {change}\n"
                    
        prompt += "\nCurrent Performance Metrics:\n"
        if metrics:
            prompt += f"""
- Runtime: {metrics['runtime_ms']:.2f} ms
- Achieved occupancy: {metrics.get('achieved_occupancy', 'N/A')}
- SM efficiency: {metrics.get('sm_efficiency', 'N/A')}%
- DRAM read throughput: {metrics.get('dram_read_throughput', 'N/A')} GB/s
- DRAM write throughput: {metrics.get('dram_write_throughput', 'N/A')} GB/s

Analysis of Current Performance:
"""
            if self.llm:
                prompt += self.llm.analyze_metrics(metrics)
            else:
                prompt += "Using mock analysis for local development."
                
        prompt += """

Requirements for the Optimized Implementation:
1. Must maintain numerical accuracy with the baseline
2. Focus on:
   - Coalesced memory access patterns
   - Optimal thread block configuration
   - Effective use of shared memory
   - Reduced thread divergence
   - Instruction-level parallelism

Please provide an optimized CUDA kernel implementation with:
1. Detailed comments explaining each optimization
2. Clear reasoning for thread block size choices
3. Documentation of any assumptions or limitations

Generate only the CUDA kernel code without any surrounding explanation."""
        
        return prompt
        
    def _tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert a tensor to a serializable dictionary."""
        return {
            'type': 'tensor',
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'data': tensor.tolist()
        }
        
    def _serialize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics dictionary to JSON-serializable format."""
        serialized = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                serialized[k] = self._tensor_to_dict(v)
            elif isinstance(v, (int, float, str, bool, list, dict)):
                serialized[k] = v
            else:
                serialized[k] = str(v)
        return serialized
        
    def optimize(self, max_iterations: int = 20,
                perf_threshold: float = 0.85) -> Dict[str, Any]:
        """Run the main optimization loop.
        
        Args:
            max_iterations: Maximum number of optimization attempts
            perf_threshold: Target performance threshold (vs. PyTorch baseline)
            
        Returns:
            Dict containing the best candidate and its metrics
        """
        # Load and compile baseline
        baseline_binary, baseline_metrics = self._load_baseline()
        baseline_code = (Path(__file__).parent / "baseline_spmv.cu").read_text()
        
        best_candidate = None
        best_metrics = None
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            
            # Generate candidate code using LLM
            prompt = self._generate_llm_prompt(
                baseline_code,
                current_candidate=best_candidate,
                metrics=best_metrics
            )
            
            if self.llm:
                candidate_code = self.llm.optimize_kernel(prompt)
                if not candidate_code:
                    print("Failed to get LLM response, skipping iteration")
                    continue
            else:
                # For mock environment, use baseline with minor modifications
                candidate_code = baseline_code.replace(
                    "const int block_size = 256",
                    f"const int block_size = {256 + iteration * 32}"
                )
                candidate_code = candidate_code.replace(
                    "float sum = 0.0f;",
                    "// OPTIMIZATION: Unrolled inner loop for better instruction-level parallelism\n"
                    "float sum1 = 0.0f, sum2 = 0.0f;"
                )
            
            # Save candidate
            candidate_file = self.output_dir / f"candidate_{iteration}.cu"
            candidate_file.write_text(candidate_code)
            
            # Evaluate candidate
            metrics = self._evaluate_candidate(candidate_file)
            if not metrics:
                continue
                
            # Update best if improved
            if (not best_metrics or 
                metrics["runtime_ms"] < best_metrics["runtime_ms"]):
                best_candidate = candidate_code
                best_metrics = metrics
                
            # Save optimization history
            with open(self.log_file, "w") as f:
                json.dump({
                    "baseline_metrics": self._serialize_metrics(baseline_metrics),
                    "best_metrics": self._serialize_metrics(best_metrics) if best_metrics else None,
                    "best_candidate": best_candidate,
                    "optimization_history": [
                        {
                            **h,
                            "metrics": self._serialize_metrics(h["metrics"]) if "metrics" in h else None
                        }
                        for h in self.optimization_history
                    ]
                }, f, indent=2)
                
            # Check if we've met our performance target
            if best_metrics:
                speedup = baseline_metrics["runtime_ms"] / best_metrics["runtime_ms"]
                if speedup >= perf_threshold:
                    print(f"\nReached performance target with {speedup:.2f}x speedup")
                    break
                    
        return {
            "baseline_metrics": baseline_metrics,
            "best_metrics": best_metrics,
            "best_candidate": best_candidate,
            "optimization_history": self.optimization_history
        }
