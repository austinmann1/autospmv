import json
import requests
from typing import Dict, Any, Optional
from pathlib import Path

class GPUOptimizer:
    """LLM-based GPU kernel optimization."""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-4-turbo-preview"):
        """Initialize the GPU optimizer.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use (default: GPT-4 Turbo)
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def _construct_system_prompt(self) -> str:
        """Construct the system prompt for GPU optimization."""
        return """You are an expert CUDA programmer specializing in GPU kernel optimization.
Your task is to analyze CUDA kernels and suggest optimizations based on profiling metrics.

Focus on these optimization strategies:
1. Memory Access Patterns
   - Coalesced global memory access
   - Effective use of shared memory
   - Register pressure management
   - Memory bank conflicts

2. Thread and Block Configuration
   - Optimal thread block size
   - Grid configuration for maximum occupancy
   - Warp efficiency
   - Load balancing

3. Instruction-Level Optimization
   - Loop unrolling and loop fusion
   - Instruction-level parallelism
   - Arithmetic intensity
   - Branch divergence reduction

4. Advanced CUDA Features
   - Shared memory usage
   - Warp-level primitives
   - Stream processing
   - Asynchronous operations

When suggesting optimizations:
1. Always explain each optimization with detailed comments starting with "// OPTIMIZATION:"
2. Focus on one or two major optimizations per iteration
3. Maintain numerical accuracy with the baseline
4. Consider the impact on occupancy and memory throughput

Your output should be ONLY the optimized CUDA kernel code, with optimization comments."""
        
    def _construct_messages(self, prompt: str) -> list:
        """Construct message list for the API."""
        return [
            {"role": "system", "content": self._construct_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
    def optimize_kernel(self, prompt: str) -> Optional[str]:
        """Generate an optimized CUDA kernel using the LLM.
        
        Args:
            prompt: Detailed prompt including baseline code and metrics
            
        Returns:
            Optimized CUDA kernel code or None if API call fails
        """
        print("\nMaking LLM API call...")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = self._construct_messages(prompt)
        print("\nPrompt:")
        for msg in messages:
            print(f"\n{msg['role'].upper()}:")
            print(msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'])
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            print("\nSending request to OpenRouter API...")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            print(f"Response status: {response.status_code}")
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"API call failed: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError):
                print(f"Response content: {e.response.text}")
            return None
            
    def analyze_metrics(self, metrics: Dict[str, Any]) -> str:
        """Analyze GPU profiling metrics and suggest optimizations.
        
        Args:
            metrics: Dict containing GPU profiling metrics
            
        Returns:
            String containing analysis and suggestions
        """
        if not self.api_key:
            return "Using mock analysis for local development."
            
        prompt = f"""Analyze these GPU kernel metrics and suggest optimizations:

Runtime: {metrics['runtime_ms']:.2f} ms
Achieved occupancy: {metrics.get('achieved_occupancy', 'N/A')}
SM efficiency: {metrics.get('sm_efficiency', 'N/A')}%
DRAM read throughput: {metrics.get('dram_read_throughput', 'N/A')} GB/s
DRAM write throughput: {metrics.get('dram_write_throughput', 'N/A')} GB/s

Focus on:
1. Memory access patterns
2. Thread block configuration
3. Instruction-level optimizations
4. Resource utilization

Provide a brief analysis and specific suggestions for improvement."""
        
        try:
            print("\nRequesting metric analysis from LLM...")
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": self._construct_messages(prompt)
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"Metric analysis failed: {str(e)}")
            return "Error getting LLM analysis."
