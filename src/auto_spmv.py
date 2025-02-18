import os
import numpy as np
import subprocess
import time
import requests
import json
from pathlib import Path
import re

class AutoSpMV:
    def __init__(self, baseline_cpp, output_dir):
        """Initialize with paths."""
        self.baseline_cpp = Path(baseline_cpp)
        self.output_dir = Path(output_dir)
        self.baseline_bin = self.output_dir / 'baseline'
        self.optimized_cpp = self.output_dir / 'optimized.cpp'
        self.optimized_bin = self.output_dir / 'optimized'
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_dependencies(self, code):
        """Detect which dependencies and features the code uses."""
        deps = {
            'openmp': '#include <omp.h>' in code,
            'eigen': '#include <Eigen/' in code,
            'simd': any(hint in code for hint in ['#pragma omp simd', '#pragma clang loop vectorize', '__builtin_assume_aligned']),
            'avx': '__m256' in code or 'immintrin.h' in code,
            'neon': 'arm_neon.h' in code,
        }
        return deps

    def get_compiler_flags(self, deps):
        """Get appropriate compiler flags based on detected dependencies."""
        flags = ['-O3']  # Always use aggressive optimization
        env = os.environ.copy()
        
        # Platform-specific flags
        if deps['openmp']:
            flags.extend(['-Xpreprocessor', '-fopenmp'])
            flags.append('-I/opt/homebrew/opt/libomp/include')
            flags.append('-L/opt/homebrew/opt/libomp/lib')
            flags.append('-lomp')
            # Update environment for runtime
            env['LIBRARY_PATH'] = f"/opt/homebrew/opt/libomp/lib:{env.get('LIBRARY_PATH', '')}"
            env['CPATH'] = f"/opt/homebrew/opt/libomp/include:{env.get('CPATH', '')}"
            
        if deps['eigen']:
            flags.append('-I/opt/homebrew/opt/eigen/include')
            
        # Architecture-specific flags for M1
        flags.extend(['-mcpu=apple-m1', '-mtune=native'])
        
        # Enable all available SIMD features
        if deps['simd'] or deps['neon']:
            flags.extend(['-march=armv8.2-a+fp16+neon+crc'])
            
        return flags, env

    def verify_openmp(self):
        """Verify that OpenMP is working correctly."""
        test_file = self.output_dir / 'omp_test.cpp'
        test_bin = self.output_dir / 'omp_test'
        
        # Simple OpenMP test program
        test_code = """
        #include <omp.h>
        #include <iostream>
        
        int main() {
            if(omp_get_max_threads() > 1) {
                std::cout << "OpenMP working with " << omp_get_max_threads() << " threads\\n";
                return 0;
            }
            return 1;
        }
        """
        
        try:
            # Write test file
            test_file.write_text(test_code)
            
            # Try to compile with OpenMP flags
            flags = ['-Xpreprocessor', '-fopenmp', '-I/opt/homebrew/opt/libomp/include', 
                    '-L/opt/homebrew/opt/libomp/lib', '-lomp']
            cmd = ['g++'] + flags + [str(test_file), '-o', str(test_bin)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, result.stderr
            
            # Try to run
            result = subprocess.run([str(test_bin)], capture_output=True, text=True)
            success = result.returncode == 0
            
            # Cleanup
            if test_file.exists():
                test_file.unlink()
            if test_bin.exists():
                test_bin.unlink()
                
            return success, result.stdout if success else result.stderr
            
        except Exception as e:
            return False, str(e)
        finally:
            # Ensure cleanup
            if test_file.exists():
                test_file.unlink()
            if test_bin.exists():
                test_bin.unlink()

    def compile_code(self, src, bin_path):
        """Compile code with OpenMP support."""
        cmd = [
            'clang++',  # Use Apple Clang
            '-O3',
            '-Xpreprocessor', '-fopenmp',  # OpenMP for Clang
            '-I/opt/homebrew/opt/libomp/include',
            '-L/opt/homebrew/opt/libomp/lib',
            '-lomp',  # Link OpenMP runtime
            str(src),
            '-o',
            str(bin_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, result.stderr
            return True, None
        except Exception as e:
            return False, str(e)

    def run_binary(self, binary):
        """Run a binary and measure its execution time."""
        try:
            start = time.time()
            result = subprocess.run(
                [binary], capture_output=True, text=True, timeout=5
            )
            elapsed = time.time() - start
            return elapsed, result.stdout.strip()
        except subprocess.TimeoutExpired:
            print(f"Binary {binary} timed out after 5 seconds")
            return None, None
        except Exception as e:
            print(f"Error running binary {binary}: {e}")
            return None, None

    def verify_output(self, candidate_out, reference_out):
        """Verify that candidate output matches reference output."""
        try:
            candidate = np.array([float(x) for x in candidate_out.split()])
            reference = np.array([float(x) for x in reference_out.split()])
            # Use a larger tolerance for floating-point comparisons
            return np.allclose(candidate, reference, rtol=1e-4, atol=1e-4)
        except Exception as e:
            print(f"Error verifying output: {e}")
            return False

    def get_candidate_code(self, prompt):
        """Get candidate code from LLM."""
        print("\n=== SENDING PROMPT TO LLM ===")
        print(prompt)
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/codeium/cascade", 
                    "X-Title": "Cascade IDE"
                },
                json={
                    "model": "openai/gpt-3.5-turbo",  # Use GPT-3.5 for faster testing
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7  # Add some randomness to explore different optimizations
                }
            ).json()
            
            if "choices" not in response or not response["choices"]:
                print("Error: No choices in response")
                print(f"Response: {response}")
                return None
            
            print("\n=== LLM RESPONSE ===")
            content = response["choices"][0]["message"]["content"]
            print(content)
            print("\n=== END RESPONSE ===")
            
            # Extract code between cpp delimiters
            if '```cpp' not in content or '```' not in content.split('```cpp', 1)[1]:
                print("No valid code block found")
                return None
                
            spmv_code = content.split('```cpp', 1)[1].split('```', 1)[0].strip()
            
            # Allow include statements before the spmv function
            spmv_start = spmv_code.find('std::vector<double> spmv')
            if spmv_start == -1:
                print("Response must contain spmv function signature")
                return None
            
            # Keep any includes and the spmv function
            spmv_code = spmv_code[:spmv_start] + spmv_code[spmv_start:]
            
            # Load baseline code
            baseline = self.baseline_cpp.read_text()
            
            # Find the spmv function in baseline
            baseline_spmv_start = baseline.find('std::vector<double> spmv')
            if baseline_spmv_start == -1:
                print("Could not find spmv function in baseline")
                return None
                
            # Find the end of the spmv function
            main_start = baseline.find('int main', baseline_spmv_start)
            if main_start == -1:
                print("Could not find main function after spmv")
                return None
                
            # Count backwards to find the last } before main
            spmv_end = baseline.rfind('}', baseline_spmv_start, main_start) + 1
            while spmv_end < len(baseline) and baseline[spmv_end].isspace():
                spmv_end += 1
                
            # Replace just the spmv function, preserving any includes
            new_code = baseline[:baseline_spmv_start] + spmv_code + baseline[spmv_end:]
            
            # Write to optimized.cpp
            self.optimized_cpp.write_text(new_code)
            
            return new_code
            
        except Exception as e:
            print(f"Error getting code from LLM: {e}")
            return None

    def analyze_error(self, error_msg):
        """Analyze compilation/runtime errors and provide specific guidance."""
        analysis = []
        suggestions = []
        
        # OpenMP errors
        if "omp.h' file not found" in error_msg:
            analysis.append("OpenMP header not found")
            suggestions.extend([
                "Make sure to use the correct include path: /opt/homebrew/opt/libomp/include",
                "Consider using platform-independent vectorization hints if OpenMP is not critical"
            ])
        elif "undefined symbol: _GOMP" in error_msg:
            analysis.append("OpenMP runtime library not found")
            suggestions.append("Ensure OpenMP runtime is properly linked with -L/opt/homebrew/opt/libomp/lib -lomp")
        
        # SIMD errors
        elif "error: use of undeclared identifier '__builtin_ia32" in error_msg:
            analysis.append("x86 SIMD intrinsics are not supported on ARM")
            suggestions.extend([
                "Use ARM NEON intrinsics instead (#include <arm_neon.h>)",
                "Or use platform-independent vectorization hints (#pragma omp simd)"
            ])
        
        # Library errors
        elif "Eigen" in error_msg:
            analysis.append("Eigen library error")
            suggestions.append("Check Eigen include path and version compatibility")
        
        # Common C++ errors
        elif "no member named" in error_msg:
            analysis.append("Missing std:: namespace qualifier or include")
            suggestions.extend([
                "Add missing include directives (#include <iostream>, etc.)",
                "Use std:: namespace qualifier for standard library items"
            ])
        elif "use of undeclared identifier" in error_msg:
            analysis.append("Variable or function not declared")
            suggestions.extend([
                "Declare variables before use",
                "Check for typos in variable names",
                "Make sure all required functions are defined"
            ])
        elif "invalid conversion" in error_msg or "no matching function" in error_msg:
            analysis.append("Type mismatch or incorrect function signature")
            suggestions.append("Review data types and function parameters")
        elif "expected" in error_msg:
            analysis.append("Syntax error in code")
            suggestions.extend([
                "Check for missing semicolons, braces, or parentheses",
                "Verify all code blocks are properly closed"
            ])
        
        # If no specific error caught, provide general guidance
        if not analysis:
            analysis.append("Compilation error")
            suggestions.extend([
                "Check for missing includes",
                "Verify all variables are declared",
                "Ensure proper use of std:: namespace"
            ])
        
        return analysis, suggestions

    def analyze_performance(self, cand_time, base_time, compile_output="", runtime_output=""):
        """Analyze performance and generate feedback."""
        feedback = []
        suggestions = []
        
        # Compilation analysis
        if compile_output:
            if "error: use of undeclared identifier '__builtin_ia32" in compile_output:
                feedback.append("x86 SIMD intrinsics failed to compile on ARM platform.")
                suggestions.append("Use ARM NEON intrinsics or platform-independent vectorization hints.")
            elif "omp.h" in compile_output and "file not found" in compile_output:
                feedback.append("OpenMP header not found. Installing OpenMP support.")
                suggestions.append("OpenMP support has been added. Try running again.")
        
        # Performance analysis
        if cand_time is not None and base_time is not None:
            perf_change = (base_time - cand_time) / base_time * 100
            if perf_change > 0:
                feedback.append(f"Runtime improved by {perf_change:.1f}%")
                if perf_change > 20:
                    suggestions.append("Significant improvement! Consider further loop optimizations while maintaining the successful changes.")
                else:
                    suggestions.append("Moderate improvement. Try combining with cache-friendly access patterns.")
            else:
                feedback.append(f"Runtime degraded by {-perf_change:.1f}%")
                suggestions.append("Focus on reducing memory operations and improving data locality.")
        
        return "\n".join(feedback), "\n".join(suggestions)

    def optimize(self, max_iterations=20, perf_threshold=0.85):
        """Main optimization loop with verifier-guided refinement."""
        print("=== Starting AutoSpMV Optimization ===")
        
        # Run baseline multiple times and take the median to get a stable baseline
        baseline_times = []
        for _ in range(5):
            # Compile baseline
            success, output = self.compile_code(self.baseline_cpp, self.baseline_bin)
            if not success:
                print(f"Failed to compile baseline: {output}")
                return
            
            # Run baseline
            base_time, base_output = self.run_binary(self.baseline_bin)
            if not base_time:
                print("Failed to run baseline")
                return
            baseline_times.append(base_time)
        
        # Use median baseline time and store baseline output
        base_time = sorted(baseline_times)[len(baseline_times)//2]
        print(f"Baseline runtime (median of {len(baseline_times)} runs): {base_time:.4f}s\n")
        
        # Initialize optimization history
        best_time = float('inf')
        optimization_history = []
        best_code = None
        
        for iteration in range(max_iterations):
            print(f"=== Iteration {iteration + 1}/{max_iterations} ===\n")
            
            # Build prompt with optimization history
            history_entries = []
            for i, (code, time, error, analysis, suggestions) in enumerate(optimization_history[-3:]):
                entry = f"Iteration {len(optimization_history) - len(optimization_history[-3:]) + i + 1}:\n"
                if time:
                    entry += f"- Runtime: {time:.4f}s\n"
                else:
                    entry += "- Runtime: Failed\n"
                entry += f"- Feedback: {error}\n"
                if analysis:
                    entry += f"  Raw Error: {error}\n"
                    entry += f"  Analysis: {analysis[0]}\n"
                    entry += f"  Suggestions: {'; '.join(suggestions)}\n"
                # Add the code from this iteration
                if code:
                    # Extract just the spmv function if it exists
                    spmv_start = code.find('std::vector<double> spmv')
                    if spmv_start != -1:
                        main_start = code.find('int main', spmv_start)
                        if main_start != -1:
                            spmv_end = code.rfind('}', spmv_start, main_start) + 1
                            spmv_code = code[spmv_start:spmv_end].strip()
                            entry += f"- Code:\n```cpp\n{spmv_code}\n```\n"
                history_entries.append(entry)
            
            history_summary = "\n".join(history_entries) if history_entries else "No previous attempts"
            
            # Add best performing code if we have it
            best_code_section = ""
            if best_code and best_time < float('inf'):
                spmv_start = best_code.find('std::vector<double> spmv')
                if spmv_start != -1:
                    main_start = best_code.find('int main', spmv_start)
                    if main_start != -1:
                        spmv_end = best_code.rfind('}', spmv_start, main_start) + 1
                        best_spmv = best_code[spmv_start:spmv_end].strip()
                        best_code_section = f"""
Best performing implementation so far (runtime: {best_time:.4f}s):
```cpp
{best_spmv}
```
"""
            
            prompt = f"""You are an expert C++ developer optimizing a Sparse Matrix-Vector Multiplication (SpMV) kernel.
            Current best runtime: {min(best_time, base_time):.4f}s (baseline: {base_time:.4f}s)
            Previous optimizations attempted: {len(optimization_history)} times
            
            Recent optimization history with full errors and analysis:
            {history_summary}
            
            {best_code_section}
            
            TASK: Optimize the spmv() function below. DO NOT modify any other code.
            
            AVAILABLE OPTIMIZATIONS:
            1. OpenMP parallelization:
               - #include <omp.h> is supported
               - Use -fopenmp flag automatically
               - Headers in /opt/homebrew/opt/libomp/include
               - IMPORTANT: Since each thread works on different rows, NO atomic operations needed
               - Consider using schedule(static) for balanced workload
            2. ARM NEON SIMD intrinsics:
               - #include <arm_neon.h>
               - Native M1 support
            3. Auto-vectorization:
               - #pragma omp simd
               - #pragma clang loop vectorize
            4. Loop optimizations:
               - Unrolling and tiling
               - Cache-friendly patterns
            
            OPTIMIZATION TIPS:
            1. Each row of the matrix can be processed independently - NO NEED for atomic operations
            2. Use OpenMP's static schedule since workload is uniform (1% density)
            3. Each thread writes to different y[i] indices, so NO race conditions possible
            4. Consider using temporary variables for better cache usage
            
            Your response must ONLY contain the optimized spmv function between ```cpp and ``` delimiters.
            Here is the baseline implementation:
            
            ```cpp
            std::vector<double> spmv(const CSRMatrix& A, const std::vector<double>& x) {{
    std::vector<double> y(A.rows, 0.0);
    
    for (int i = 0; i < A.rows; i++) {{
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {{
            y[i] += A.values[j] * x[A.col_indices[j]];
        }}
    }}
    
    return y;
}}
            ```
            """
            
            # Get candidate code from LLM
            candidate = self.get_candidate_code(prompt)
            if not candidate:
                print("Failed to get valid code from LLM")
                continue
            
            # Compile candidate
            success, output = self.compile_code(self.optimized_cpp, self.optimized_bin)
            if not success:
                analysis, suggestions = self.analyze_error(output)
                optimization_history.append((candidate, None, "Compilation failed", analysis, suggestions))
                print(f"Failed to compile: {output}")
                continue
                
            # Run candidate multiple times and take median
            candidate_times = []
            for _ in range(5):
                cand_time, cand_output = self.run_binary(self.optimized_bin)
                if not cand_time:
                    break
                if not self.verify_output(cand_output, base_output):
                    break
                candidate_times.append(cand_time)
            
            if len(candidate_times) < 5:
                optimization_history.append((candidate, None, "Runtime failed or output verification failed", None, None))
                print("Failed to run candidate or output verification failed")
                continue
            
            # Use median candidate time
            cand_time = sorted(candidate_times)[len(candidate_times)//2]
                
            # Update best if improved
            if cand_time < best_time:
                best_time = cand_time
                best_code = candidate
                print(f"New best time: {best_time:.4f}s")
                
            # Add to history
            feedback, suggestions = self.analyze_performance(cand_time, base_time)
            optimization_history.append((candidate, cand_time, feedback, None, suggestions.split('; ')))
            
            # Check if good enough
            if best_time <= base_time * perf_threshold:
                print(f"\nReached performance threshold ({perf_threshold*100}% of baseline)")
                break
                
        print("\nOptimization complete")
        if best_code and best_time < base_time:
            speedup = (base_time - best_time) / base_time * 100
            print(f"Best speedup: {speedup:.1f}%")
        else:
            print("No successful optimizations")

if __name__ == "__main__":
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        exit(1)
        
    optimizer = AutoSpMV('baseline.cpp', 'output')
    optimizer.optimize()
