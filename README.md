# AutoSpMV: Automated Sparse Matrix-Vector Multiplication Optimization

AutoSpMV is an automated optimization system for Sparse Matrix-Vector Multiplication (SpMV) using LLM-guided code generation. It leverages OpenAI's GPT models to iteratively optimize SpMV implementations, focusing on:

- OpenMP parallelization
- SIMD vectorization
- Loop optimizations
- Cache utilization

## Inspiration

This project was inspired by NVIDIA's work on using LLMs for CUDA kernel optimization, as presented in their paper ["Optimizing CUDA Kernels via LLM-Guided Evolution"](https://arxiv.org/abs/2402.00882). While NVIDIA focused on CUDA-specific optimizations using genetic algorithms guided by LLMs, AutoSpMV adapts this concept for CPU-based SpMV kernels, using iterative refinement with OpenMP and SIMD optimizations.

## Features

- Automated code generation and optimization
- Performance verification with numerical validation
- Support for OpenMP and ARM NEON SIMD
- Stable performance benchmarking using median of multiple runs
- Iterative optimization with history-aware prompting

## Requirements

- C++ compiler with OpenMP support
- Python 3.8+
- OpenAI API key (or compatible LLM API)

## Installation

```bash
git clone https://github.com/yourusername/autospmv.git
cd autospmv
pip install -r requirements.txt
```

## Usage

1. Set your API key:
```bash
export OPENROUTER_API_KEY=your-api-key
```

2. Run the optimizer:
```bash
python src/auto_spmv.py
```

## Project Structure

```
autospmv/
├── src/
│   ├── auto_spmv.py      # Main optimization loop
│   ├── baseline_spmv.cpp # Baseline implementation
│   └── test_omp.cpp      # OpenMP test utilities
├── tests/
│   └── test_spmv.py      # Test suite
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## How It Works

1. Starts with a baseline SpMV implementation
2. Iteratively generates optimized versions using LLM
3. Compiles and tests each candidate
4. Verifies numerical correctness
5. Measures performance (median of 5 runs)
6. Updates optimization history for context-aware improvements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
