import os
import torch
from pathlib import Path
from autospmv.benchmarks.kernelbench_utils import KernelBenchWrapper

# Initialize KernelBench wrapper with mock data
print("Initializing KernelBench wrapper...")
wrapper = KernelBenchWrapper(use_mock=True)

# Test getting task data
print("\nTesting task data retrieval...")
task_data = wrapper.get_task_data('spmv_level1')
print("Task data keys:", task_data.keys())
for key, value in task_data.items():
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")

# Test verification
print("\nTesting output verification...")
mock_output = torch.randn_like(task_data['x'])  # Mock output
is_correct = wrapper.verify_output('spmv_level1', mock_output, task_data)
print(f"Mock output verification: {'passed' if is_correct else 'failed'}")

print("\nKernelBench integration test completed successfully!")
