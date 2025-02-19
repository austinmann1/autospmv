import pytest
from pathlib import Path
import torch
import json
import tempfile
import shutil

from src.cuda.auto_cuda import AutoCUDA

class TestAutoCUDA:
    """Test suite for AutoCUDA optimization system."""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.output_dir = cls.temp_dir / "output"
        cls.output_dir.mkdir(parents=True)
        
    @classmethod
    def teardown_class(cls):
        """Cleanup test artifacts."""
        shutil.rmtree(cls.temp_dir)
        
    def test_initialization(self):
        """Test basic initialization."""
        auto_cuda = AutoCUDA(self.output_dir)
        assert auto_cuda.output_dir.exists()
        assert auto_cuda.task_id == "mock_spmv_1"
        
    @pytest.mark.basic
    def test_baseline_loading(self):
        """Test baseline kernel loading and compilation."""
        auto_cuda = AutoCUDA(self.output_dir)
        baseline_binary, metrics = auto_cuda._load_baseline()
        
        assert baseline_binary.exists()
        assert "runtime_ms" in metrics
        assert metrics["runtime_ms"] > 0
        
    @pytest.mark.basic
    def test_candidate_evaluation(self):
        """Test candidate kernel evaluation."""
        auto_cuda = AutoCUDA(self.output_dir)
        
        # Create a test candidate file
        candidate_file = self.output_dir / "test_candidate.cu"
        baseline_file = Path(__file__).parent.parent / "src" / "cuda" / "baseline_spmv.cu"
        shutil.copy(baseline_file, candidate_file)
        
        metrics = auto_cuda._evaluate_candidate(candidate_file)
        assert metrics is not None
        assert "runtime_ms" in metrics
        assert "achieved_occupancy" in metrics
        assert "sm_efficiency" in metrics
        
    @pytest.mark.basic
    def test_prompt_generation(self):
        """Test LLM prompt generation with metrics."""
        auto_cuda = AutoCUDA(self.output_dir)
        
        # Test baseline-only prompt
        baseline_code = "// Test baseline code"
        prompt = auto_cuda._generate_llm_prompt(baseline_code)
        assert "baseline" in prompt.lower()
        assert "cuda kernel" in prompt.lower()
        
        # Test prompt with metrics
        metrics = {
            "runtime_ms": 1.5,
            "achieved_occupancy": 0.75,
            "sm_efficiency": 85.0,
            "dram_read_throughput": 100.0,
            "dram_write_throughput": 80.0
        }
        prompt = auto_cuda._generate_llm_prompt(
            baseline_code,
            current_candidate="// Test candidate",
            metrics=metrics
        )
        assert "current candidate" in prompt.lower()
        assert "performance metrics" in prompt.lower()
        assert "1.50 ms" in prompt
        assert "85.0%" in prompt
        
    @pytest.mark.basic
    def test_optimization_loop(self):
        """Test main optimization loop."""
        auto_cuda = AutoCUDA(self.output_dir)
        
        # Run a short optimization
        result = auto_cuda.optimize(max_iterations=2)
        
        assert "best_candidate" in result
        assert "best_metrics" in result
        assert "baseline_metrics" in result
        assert "optimization_history" in result
        assert len(result["optimization_history"]) <= 2
        
        # Check log file
        assert auto_cuda.log_file.exists()
        with open(auto_cuda.log_file) as f:
            log_data = json.load(f)
            
        # Verify log structure
        assert isinstance(log_data, dict)
        assert "baseline_metrics" in log_data
        assert "best_metrics" in log_data
        assert "best_candidate" in log_data
        assert "optimization_history" in log_data
        assert isinstance(log_data["optimization_history"], list)
        
        # Verify metrics serialization
        assert "runtime_ms" in log_data["baseline_metrics"]
        if log_data["best_metrics"]:
            assert "runtime_ms" in log_data["best_metrics"]
            
        # Verify optimization progress
        assert len(log_data["optimization_history"]) <= 2
        for entry in log_data["optimization_history"]:
            assert "iteration" in entry
            if "metrics" in entry:
                assert "runtime_ms" in entry["metrics"]
