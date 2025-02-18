import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path to import AutoSpMV
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from auto_spmv import AutoSpMV

class TestAutoSpMV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv("OPENROUTER_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("OPENROUTER_API_KEY not set")
        cls.optimizer = AutoSpMV(cls.api_key)
        
    def test_compilation(self):
        """Test that baseline code compiles successfully."""
        result = self.optimizer.compile_code(
            self.optimizer.baseline_cpp,
            self.optimizer.baseline_bin
        )
        self.assertTrue(result)
        
    def test_baseline_execution(self):
        """Test that baseline SpMV runs and produces output."""
        runtime, output = self.optimizer.run_binary(self.optimizer.baseline_bin)
        self.assertIsNotNone(runtime)
        self.assertIsNotNone(output)
        
        # Check that output can be parsed as floats
        values = [float(x) for x in output.split()]
        self.assertTrue(len(values) > 0)
        
    def test_output_verification(self):
        """Test output verification function."""
        # Test with identical outputs
        out1 = "1.0 2.0 3.0"
        self.assertTrue(self.optimizer.verify_output(out1, out1))
        
        # Test with different outputs
        out2 = "1.0 2.0 3.1"
        self.assertFalse(self.optimizer.verify_output(out1, out2))
        
        # Test with invalid output
        out3 = "not a number"
        self.assertFalse(self.optimizer.verify_output(out1, out3))

if __name__ == '__main__':
    unittest.main()
