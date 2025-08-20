#!/usr/bin/env python3
"""Test RANGE operation in TTNN backend"""

import os
os.environ["TTNN"] = "1"

from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
import numpy as np

def test_simple_loop():
    """Test a simple loop that adds a constant in each iteration"""
    print("\n=== Testing simple loop with RANGE ===")
    
    # Create a simple computation with a loop
    # This should sum numbers from 0 to 4 (0+1+2+3+4 = 10)
    device = Device["TTNN"]
    
    # Create a tensor and perform a reduction that will use RANGE internally
    x = Tensor.arange(5, device=device)
    result = x.sum()
    
    print(f"Sum of 0..4 = {result.numpy()}")
    expected = 10
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)
    print("✅ Simple loop test passed!")

def test_nested_computation():
    """Test computation with RANGE in more complex scenarios"""
    print("\n=== Testing nested computation ===")
    
    device = Device["TTNN"]
    
    # Matrix multiplication internally uses loops
    a = Tensor([[1, 2], [3, 4]], device=device)
    b = Tensor([[5, 6], [7, 8]], device=device)
    c = a @ b
    
    print(f"Matrix multiplication result:\n{c.numpy()}")
    expected = np.array([[19, 22], [43, 50]])
    np.testing.assert_allclose(c.numpy(), expected, rtol=1e-5)
    print("✅ Nested computation test passed!")

if __name__ == "__main__":
    if not getenv("TTNN"):
        print("TTNN backend not enabled. Set TTNN=1 to run tests.")
        exit(0)
    
    try:
        test_simple_loop()
        test_nested_computation()
        print("\n🎉 All RANGE tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

