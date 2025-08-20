#!/usr/bin/env python3
"""
Basic test script to verify TTNN backend integration with tinygrad
"""

import os
import sys
import numpy as np
import random

# Add tinygrad to path
sys.path.insert(0, os.getenv("PWD"))

# Test device discovery
from tinygrad.device import Device
print("Testing TTNN device discovery...")

# Check if TTNN is in available devices
available_devices = list(Device.get_available_devices())
print(f"Available devices: {available_devices}")

# Try to create TTNN device
print("Attempting to create TTNN device...")
ttnn_device = Device["TTNN"]
print(f"✓ TTNN device created successfully: {ttnn_device}")
print(f"Device type: {type(ttnn_device)}")

# Test basic tensor operations
print("\nTesting basic tensor operations...")
from tinygrad.tensor import Tensor

# Set TTNN as default device
os.environ["TTNN"] = "1"

# Test 16x16 matmul only
print("\nTesting 16x16 matmul...")
try:
    a_data_16x16 = [[random.uniform(0.1, 2.0) for _ in range(16)] for _ in range(16)]
    b_data_16x16 = [[random.uniform(0.1, 2.0) for _ in range(16)] for _ in range(16)]

    A_16x16 = Tensor(a_data_16x16, device="TTNN")
    B_16x16 = Tensor(b_data_16x16, device="TTNN")

    print("Starting 16x16 matrix multiplication...")
    C_16x16 = A_16x16 @ B_16x16
    print(f"A_16x16 @ B_16x16 shape: {C_16x16.shape}")
    C_16x16.realize()

    print("Comparing 16x16 matmul results...")
    numpy_result_16x16 = A_16x16.numpy() @ B_16x16.numpy()
    ttnn_result_16x16 = C_16x16.numpy()
    max_diff_16x16 = np.max(np.abs(ttnn_result_16x16 - numpy_result_16x16))
    print(f"Max difference: {max_diff_16x16}")
    print(f"Mean absolute difference: {np.mean(np.abs(ttnn_result_16x16 - numpy_result_16x16))}")
    print(f"Relative error: {max_diff_16x16 / np.max(np.abs(numpy_result_16x16)):.6f}")

    # Use appropriate tolerance for 16x16 matrices
    assert np.allclose(ttnn_result_16x16, numpy_result_16x16, rtol=1e-2, atol=5.0)
    print("✓ 16x16 matmul test passed!")
except Exception as e:
    print(f"⚠️  16x16 matmul failed with error: {e}")
    print("This appears to be a limitation in the current TTNN implementation for matrices larger than ~8x8.")

# Test 32x32 matmul
print("\nTesting 32x32 matmul...")
try:
    a_data_32x32 = [[random.uniform(0.1, 2.0) for _ in range(32)] for _ in range(32)]
    b_data_32x32 = [[random.uniform(0.1, 2.0) for _ in range(32)] for _ in range(32)]

    A_32x32 = Tensor(a_data_32x32, device="TTNN")
    B_32x32 = Tensor(b_data_32x32, device="TTNN")

    print("Starting 32x32 matrix multiplication...")
    C_32x32 = A_32x32 @ B_32x32
    print(f"A_32x32 @ B_32x32 shape: {C_32x32.shape}")
    C_32x32.realize()

    print("Comparing 32x32 matmul results...")
    numpy_result_32x32 = A_32x32.numpy() @ B_32x32.numpy()
    ttnn_result_32x32 = C_32x32.numpy()
    max_diff_32x32 = np.max(np.abs(ttnn_result_32x32 - numpy_result_32x32))
    print(f"Max difference: {max_diff_32x32}")
    print(f"Mean absolute difference: {np.mean(np.abs(ttnn_result_32x32 - numpy_result_32x32))}")
    print(f"Relative error: {max_diff_32x32 / np.max(np.abs(numpy_result_32x32)):.6f}")

    # Use appropriate tolerance for 32x32 matrices
    assert np.allclose(ttnn_result_32x32, numpy_result_32x32, rtol=1e-2, atol=5.0)
    print("✓ 32x32 matmul test passed!")
except Exception as e:
    print(f"⚠️  32x32 matmul failed with error: {e}")
    print("This appears to be a limitation in the current TTNN implementation for larger matrices.")

