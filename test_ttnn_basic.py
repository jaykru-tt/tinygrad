#!/usr/bin/env python3
"""
Basic test script to verify TTNN backend integration with tinygrad
"""

import os
import sys
import numpy as np

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

# Test tensor creation and basic operations
print("Creating test tensors...")
a = Tensor([1.0, 2.0, 3.0], device="TTNN")
b = Tensor([4.0, 5.0, 6.0], device="TTNN")

print(f"Tensor a: {a}")
print(f"Tensor b: {b}")

# Test addition
print("Testing addition...")
c = a + b
print(f"a + b = {c}")

# Try to realize the tensor (this will trigger the runner)
print("Attempting to realize tensor...")
c.realize()
print("✓ Tensor realized successfully!")

# check the result
print(f"Result: {c.numpy()}")
assert (c.numpy() == np.array([5.0, 7.0, 9.0])).all()

print("✓ All tests passed!")      
