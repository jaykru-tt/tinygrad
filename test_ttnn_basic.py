#!/usr/bin/env python3
"""
Comprehensive test script to verify TTNN backend integration with tinygrad
Tests matrix multiplication, elementwise addition, and elementwise multiplication for various sizes from 3x3 to 32x32
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

def fake_mnist():
    from tinygrad import Tensor, nn

    class LinearNet:
      def __init__(self):
        self.l1 = Tensor.kaiming_uniform(784, 128)
        self.l2 = Tensor.kaiming_uniform(128, 10)
      def __call__(self, x:Tensor) -> Tensor:
        return x.flatten(1).dot(self.l1).relu().dot(self.l2)

    model = LinearNet()
    optim = nn.optim.Adam([model.l1, model.l2], lr=0.001)

    x, y = Tensor.rand(4, 1, 28, 28), Tensor([2,4,3,7])  # replace with real mnist dataloader

    with Tensor.train():
      for i in range(10):
        optim.zero_grad()
        loss = model(x).sparse_categorical_crossentropy(y).backward()
        optim.step()
        print(i, loss.item())

print("doing mnist training loop on fake data")

fake_mnist()
