#!/usr/bin/env python3
"""
Test script for 3D tensor matmul (batched matmul) with TTNN backend
"""

import os
import sys
import numpy as np
import random

# Add tinygrad to path
sys.path.insert(0, os.getenv("PWD") or os.getcwd())

# Test device discovery
from tinygrad.device import Device
from tinygrad.tensor import Tensor

print("Testing 3D batched matmul with TTNN...")

# Set TTNN as default device
os.environ["TTNN"] = "1"

# Test 3D batched matmul: [batch, M, K] @ [batch, K, N] -> [batch, M, N]
print("\nTesting [4, 16, 8] @ [4, 8, 12] (batched matmul)...")

batch = 4
M, K, N = 16, 8, 12

# Create random data
a_data = [[[random.uniform(-1, 1) for _ in range(K)] for _ in range(M)] for _ in range(batch)]
b_data = [[[random.uniform(-1, 1) for _ in range(N)] for _ in range(K)] for _ in range(batch)]

A = Tensor(a_data, device="TTNN")
B = Tensor(b_data, device="TTNN")
print(f"Tensor A shape: {A.shape}")
print(f"Tensor B shape: {B.shape}")

C = A @ B
print(f"A @ B shape: {C.shape}")
print(f"Buffer sizes - A: {A.nbytes()}, B: {B.nbytes()}, C: {C.nbytes()}")
C.realize()

print("Comparing batched matmul results...")
ttnn_out = C.numpy()
cpu_ref = A.numpy() @ B.numpy()
print(f"TTNN output shape: {ttnn_out.shape}")
print(f"CPU reference shape: {cpu_ref.shape}")

# TTNN may compute in reduced precision; compare with tolerance
import numpy as _np
assert _np.allclose(ttnn_out, cpu_ref, rtol=1e-1, atol=1e-1)
print("✓ 3D batched matmul test passed!")

# Test with broadcasting: [1, M, K] @ [batch, K, N] -> [batch, M, N]
# NOTE: Broadcasting support is not yet implemented in the shape inference
# print("\nTesting [1, 16, 8] @ [4, 8, 12] (broadcasting)...")
# a_data_broadcast = [[[random.uniform(-1, 1) for _ in range(K)] for _ in range(M)]]
# A_broadcast = Tensor(a_data_broadcast, device="TTNN")
# C_broadcast = A_broadcast @ B
# print(f"A shape: {A_broadcast.shape}, B shape: {B.shape}, C shape: {C_broadcast.shape}")
# C_broadcast.realize()

# ttnn_out_broadcast = C_broadcast.numpy()
# cpu_ref_broadcast = A_broadcast.numpy() @ B.numpy()
# assert _np.allclose(ttnn_out_broadcast, cpu_ref_broadcast, rtol=1e-1, atol=1e-1)
# print("✓ Broadcasting matmul test passed!")

print("\nNote: Broadcasting for batched matmul is not yet supported in shape inference")

print("\n✓ All 3D tests passed!")
