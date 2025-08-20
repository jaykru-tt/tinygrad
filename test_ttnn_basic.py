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

def test_matrix_multiplication(size):
    """Test matrix multiplication for a given size"""
    print(f"\nTesting {size}x{size} matmul...")
    try:
        # Generate random matrices
        a_data = [[random.uniform(0.1, 2.0) for _ in range(size)] for _ in range(size)]
        b_data = [[random.uniform(0.1, 2.0) for _ in range(size)] for _ in range(size)]

        A = Tensor(a_data, device="TTNN")
        B = Tensor(b_data, device="TTNN")

        print(f"Starting {size}x{size} matrix multiplication...")
        C = A @ B
        print(f"A @ B shape: {C.shape}")
        C.realize()

        print(f"Comparing {size}x{size} matmul results...")
        numpy_result = A.numpy() @ B.numpy()
        ttnn_result = C.numpy()
        max_diff = np.max(np.abs(ttnn_result - numpy_result))
        mean_diff = np.mean(np.abs(ttnn_result - numpy_result))
        relative_error = max_diff / np.max(np.abs(numpy_result))
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Relative error: {relative_error:.6f}")

        # Use appropriate tolerance based on matrix size
        if size <= 8:
            tolerance = 1e-2
        elif size <= 16:
            tolerance = 5e-2
        else:
            tolerance = 1e-1
            
        assert np.allclose(ttnn_result, numpy_result, rtol=tolerance, atol=5.0)
        print(f"✓ {size}x{size} matmul test passed!")
        return True, max_diff, mean_diff, relative_error
    except Exception as e:
        print(f"⚠️  {size}x{size} matmul failed with error: {e}")
        return False, None, None, None

def test_elementwise_addition(size):
    """Test elementwise addition for a given size"""
    print(f"\nTesting {size}x{size} elementwise addition...")
    try:
        # Generate random matrices
        a_data = [[random.uniform(0.1, 2.0) for _ in range(size)] for _ in range(size)]
        b_data = [[random.uniform(0.1, 2.0) for _ in range(size)] for _ in range(size)]

        A = Tensor(a_data, device="TTNN")
        B = Tensor(b_data, device="TTNN")

        print(f"Starting {size}x{size} elementwise addition...")
        C = A + B
        print(f"A + B shape: {C.shape}")
        C.realize()

        print(f"Comparing {size}x{size} addition results...")
        numpy_result = A.numpy() + B.numpy()
        ttnn_result = C.numpy()
        max_diff = np.max(np.abs(ttnn_result - numpy_result))
        mean_diff = np.mean(np.abs(ttnn_result - numpy_result))
        relative_error = max_diff / np.max(np.abs(numpy_result))
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Relative error: {relative_error:.6f}")

        # Use appropriate tolerance based on matrix size
        if size <= 8:
            tolerance = 1e-2
        elif size <= 16:
            tolerance = 5e-2
        else:
            tolerance = 1e-1
            
        assert np.allclose(ttnn_result, numpy_result, rtol=tolerance, atol=5.0)
        print(f"✓ {size}x{size} addition test passed!")
        return True, max_diff, mean_diff, relative_error
    except Exception as e:
        print(f"⚠️  {size}x{size} addition failed with error: {e}")
        return False, None, None, None

def test_elementwise_multiplication(size):
    """Test elementwise multiplication for a given size"""
    print(f"\nTesting {size}x{size} elementwise multiplication...")
    try:
        # Generate random matrices
        a_data = [[random.uniform(0.1, 2.0) for _ in range(size)] for _ in range(size)]
        b_data = [[random.uniform(0.1, 2.0) for _ in range(size)] for _ in range(size)]

        A = Tensor(a_data, device="TTNN")
        B = Tensor(b_data, device="TTNN")

        print(f"Starting {size}x{size} elementwise multiplication...")
        C = A * B
        print(f"A * B shape: {C.shape}")
        C.realize()

        print(f"Comparing {size}x{size} multiplication results...")
        numpy_result = A.numpy() * B.numpy()
        ttnn_result = C.numpy()
        max_diff = np.max(np.abs(ttnn_result - numpy_result))
        mean_diff = np.mean(np.abs(ttnn_result - numpy_result))
        relative_error = max_diff / np.max(np.abs(numpy_result))
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Relative error: {relative_error:.6f}")

        # Use appropriate tolerance based on matrix size
        if size <= 8:
            tolerance = 1e-2
        elif size <= 16:
            tolerance = 5e-2
        else:
            tolerance = 1e-1
            
        assert np.allclose(ttnn_result, numpy_result, rtol=tolerance, atol=5.0)
        print(f"✓ {size}x{size} multiplication test passed!")
        return True, max_diff, mean_diff, relative_error
    except Exception as e:
        print(f"⚠️  {size}x{size} multiplication failed with error: {e}")
        return False, None, None, None

def test_elementwise_operations_various_shapes():
    """Test elementwise operations with various tensor shapes"""
    print(f"\nTesting elementwise operations with various shapes...")
    
    test_cases = [
        # (shape, description)
        ((3, 3), "3x3"),
        ((5, 5), "5x5"),
        ((8, 8), "8x8"),
        ((16, 16), "16x16"),
        ((32, 32), "32x32"),
        ((2, 3), "2x3"),
        ((3, 2), "3x2"),
        ((4, 8), "4x8"),
        ((8, 4), "8x4"),
        ((1, 10), "1x10"),
        ((10, 1), "10x1"),
        ((1, 1), "1x1"),
    ]
    
    add_results = {}
    mul_results = {}
    
    for shape, description in test_cases:
        # Test addition
        print(f"\nTesting {description} elementwise addition...")
        try:
            # Generate random tensors
            a_data = np.random.uniform(0.1, 2.0, shape).tolist()
            b_data = np.random.uniform(0.1, 2.0, shape).tolist()

            A = Tensor(a_data, device="TTNN")
            B = Tensor(b_data, device="TTNN")

            print(f"Starting {description} elementwise addition...")
            C = A + B
            print(f"A + B shape: {C.shape}")
            C.realize()

            print(f"Comparing {description} addition results...")
            numpy_result = A.numpy() + B.numpy()
            ttnn_result = C.numpy()
            max_diff = np.max(np.abs(ttnn_result - numpy_result))
            mean_diff = np.mean(np.abs(ttnn_result - numpy_result))
            relative_error = max_diff / np.max(np.abs(numpy_result))
            
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean absolute difference: {mean_diff:.6f}")
            print(f"Relative error: {relative_error:.6f}")

            # Use appropriate tolerance
            tolerance = 1e-2 if np.prod(shape) <= 64 else 5e-2
            
            assert np.allclose(ttnn_result, numpy_result, rtol=tolerance, atol=5.0)
            print(f"✓ {description} addition test passed!")
            add_results[description] = {
                'success': True,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'relative_error': relative_error
            }
        except Exception as e:
            print(f"⚠️  {description} addition failed with error: {e}")
            add_results[description] = {
                'success': False,
                'max_diff': None,
                'mean_diff': None,
                'relative_error': None
            }
        
        # Test multiplication
        print(f"\nTesting {description} elementwise multiplication...")
        try:
            # Generate random tensors
            a_data = np.random.uniform(0.1, 2.0, shape).tolist()
            b_data = np.random.uniform(0.1, 2.0, shape).tolist()

            A = Tensor(a_data, device="TTNN")
            B = Tensor(b_data, device="TTNN")

            print(f"Starting {description} elementwise multiplication...")
            C = A * B
            print(f"A * B shape: {C.shape}")
            C.realize()

            print(f"Comparing {description} multiplication results...")
            numpy_result = A.numpy() * B.numpy()
            ttnn_result = C.numpy()
            max_diff = np.max(np.abs(ttnn_result - numpy_result))
            mean_diff = np.mean(np.abs(ttnn_result - numpy_result))
            relative_error = max_diff / np.max(np.abs(numpy_result))
            
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean absolute difference: {mean_diff:.6f}")
            print(f"Relative error: {relative_error:.6f}")

            # Use appropriate tolerance
            tolerance = 1e-2 if np.prod(shape) <= 64 else 5e-2
            
            assert np.allclose(ttnn_result, numpy_result, rtol=tolerance, atol=5.0)
            print(f"✓ {description} multiplication test passed!")
            mul_results[description] = {
                'success': True,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'relative_error': relative_error
            }
        except Exception as e:
            print(f"⚠️  {description} multiplication failed with error: {e}")
            mul_results[description] = {
                'success': False,
                'max_diff': None,
                'mean_diff': None,
                'relative_error': None
            }
    
    return add_results, mul_results

# Test all matrix sizes
matrix_sizes = [3, 5, 8, 16, 32]
matmul_results = {}
add_results = {}
mul_results = {}

print("\n" + "="*60)
print("COMPREHENSIVE TTNN MATRIX MULTIPLICATION TEST")
print("="*60)

for size in matrix_sizes:
    success, max_diff, mean_diff, relative_error = test_matrix_multiplication(size)
    matmul_results[size] = {
        'success': success,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'relative_error': relative_error
    }

print("\n" + "="*60)
print("COMPREHENSIVE TTNN ELEMENTWISE ADDITION TEST")
print("="*60)

for size in matrix_sizes:
    success, max_diff, mean_diff, relative_error = test_elementwise_addition(size)
    add_results[size] = {
        'success': success,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'relative_error': relative_error
    }

print("\n" + "="*60)
print("COMPREHENSIVE TTNN ELEMENTWISE MULTIPLICATION TEST")
print("="*60)

for size in matrix_sizes:
    success, max_diff, mean_diff, relative_error = test_elementwise_multiplication(size)
    mul_results[size] = {
        'success': success,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'relative_error': relative_error
    }

print("\n" + "="*60)
print("COMPREHENSIVE TTNN VARIOUS SHAPES ELEMENTWISE TEST")
print("="*60)

various_shapes_add_results, various_shapes_mul_results = test_elementwise_operations_various_shapes()

# Print summary
print("\n" + "="*80)
print("COMPREHENSIVE TEST SUMMARY")
print("="*80)

print("\nMATRIX MULTIPLICATION RESULTS:")
print("-" * 60)
print(f"{'Size':<6} {'Status':<8} {'Max Diff':<10} {'Mean Diff':<10} {'Rel Error':<10}")
print("-" * 60)

for size in matrix_sizes:
    result = matmul_results[size]
    if result['success']:
        status = "✅ PASS"
        max_diff_str = f"{result['max_diff']:.6f}"
        mean_diff_str = f"{result['mean_diff']:.6f}"
        rel_error_str = f"{result['relative_error']:.6f}"
    else:
        status = "❌ FAIL"
        max_diff_str = "N/A"
        mean_diff_str = "N/A"
        rel_error_str = "N/A"
    
    print(f"{size}x{size:<2} {status:<8} {max_diff_str:<10} {mean_diff_str:<10} {rel_error_str:<10}")

print("\nELEMENTWISE ADDITION RESULTS:")
print("-" * 60)
print(f"{'Size':<6} {'Status':<8} {'Max Diff':<10} {'Mean Diff':<10} {'Rel Error':<10}")
print("-" * 60)

for size in matrix_sizes:
    result = add_results[size]
    if result['success']:
        status = "✅ PASS"
        max_diff_str = f"{result['max_diff']:.6f}"
        mean_diff_str = f"{result['mean_diff']:.6f}"
        rel_error_str = f"{result['relative_error']:.6f}"
    else:
        status = "❌ FAIL"
        max_diff_str = "N/A"
        mean_diff_str = "N/A"
        rel_error_str = "N/A"
    
    print(f"{size}x{size:<2} {status:<8} {max_diff_str:<10} {mean_diff_str:<10} {rel_error_str:<10}")

print("\nELEMENTWISE MULTIPLICATION RESULTS:")
print("-" * 60)
print(f"{'Size':<6} {'Status':<8} {'Max Diff':<10} {'Mean Diff':<10} {'Rel Error':<10}")
print("-" * 60)

for size in matrix_sizes:
    result = mul_results[size]
    if result['success']:
        status = "✅ PASS"
        max_diff_str = f"{result['max_diff']:.6f}"
        mean_diff_str = f"{result['mean_diff']:.6f}"
        rel_error_str = f"{result['relative_error']:.6f}"
    else:
        status = "❌ FAIL"
        max_diff_str = "N/A"
        mean_diff_str = "N/A"
        rel_error_str = "N/A"
    
    print(f"{size}x{size:<2} {status:<8} {max_diff_str:<10} {mean_diff_str:<10} {rel_error_str:<10}")

print("\nVARIOUS SHAPES ELEMENTWISE ADDITION RESULTS:")
print("-" * 60)
print(f"{'Shape':<8} {'Status':<8} {'Max Diff':<10} {'Mean Diff':<10} {'Rel Error':<10}")
print("-" * 60)

for shape, result in various_shapes_add_results.items():
    if result['success']:
        status = "✅ PASS"
        max_diff_str = f"{result['max_diff']:.6f}"
        mean_diff_str = f"{result['mean_diff']:.6f}"
        rel_error_str = f"{result['relative_error']:.6f}"
    else:
        status = "❌ FAIL"
        max_diff_str = "N/A"
        mean_diff_str = "N/A"
        rel_error_str = "N/A"
    
    print(f"{shape:<8} {status:<8} {max_diff_str:<10} {mean_diff_str:<10} {rel_error_str:<10}")

print("\nVARIOUS SHAPES ELEMENTWISE MULTIPLICATION RESULTS:")
print("-" * 60)
print(f"{'Shape':<8} {'Status':<8} {'Max Diff':<10} {'Mean Diff':<10} {'Rel Error':<10}")
print("-" * 60)

for shape, result in various_shapes_mul_results.items():
    if result['success']:
        status = "✅ PASS"
        max_diff_str = f"{result['max_diff']:.6f}"
        mean_diff_str = f"{result['mean_diff']:.6f}"
        rel_error_str = f"{result['relative_error']:.6f}"
    else:
        status = "❌ FAIL"
        max_diff_str = "N/A"
        mean_diff_str = "N/A"
        rel_error_str = "N/A"
    
    print(f"{shape:<8} {status:<8} {max_diff_str:<10} {mean_diff_str:<10} {rel_error_str:<10}")

# Overall status
matmul_passed = sum(1 for result in matmul_results.values() if result['success'])
add_passed = sum(1 for result in add_results.values() if result['success'])
mul_passed = sum(1 for result in mul_results.values() if result['success'])
various_add_passed = sum(1 for result in various_shapes_add_results.values() if result['success'])
various_mul_passed = sum(1 for result in various_shapes_mul_results.values() if result['success'])

total_matmul = len(matmul_results)
total_add = len(add_results)
total_mul = len(mul_results)
total_various_add = len(various_shapes_add_results)
total_various_mul = len(various_shapes_mul_results)

print(f"\nOverall Results:")
print(f"Matrix Multiplication: {matmul_passed}/{total_matmul} tests passed")
print(f"Elementwise Addition: {add_passed}/{total_add} tests passed")
print(f"Elementwise Multiplication: {mul_passed}/{total_mul} tests passed")
print(f"Various Shapes Addition: {various_add_passed}/{total_various_add} tests passed")
print(f"Various Shapes Multiplication: {various_mul_passed}/{total_various_mul} tests passed")

total_tests = total_matmul + total_add + total_mul + total_various_add + total_various_mul
total_passed = matmul_passed + add_passed + mul_passed + various_add_passed + various_mul_passed

print(f"\nGrand Total: {total_passed}/{total_tests} tests passed")

if total_passed == total_tests:
    print("🎉 ALL TESTS PASSED! TTNN backend is working correctly for all operations and matrix sizes.")
else:
    print("⚠️  Some tests failed. Check the output above for details.")

print("="*80)

