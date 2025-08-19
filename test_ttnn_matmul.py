#!/usr/bin/env python3
"""
Test TTNN renderer with matrix multiplication
"""

import os
import sys
import random

# Add tinygrad to path
sys.path.insert(0, os.getenv("PWD"))

def test_ttnn_matmul():
    print("Testing TTNN renderer with matrix multiplication...")
    
    try:
        from tinygrad.tensor import Tensor
        from tinygrad.device import Device
        
        # Create TTNN device
        print("Creating TTNN device...")
        ttnn_device = Device["TTNN"]
        print(f"✓ TTNN device created: {ttnn_device}")
        print(f"Device has ttnn_device: {hasattr(ttnn_device, 'ttnn_device')}")
        
        # Create test matrices
        print("\nCreating test matrices...")
        random.seed(42)
        
        # Create 4x8 matrix A
        a_data = [[random.uniform(-1, 1) for _ in range(8)] for _ in range(4)]
        # Create 8x4 matrix B  
        b_data = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(8)]
        
        print(f"Matrix A shape: (4, 8)")
        print(f"Matrix B shape: (8, 4)")
        print(f"A data[:2,:2]: {[row[:2] for row in a_data[:2]]}")
        print(f"B data[:2,:2]: {[row[:2] for row in b_data[:2]]}")
        
        # Create tensors on TTNN device
        print("\nCreating TTNN tensors...")
        a_ttnn = Tensor(a_data, device="TTNN")
        b_ttnn = Tensor(b_data, device="TTNN")
        
        print(f"Tensor A: {a_ttnn}")
        print(f"Tensor B: {b_ttnn}")
        
        # Perform matrix multiplication
        print("\nPerforming matrix multiplication on TTNN...")
        try:
            c_ttnn = a_ttnn @ b_ttnn
            print(f"Result tensor: {c_ttnn}")
            
            # Realize the tensor to trigger actual computation
            print("Realizing tensor (triggering TTNN execution)...")
            c_ttnn.realize()
            print("✓ TTNN computation completed!")
            
            # Get result data
            print("Getting result data...")
            c_result = c_ttnn.tolist()
            print(f"Result shape: (4, 4)")
            print(f"Result data[:2,:2]: {[row[:2] for row in c_result[:2]]}")
            
            # Simple CPU computation for comparison
            print("\nComparing with CPU computation...")
            # Manual matrix multiplication: C[i][j] = sum(A[i][k] * B[k][j])
            c_cpu = []
            for i in range(4):
                row = []
                for j in range(4):
                    sum_val = 0.0
                    for k in range(8):
                        sum_val += a_data[i][k] * b_data[k][j]
                    row.append(sum_val)
                c_cpu.append(row)
            
            print(f"CPU result[:2,:2]: {[row[:2] for row in c_cpu[:2]]}")
            
            # Check if results are close  
            max_diff = 0.0
            for i in range(4):
                for j in range(4):
                    diff = abs(c_result[i][j] - c_cpu[i][j])
                    if diff > max_diff:
                        max_diff = diff
            
            print(f"Maximum difference: {max_diff}")
            
            if max_diff < 1e-3:  # Allow for some floating point differences
                print("✓ Results match CPU computation!")
                return True
            else:
                print(f"✗ Results differ significantly (max diff: {max_diff})")
                return False
                
        except Exception as e:
            print(f"✗ Error during TTNN computation: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure ttnn and torch are installed")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensor_verification():
    """Verify that we're actually using TTNN tensors"""
    print("\n" + "="*50)
    print("VERIFYING TTNN TENSOR USAGE")
    print("="*50)
    
    try:
        from tinygrad.tensor import Tensor
        from tinygrad.device import Device
        
        # Create small test tensor
        print("Creating small test tensor on TTNN...")
        test_data = [[1.0, 2.0], [3.0, 4.0]]
        tensor = Tensor(test_data, device="TTNN")
        
        # Access the underlying buffer to check TTNN tensor
        print("Checking underlying UOp and buffer...")
        print(f"Tensor device: {tensor.device}")
        print(f"Tensor UOp: {tensor.uop}")
        
        # Force tensor creation by doing a simple operation
        print("\nForcing tensor creation with simple add operation...")
        result = tensor + 1.0
        print(f"Result tensor device: {result.device}")
        print("Realizing result tensor...")
        result.realize()
        print("✓ Tensor realized successfully!")
        
        # Try to access buffer information if possible
        try:
            # Check if we can access internal buffer info
            from tinygrad.device import Device
            ttnn_device = Device["TTNN"]
            print(f"TTNN device type: {type(ttnn_device)}")
            print(f"TTNN device has ttnn_device: {hasattr(ttnn_device, 'ttnn_device')}")
            if hasattr(ttnn_device, 'ttnn_device'):
                print(f"TTNN device initialized: {ttnn_device.ttnn_device is not None}")
            return True
        except Exception as e:
            print(f"Could not access device info: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error during tensor verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TTNN Matrix Multiplication Test")
    print("=" * 40)
    
    # Test basic matmul
    matmul_success = test_ttnn_matmul()
    
    # Test tensor verification
    tensor_success = test_tensor_verification()
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Matrix multiplication test: {'✓ PASSED' if matmul_success else '✗ FAILED'}")
    print(f"TTNN tensor verification: {'✓ PASSED' if tensor_success else '✗ FAILED'}")
    
    if matmul_success and tensor_success:
        print("\n🎉 All tests passed! TTNN renderer is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)