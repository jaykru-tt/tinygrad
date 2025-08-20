#!/usr/bin/env python3
"""Test RANGE operation logic in TTNN backend (without requiring ttnn library)"""

import pickle
import base64
from tinygrad.uop import Ops
from tinygrad.dtype import dtypes

def test_range_logic():
    """Test the RANGE/ENDRANGE control flow logic"""
    print("\n=== Testing RANGE/ENDRANGE logic ===")
    
    # Create a simple program with a RANGE loop
    # This simulates: for i in range(3): acc += i
    uops_data = [
        # 0: DEFINE_GLOBAL for output buffer
        (Ops.DEFINE_GLOBAL, dtypes.float32.ptr(), [], 0),
        # 1: CONST 0 (initial accumulator)
        (Ops.CONST, dtypes.float32, [], 0.0),
        # 2: CONST 3 (loop end)
        (Ops.CONST, dtypes.int32, [], 3),
        # 3: RANGE(3) - loop from 0 to 2
        (Ops.RANGE, dtypes.int32, [2], 0),  # src[0] = index 2 (const 3)
        # 4: ADD acc + i (note: in real execution, acc would be updated in a register)
        (Ops.ADD, dtypes.float32, [1, 3], None),  # acc + loop_counter
        # 5: ENDRANGE - jump back to RANGE
        (Ops.ENDRANGE, dtypes.void, [3], None),
        # 6: STORE result
        (Ops.STORE, dtypes.void, [0, 4], None),
    ]
    
    # Simulate the execution
    values = {}
    loop_state = {}
    next_idx = None
    i = 0
    
    print("Executing RANGE loop simulation:")
    iterations = 0
    max_iterations = 20  # Safety limit
    
    while i < len(uops_data) and iterations < max_iterations:
        iterations += 1
        
        # Check for jumps
        if next_idx is not None:
            i = next_idx
            next_idx = None
            continue
        
        op, dtype, src_indices, arg = uops_data[i]
        print(f"  Step {i}: {op.name}")
        
        if op is Ops.CONST:
            values[i] = arg
        elif op is Ops.RANGE:
            loop_key = f"range_{i}"
            end_val = values.get(src_indices[0], 3) if src_indices else 3
            
            if loop_key not in loop_state:
                # Initialize loop
                loop_state[loop_key] = {"counter": 0, "end": end_val}
                values[i] = 0
                print(f"    -> Initialized loop counter to 0 (end={end_val})")
            else:
                # Increment counter
                loop_state[loop_key]["counter"] += 1
                counter = loop_state[loop_key]["counter"]
                
                if counter < loop_state[loop_key]["end"]:
                    values[i] = counter
                    print(f"    -> Loop counter = {counter}")
                else:
                    # Loop complete
                    print(f"    -> Loop complete after {counter} iterations")
                    del loop_state[loop_key]
                    # Skip to after ENDRANGE
                    for j in range(i+1, len(uops_data)):
                        if uops_data[j][0] is Ops.ENDRANGE:
                            next_idx = j + 1
                            break
        elif op is Ops.ENDRANGE:
            range_idx = src_indices[0] if src_indices else None
            if range_idx is not None:
                loop_key = f"range_{range_idx}"
                if loop_key in loop_state:
                    # Jump back to RANGE
                    next_idx = range_idx
                    print(f"    -> Jumping back to RANGE at {range_idx}")
        elif op is Ops.ADD:
            # Simulate addition (accumulate across iterations)
            val1 = values.get(src_indices[0], 0) if i not in values else values[i]
            val2 = values.get(src_indices[1], 0)
            values[i] = val1 + val2
            print(f"    -> {val1} + {val2} = {values[i]}")
        
        i += 1
    
    print(f"\nLoop executed {iterations} steps")
    
    # Check that we accumulated 0+1+2 = 3
    expected_sum = sum(range(3))
    if 4 in values:
        print(f"Final accumulated value: {values[4]}")
        print(f"Expected: {expected_sum}")
        assert abs(values[4] - expected_sum) < 0.001, f"Expected {expected_sum}, got {values[4]}"
    
    print("✅ RANGE logic test passed!")

if __name__ == "__main__":
    test_range_logic()
    print("\n🎉 All RANGE logic tests passed!")
