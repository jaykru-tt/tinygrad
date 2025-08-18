from __future__ import annotations
from typing import Any, TYPE_CHECKING
import struct
from tinygrad.device import Compiled, Allocator, BufferSpec
from tinygrad.engine.realize import Runner
from tinygrad.uop.ops import UOp
from tinygrad.uop import Ops, GroupOp
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import getenv, flatten

if TYPE_CHECKING:
  from tinygrad.device import Buffer

try:
  import ttnn
  import torch
except ImportError as e:
  raise ImportError("TTNN backend requires ttnn and torch to be installed") from e

# ---- Allocator for TTNN tensors -----------------------------------------------
class TTNNAllocator(Allocator['TTNNDevice']):
  """Allocator that manages ttnn.Tensor objects on device"""

  def _alloc(self, size: int, options: BufferSpec) -> dict[str, Any]:
    # Store metadata for lazy tensor creation
    return {
      "ttnn_tensor": None,  # Will be created lazily
      "host_buffer": bytearray(size),  # Staging area for host data
      "size": size
    }
  def _copyin(self, dest: dict[str, Any], src: memoryview) -> None:
    """Copy data from host to device buffer"""
    # Stage data in host buffer, ttnn tensor will be created on first use
    dest["host_buffer"][:len(src)] = src
    # Invalidate any existing ttnn tensor
    dest["ttnn_tensor"] = None
  def _copyout(self, dest: memoryview, src: dict[str, Any]) -> None:
    """Copy data from device buffer to host"""
    if src["ttnn_tensor"] is not None:
      # Convert ttnn tensor back to torch, then to numpy, then to bytes
      ttnn_tensor = src["ttnn_tensor"]
      # Convert to row major layout for host transfer
      row_major_tensor = ttnn.to_layout(ttnn_tensor, ttnn.ROW_MAJOR_LAYOUT)
      torch_tensor = ttnn.to_torch(row_major_tensor)
      # Convert torch tensor to bytes 
      tensor_bytes = torch_tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
      dest[:] = tensor_bytes
    else:
      # No ttnn tensor exists, copy from host buffer
      dest[:] = src["host_buffer"]

  def _free(self, opaque: dict[str, Any], options: BufferSpec) -> None:
    """Free the allocated buffer"""
    opaque.clear()

# ---- TTNN Runner that interprets UOps into TTNN operations --------------------
class TTNNRunner(Runner):
  """Runner that executes UOps by mapping them to TTNN operations"""

  def __init__(self, device: TTNNDevice, uops: list[UOp]):
    self.device = device
    self.uops = uops
    # Build execution plan by analyzing the UOps
    self.plan = self._build_plan(uops)

  def _build_plan(self, uops: list[UOp]) -> list[dict[str, Any]]:
    """Build execution plan from UOps"""
    plan = []

    for i, uop in enumerate(uops):
      step = {
        "index": i,
        "op": uop.op,
        "dtype": uop.dtype,
        "src_indices": [uops.index(src) for src in uop.src],
        "arg": uop.arg
      }
      plan.append(step)

    return plan

  def _ensure_ttnn_tensor(self, buffer: Buffer, shape: tuple[int, ...], dtype: DType) -> Any:
    """Convert buffer to ttnn.Tensor if needed"""
    meta = buffer._buf  # Get allocator metadata

    if meta["ttnn_tensor"] is None:
      # Create ttnn tensor from host buffer
      host_data = meta["host_buffer"]

      # Convert bytes to appropriate format based on dtype
      if dtype == dtypes.float32:
        # Unpack float32 data
        data_list = list(struct.unpack(f'{len(host_data)//4}f', host_data))
        torch_tensor = torch.tensor(data_list, dtype=torch.float32).reshape(shape)
      elif dtype == dtypes.float16:
        # For float16, we'll use float32 and let torch handle the conversion
        data_list = list(struct.unpack(f'{len(host_data)//2}e', host_data))
        torch_tensor = torch.tensor(data_list, dtype=torch.float16).reshape(shape)
      elif dtype == dtypes.int32:
        data_list = list(struct.unpack(f'{len(host_data)//4}i', host_data))
        torch_tensor = torch.tensor(data_list, dtype=torch.int32).reshape(shape)
      else:
        # Default to float32
        data_list = list(struct.unpack(f'{len(host_data)//4}f', host_data))
        torch_tensor = torch.tensor(data_list, dtype=torch.float32).reshape(shape)

      # Convert to ttnn tensor with TILE layout and bfloat16 dtype for best performance
      ttnn_dtype = ttnn.bfloat16 if dtype in [dtypes.float32, dtypes.float16, dtypes.bfloat16] else ttnn.float32
      meta["ttnn_tensor"] = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=self.device.ttnn_device
      )

    return meta["ttnn_tensor"]

  def run(self, var_vals: dict[Any, int], bufs: list[Buffer], **kwargs) -> None:
    """Execute the UOps by mapping them to TTNN operations"""
    values = {}  # Store intermediate results by UOp index

    for step in self.plan:
      op = step["op"]
      dtype = step["dtype"]
      src_indices = step["src_indices"]
      arg = step["arg"]

      # Get source values
      src_values = [values[i] for i in src_indices if i in values]
      
      if op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}:
        # These define buffer access - we'll handle them when loading
        continue
      
      elif op is Ops.LOAD:
        # Load from buffer
        # Find the buffer this load refers to
        buf_idx = None
        for i, uop in enumerate(self.uops):
          if uop.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL} and i in src_indices:
            # This load references this buffer definition
            # Map to actual buffer (simplified for now)
            buf_idx = len([u for u in self.uops[:i] if u.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}])
            break
        
        if buf_idx is not None and buf_idx < len(bufs):
          # Get tensor shape from UOp or infer from buffer
          # This is simplified - in real implementation, shape comes from the scheduler
          shape = getattr(arg, 'shape', None) or (bufs[buf_idx].size // dtype.itemsize,)
          values[step["index"]] = self._ensure_ttnn_tensor(bufs[buf_idx], shape, dtype)
      
      elif op is Ops.CONST:
        # Create constant tensor
        const_val = arg
        # Create a scalar ttnn tensor
        torch_tensor = torch.tensor(const_val, dtype=torch.float32)
        values[step["index"]] = ttnn.from_torch(
          torch_tensor,
          dtype=ttnn.bfloat16,
          layout=ttnn.TILE_LAYOUT,
          device=self.device.ttnn_device
        )
      
      # Unary operations
      elif op is Ops.EXP2:
        values[step["index"]] = ttnn.exp2(src_values[0])
      elif op is Ops.LOG2:
        values[step["index"]] = ttnn.log2(src_values[0])
      elif op is Ops.SQRT:
        values[step["index"]] = ttnn.sqrt(src_values[0])
      elif op is Ops.RECIP:
        values[step["index"]] = ttnn.reciprocal(src_values[0])
      elif op is Ops.NEG:
        values[step["index"]] = ttnn.neg(src_values[0])
      elif op is Ops.SIN:
        values[step["index"]] = ttnn.sin(src_values[0])
      
      # Binary operations
      elif op is Ops.ADD:
        values[step["index"]] = ttnn.add(src_values[0], src_values[1])
      elif op is Ops.MUL:
        values[step["index"]] = ttnn.multiply(src_values[0], src_values[1])
      elif op is Ops.SUB:
        values[step["index"]] = ttnn.subtract(src_values[0], src_values[1])
      elif op is Ops.FDIV:
        values[step["index"]] = ttnn.div(src_values[0], src_values[1])
      elif op is Ops.MAX:
        values[step["index"]] = ttnn.maximum(src_values[0], src_values[1])
      elif op is Ops.POW:
        values[step["index"]] = ttnn.pow(src_values[0], src_values[1])
      
      # Comparison operations  
      elif op is Ops.CMPLT:
        values[step["index"]] = ttnn.lt(src_values[0], src_values[1])
      elif op is Ops.CMPEQ:
        values[step["index"]] = ttnn.eq(src_values[0], src_values[1])
      elif op is Ops.CMPNE:
        values[step["index"]] = ttnn.ne(src_values[0], src_values[1])
      
      # Ternary operations
      elif op is Ops.WHERE:
        values[step["index"]] = ttnn.where(src_values[0], src_values[1], src_values[2])
      
      # Reduction operations
      elif op is Ops.REDUCE_AXIS:
        axis = arg[0] if arg else -1
        reduce_op = arg[1] if len(arg) > 1 else Ops.ADD
        
        if reduce_op is Ops.ADD:
          values[step["index"]] = ttnn.sum(src_values[0], dim=axis)
        elif reduce_op is Ops.MAX:
          values[step["index"]] = ttnn.max(src_values[0], dim=axis)
        else:
          raise NotImplementedError(f"Reduction op {reduce_op} not implemented for TTNN")
      
      # Matrix operations
      elif op is Ops.CONTRACT:
        # This is matrix multiplication in tinygrad
        values[step["index"]] = ttnn.matmul(src_values[0], src_values[1])
      
      # Movement operations
      elif op is Ops.RESHAPE:
        new_shape = arg
        values[step["index"]] = ttnn.reshape(src_values[0], new_shape)
      elif op is Ops.PERMUTE:
        dims = arg
        values[step["index"]] = ttnn.permute(src_values[0], dims)
      
      # Store operation
      elif op is Ops.STORE:
        # Store result to output buffer
        result_tensor = src_values[0] if src_values else values.get(step["index"] - 1)
        if result_tensor is not None:
          # Find the output buffer
          out_buf_idx = None
          for i, uop in enumerate(self.uops):
            if uop.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL} and i in src_indices:
              out_buf_idx = len([u for u in self.uops[:i] if u.op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}])
              break
          
          if out_buf_idx is not None and out_buf_idx < len(bufs):
            # Store the ttnn tensor in the output buffer
            bufs[out_buf_idx]._buf["ttnn_tensor"] = result_tensor
      
      # Skip operations that don't produce values
      elif op in {Ops.BARRIER, Ops.SINK, Ops.NOOP, Ops.ENDIF, Ops.IF}:
        continue
      
      else:
        # For unimplemented operations, raise an error for now
        # In a real implementation, you might want to fall back to CPU
        raise NotImplementedError(f"TTNN backend doesn't support operation: {op}")
    
    # Synchronize device to ensure all operations complete
    self.device.synchronize()

# ---- Main Device Class --------------------------------------------------------
class TTNNDevice(Compiled):
  """Tenstorrent TTNN device implementation"""

  def __init__(self, device: str = "TTNN:0"):
    # Extract device ID from device string
    device_id = int(device.split(":")[1]) if ":" in device else 0

    # Initialize TTNN device
    self.ttnn_device = ttnn.open_device(device_id=device_id)

    # Initialize the compiled device with our allocator
    super().__init__(device, TTNNAllocator(self), None, None, None)

  def synchronize(self) -> None:
    """Synchronize the device"""
    ttnn.synchronize_device(self.ttnn_device)

  def get_runner(self, *ast: UOp) -> Runner:
    """Return a runner that can execute the given UOps on TTNN"""
    uops = list(ast)
    return TTNNRunner(self, uops)

  def __del__(self):
    """Clean up TTNN device"""
    if hasattr(self, 'ttnn_device'):
      ttnn.close_device(self.ttnn_device)
