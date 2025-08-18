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

# Stub version for testing device discovery without requiring ttnn/torch

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
    # For stub version, just copy from host buffer
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
  
  def run(self, var_vals: dict[Any, int], bufs: list[Buffer], **kwargs) -> None:
    """Execute the UOps by mapping them to TTNN operations"""
    # Stub implementation - just raise an error
    raise NotImplementedError("TTNN stub backend - install ttnn and torch for full functionality")

# ---- Main Device Class --------------------------------------------------------
class TTNNDevice(Compiled):
  """Tenstorrent TTNN device implementation (stub version)"""
  
  def __init__(self, device: str = "TTNN:0"):
    # Stub version - don't actually initialize TTNN
    print(f"TTNN stub device created: {device}")
    
    # Initialize the compiled device with our allocator
    super().__init__(device, TTNNAllocator(self), None, None, None)
  
  def synchronize(self) -> None:
    """Synchronize the device"""
    pass  # Stub implementation
  
  def get_runner(self, *ast: UOp) -> Runner:
    """Return a runner that can execute the given UOps on TTNN"""
    uops = list(ast)
    return TTNNRunner(self, uops)
  
  def __del__(self):
    """Clean up TTNN device"""
    pass  # Stub implementation