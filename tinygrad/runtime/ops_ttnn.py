from __future__ import annotations
from typing import Any, TYPE_CHECKING
import struct, base64, pickle
from tinygrad.device import Compiled, Allocator, BufferSpec, Compiler
from tinygrad.engine.realize import Runner
from tinygrad.renderer import Renderer
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

# ---- TTNN Renderer that base64 decodes and interprets UOps ----------------
class TTNNRenderer(Renderer):
  """Renderer that base64 decodes UOps for direct interpretation"""
  device: str = "TTNN"
  suffix: str = ""
  
  def render(self, uops: list[UOp]) -> str:
    # Convert UOps to serializable format (opposite of PythonRenderer)
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src], u.arg) for u in uops]
    # Base64 encode like PythonRenderer - this will be decoded by compiler
    return base64.b64encode(pickle.dumps(lops)).decode()

# ---- TTNN Compiler that base64 decodes for interpretation ------------------
class TTNNCompiler(Compiler):
  """Compiler that decodes base64 encoded UOps for direct interpretation"""
  
  def compile(self, src: str) -> bytes:
    # Base64 decode the rendered UOps (opposite parity to renderer encoding)
    return base64.b64decode(src)

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

# ---- TTNN Program that holds compiled UOps ---------------------------------
class TTNNProgram:
  """Program that holds the decoded UOps for interpretation"""
  
  def __init__(self, name: str, lib: bytes):
    self.name = name
    # Decode the base64 encoded UOps (reverse of compiler encoding)
    self.uops_data = pickle.loads(lib)  # List of (op, dtype, src_indices, arg) tuples
    self.device = TTNNDevice()  # Will be set by device when program is created
    self._ttnn_device_handle = None  # lazy fallback if device is not set

  def _get_ttnn_device(self):
    return self.device.ttnn_device
  
  def _ensure_ttnn_tensor(self, buffer, shape: tuple[int, ...], dtype: DType) -> Any:
    """Convert buffer (allocator meta dict or Buffer) to ttnn.Tensor if needed"""
    meta = buffer if isinstance(buffer, dict) else buffer._buf

    if meta["ttnn_tensor"] is None:
      print(f"🔧 Creating TTNN tensor: shape={shape}, dtype={dtype}")
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

      print(f"🔧 Torch tensor shape: {torch_tensor.shape}, dtype: {torch_tensor.dtype}")
      
      # Convert to ttnn tensor; prefer float32 ROW_MAJOR for simplicity/alignment
      ttnn_dtype = ttnn.float32
      print(f"🔧 Converting to TTNN tensor with dtype: {ttnn_dtype}, layout: ROW_MAJOR_LAYOUT")
      
      meta["ttnn_tensor"] = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=self._get_ttnn_device()
      )
      
      print(f"✅ TTNN tensor created: {type(meta['ttnn_tensor'])}")
    else:
      print(f"♻️  Reusing existing TTNN tensor")

    return meta["ttnn_tensor"]
  
  def __call__(self, *bufs, global_size: tuple[int,int,int]=(1,1,1), local_size: tuple[int,int,int]=(1,1,1), vals: tuple[int, ...]=(), wait=False):
    """Execute the UOps by interpreting them directly with TTNN operations"""
    values = {}  # Store intermediate results by UOp index

    # Map DEFINE_GLOBAL uops to runtime buffer indices and base dtypes
    define_global_order: list[int] = [idx for idx,(oop,_,_,_) in enumerate(self.uops_data) if oop is Ops.DEFINE_GLOBAL]
    define_to_runtime_idx: dict[int,int] = {uop_idx: runtime_idx for runtime_idx, uop_idx in enumerate(define_global_order)}
    define_base_dtype: dict[int, DType] = {}
    for uop_idx in define_global_order:
      _, dt, _, _ = self.uops_data[uop_idx]
      base_dt = dt.base if hasattr(dt, 'base') else dt
      define_base_dtype[uop_idx] = base_dt.scalar()

    # Helpers for occasional interop when fixing up indexing/gating
    def ttnn_to_torch_row_major(t):
      rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
      return ttnn.to_torch(rm)
    def torch_to_ttnn_tile(t: torch.Tensor, force_float: bool=False):
      ttnn_dtype = ttnn.float32
      return ttnn.from_torch(t, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._get_ttnn_device())

    for i, (op, dtype, src_indices, arg) in enumerate(self.uops_data):
      # Get source values
      src_values = [values[idx] for idx in src_indices if idx in values]
      
      if op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}:
        # These define buffer access - we'll handle them when loading
        continue
      
      elif op is Ops.SPECIAL:
        # NOOP: pretend we have one massive worker, interpret at INDEX/STORE time
        continue
      
      elif op is Ops.INDEX:
        # Pointer descriptor: { buf_idx, extent (if offset is SPECIAL), offsets (optional), base_uop, gate_special (bool) }
        base_uop_idx = src_indices[0]
        if base_uop_idx not in define_to_runtime_idx:
          raise NotImplementedError("INDEX base must be DEFINE_GLOBAL in TTNN backend")
        # derive extent if offset is SPECIAL
        extent = None
        if len(src_indices) > 1:
          off_idx = src_indices[1]
          off_op, _, _, off_arg = self.uops_data[off_idx]
          if off_op is Ops.SPECIAL:
            name, ext = off_arg
            if isinstance(ext, int): extent = ext
            else:
              axis = int(name[-1]) if name[-1].isdigit() else 0
              extent = (local_size if name[0] == 'l' else global_size)[axis]
        gate_special = False
        if len(src_indices) > 2:
          g_idx = src_indices[2]
          g_op, _, _, g_arg = self.uops_data[g_idx]
          if g_op is Ops.SPECIAL: gate_special = True
        values[i] = {"buf_idx": define_to_runtime_idx[base_uop_idx], "extent": extent, "base_uop": base_uop_idx, "gate_special": gate_special}
      
      elif op is Ops.LOAD:
        # If source is an INDEX pointer descriptor, return the whole buffer tensor (shaped by extent if provided)
        ptr = src_values[0] if src_values else None
        if isinstance(ptr, dict) and "buf_idx" in ptr:
          buf_idx = ptr["buf_idx"]
          base_len = bufs[buf_idx]["size"] // dtype.itemsize
          shape = (ptr["extent"] if ptr.get("extent") is not None else base_len,)
          t = self._ensure_ttnn_tensor(bufs[buf_idx], (base_len,), dtype)
          # reshape if needed (pretend single massive worker over extent)
          if shape != (base_len,):
            torch_view = ttnn_to_torch_row_major(t).reshape(base_len)[:shape[0]]
            values[i] = torch_to_ttnn_tile(torch_view)
          else:
            values[i] = t
        else:
          # Fallback: original path, locate buffer by scanning src_indices
          buf_idx = None
          for j, (buf_op, _, buf_src_indices, _) in enumerate(self.uops_data[:i]):
            if buf_op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL} and j in src_indices:
              buf_idx = len([(op, _, _, _) for op, _, _, _ in self.uops_data[:j] if op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}])
              break
          if buf_idx is not None and buf_idx < len(bufs):
            shape = (bufs[buf_idx].size // dtype.itemsize,)
            values[i] = self._ensure_ttnn_tensor(bufs[buf_idx], shape, dtype)
      
      elif op is Ops.CONST:
        # Create constant tensor
        const_val = arg
        torch_tensor = torch.tensor(const_val, dtype=torch.float32)
        values[i] = ttnn.from_torch(
          torch_tensor,
          dtype=ttnn.bfloat16,
          layout=ttnn.TILE_LAYOUT,
          device=self._get_ttnn_device()
        )
      
      # Unary operations
      elif op is Ops.EXP2:
        values[i] = ttnn.exp2(src_values[0])
      elif op is Ops.LOG2:
        values[i] = ttnn.log2(src_values[0])
      elif op is Ops.SQRT:
        values[i] = ttnn.sqrt(src_values[0])
      elif op is Ops.RECIP:
        values[i] = ttnn.reciprocal(src_values[0])
      elif op is Ops.NEG:
        values[i] = ttnn.neg(src_values[0])
      elif op is Ops.SIN:
        values[i] = ttnn.sin(src_values[0])
      
      # Binary operations
      elif op is Ops.ADD:
        print(f"🔢 TTNN ADD: {type(src_values[0])} + {type(src_values[1])}")
        values[i] = ttnn.add(src_values[0], src_values[1])
      elif op is Ops.MUL:
        print(f"🔢 TTNN MUL: {type(src_values[0])} * {type(src_values[1])}")
        values[i] = ttnn.multiply(src_values[0], src_values[1])
      elif op is Ops.SUB:
        print(f"🔢 TTNN SUB: {type(src_values[0])} - {type(src_values[1])}")
        values[i] = ttnn.subtract(src_values[0], src_values[1])
      elif op is Ops.FDIV:
        print(f"🔢 TTNN DIV: {type(src_values[0])} / {type(src_values[1])}")
        values[i] = ttnn.div(src_values[0], src_values[1])
      elif op is Ops.MAX:
        print(f"🔢 TTNN MAX: max({type(src_values[0])}, {type(src_values[1])})")
        values[i] = ttnn.maximum(src_values[0], src_values[1])
      elif op is Ops.POW:
        print(f"🔢 TTNN POW: {type(src_values[0])} ** {type(src_values[1])}")
        values[i] = ttnn.pow(src_values[0], src_values[1])
      
      # Comparison operations  
      elif op is Ops.CMPLT:
        values[i] = ttnn.lt(src_values[0], src_values[1])
      elif op is Ops.CMPEQ:
        values[i] = ttnn.eq(src_values[0], src_values[1])
      elif op is Ops.CMPNE:
        values[i] = ttnn.ne(src_values[0], src_values[1])
      
      # Ternary operations
      elif op is Ops.WHERE:
        values[i] = ttnn.where(src_values[0], src_values[1], src_values[2])
      
      # Reduction operations
      elif op is Ops.REDUCE_AXIS:
        axis = arg[0] if arg else -1
        reduce_op = arg[1] if len(arg) > 1 else Ops.ADD
        
        if reduce_op is Ops.ADD:
          values[i] = ttnn.sum(src_values[0], dim=axis)
        elif reduce_op is Ops.MAX:
          values[i] = ttnn.max(src_values[0], dim=axis)
        else:
          raise NotImplementedError(f"Reduction op {reduce_op} not implemented for TTNN")
      
      # Matrix operations
      elif op is Ops.CONTRACT:
        # This is matrix multiplication in tinygrad
        print(f"🔶 TTNN MATMUL: {type(src_values[0])} @ {type(src_values[1])}")
        print(f"🔶 Input shapes: {src_values[0].shape if hasattr(src_values[0], 'shape') else 'unknown'} @ {src_values[1].shape if hasattr(src_values[1], 'shape') else 'unknown'}")
        values[i] = ttnn.matmul(src_values[0], src_values[1])
        print(f"🔶 MATMUL result: {type(values[i])}")
      
      # Movement operations
      elif op is Ops.RESHAPE:
        new_shape = arg
        values[i] = ttnn.reshape(src_values[0], new_shape)
      elif op is Ops.PERMUTE:
        dims = arg
        values[i] = ttnn.permute(src_values[0], dims)
      
      # Store operation
      elif op is Ops.STORE:
        # STORE: write back to output buffer; if dest is pointer descriptor with extent, update that slice
        result_tensor = src_values[1] if len(src_values) > 1 else (src_values[0] if src_values else values.get(i - 1))
        dest_ptr = src_values[0] if src_values else None
        if isinstance(dest_ptr, dict) and "buf_idx" in dest_ptr and result_tensor is not None:
          out_buf_idx = dest_ptr["buf_idx"]
          out_define = dest_ptr.get("base_uop")
          out_dtype = define_base_dtype.get(out_define, dtypes.float32)
          base_len = bufs[out_buf_idx]["size"] // out_dtype.itemsize
          # materialize output current tensor and payload to torch
          # NOTE: we intentionally use torch for the simple masked blend; compute remains on TTNN for math ops
          # current out
          # we don't know dtype here; load as float32 bytes and blend
          if bufs[out_buf_idx].get("ttnn_tensor") is None:
            # ensure one exists
            bufs[out_buf_idx]["ttnn_tensor"] = torch_to_ttnn_tile(torch.zeros((base_len,), dtype=torch.float32), force_float=True)
          out_t = ttnn_to_torch_row_major(bufs[out_buf_idx]["ttnn_tensor"]).reshape(-1)
          pay_t = result_tensor if isinstance(result_tensor, torch.Tensor) else ttnn_to_torch_row_major(result_tensor).reshape(-1)
          # determine extent and mask
          extent = dest_ptr.get("extent", len(pay_t))
          mask = None
          # gate from SPECIAL means mask out index 0
          if dest_ptr.get("gate_special", False):
            mask = torch.ones(extent, dtype=torch.bool); mask[0] = False
          elif len(src_indices) > 2 and src_indices[2] in values:
            gval = values[src_indices[2]]
            g_t = gval if isinstance(gval, torch.Tensor) else ttnn_to_torch_row_major(gval).reshape(-1)
            mask = (g_t.to(torch.long) != 0)
          if mask is None:
            mask = torch.ones(extent, dtype=torch.bool)
          # blend into beginning slice of out
          out_t = out_t.clone()
          out_t[:extent][mask] = pay_t.reshape(-1)[:extent][mask]
          bufs[out_buf_idx]["ttnn_tensor"] = torch_to_ttnn_tile(out_t, force_float=True)
        else:
          # Fallback: original simple whole-buffer assignment
          if result_tensor is not None:
            out_buf_idx = None
            for j, (buf_op, _, buf_src_indices, _) in enumerate(self.uops_data[:i]):
              if buf_op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL} and j in src_indices:
                out_buf_idx = len([(op, _, _, _) for op, _, _, _ in self.uops_data[:j] if op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}])
                break
            if out_buf_idx is not None and out_buf_idx < len(bufs):
              bufs[out_buf_idx]["ttnn_tensor"] = result_tensor
      
      # Skip operations that don't produce values
      elif op in {Ops.BARRIER, Ops.SINK, Ops.NOOP, Ops.ENDIF, Ops.IF}:
        continue
      
      else:
        # For unimplemented operations, raise an error for now
        raise NotImplementedError(f"TTNN backend doesn't support operation: {op}")
    
    # Synchronize device to ensure all operations complete
    if self.device:
      self.device.synchronize()

# ---- Main Device Class --------------------------------------------------------
class TTNNDevice(Compiled):
  """Tenstorrent TTNN device implementation"""

  def __init__(self, device: str = "TTNN:0"):
    # Extract device ID from device string
    device_id = int(device.split(":")[1]) if ":" in device else 0

    # Initialize TTNN device
    self.ttnn_device = ttnn.open_device(device_id=device_id)
    print(f"TTNN device created: {self.ttnn_device}")

    # Initialize the compiled device with renderer, compiler, and program
    super().__init__(device, TTNNAllocator(self), TTNNRenderer(), TTNNCompiler(), TTNNProgram)

  def synchronize(self) -> None:
    """Synchronize the device"""
    ttnn.synchronize_device(self.ttnn_device)

  def runtime(self, name: str, lib: bytes):
    """Create a program and set up the device reference"""
    program = TTNNProgram(name, lib)
    program.device = self  # Set device reference for tensor creation
    return program

  def __del__(self):
    """Clean up TTNN device"""
    if hasattr(self, 'ttnn_device'):
      ttnn.close_device(self.ttnn_device)
