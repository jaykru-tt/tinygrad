from __future__ import annotations
from typing import Any, TYPE_CHECKING
import struct, base64, pickle
from tinygrad.device import Compiled, Allocator, BufferSpec, Compiler
from tinygrad.engine.realize import Runner
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, PatternMatcher, UPat
from tinygrad.uop import Ops, GroupOp
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import getenv, flatten

def trace(s):
  print(s)
  return s

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
    self.device = None  # Will be set by device when program is created

  def _get_ttnn_device(self):
    if self.device is not None and hasattr(self.device, 'ttnn_device'):
      return self.device.ttnn_device
    else:
      self.device = TTNNDevice()
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

    import ipdb; ipdb.set_trace()
    for i, (op, dtype, src_indices, arg) in enumerate(self.uops_data):
      # Helper to lazily materialize trivial sources (CONST scalars). SPECIAL won't be present for TTNN.
      def _maybe_materialize_src(idx: int):
        if idx in values: return values[idx]
        s_op, _, _, s_arg = self.uops_data[idx]
        if s_op is Ops.CONST and isinstance(s_arg, (int, float, bool)): return s_arg
        return None
      # Get source values (preserve arity; some ops can handle Nones for special cases)
      src_values = [_maybe_materialize_src(idx) for idx in src_indices]
      
      # Debug helpers
      before_keys = set(values.keys())
      debug_note: str | None = None
      def _fmt_val(v: Any) -> str:
        if isinstance(v, dict) and "buf_idx" in v:
          return f"Ptr(buf={v.get('buf_idx')}, extent={v.get('extent')}, gate={v.get('gate_special')})"
        shape = getattr(v, 'shape', None)
        dtype_str = getattr(v, 'dtype', None)
        if shape is not None:
          try:
            shape_tuple = tuple(shape)
          except Exception:
            shape_tuple = shape
          tn = type(v).__name__
          return f"{tn}(shape={shape_tuple}, dtype={dtype_str})"
        return repr(v)
      def _emit_debug():
        op_name = getattr(op, 'name', str(op))
        inputs_str = ", ".join(_fmt_val(v) for v in src_values) if src_values else ""
        added = set(values.keys()) - before_keys
        if i in values:
          print(f"🧩 UOp {i} {op_name}: inputs=[{inputs_str}] -> values[{i}]={_fmt_val(values[i])}")
          other = [k for k in added if k != i]
          if other:
            print(f"   ↳ also updated keys={sorted(other)}")
        else:
          tail = f" ({debug_note})" if debug_note else ""
          print(f"🧩 UOp {i} {op_name}: inputs=[{inputs_str}] -> no values write{tail}")
      
      if op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}:
        # These define buffer access - we'll handle them when loading
        pass

      elif op is Ops.RANGE:
        # Single-worker: loop index is 0
        values[i] = 0
      elif op is Ops.ENDRANGE:
        # No value to produce
        pass

      elif op is Ops.INDEX:
        # Build/extend a pointer descriptor consumed at LOAD/STORE time.
        # schema: { buf_idx, base_uop, start, extent, gate_idx }
        base_uop_idx = src_indices[0]
        base_val = values.get(base_uop_idx)
        if isinstance(base_val, dict) and "buf_idx" in base_val:
          ptr = dict(base_val)
        else:
          if base_uop_idx not in define_to_runtime_idx:
            raise NotImplementedError("INDEX base must be a pointer or DEFINE_* in TTNN backend")
          ptr = {"buf_idx": define_to_runtime_idx[base_uop_idx], "base_uop": base_uop_idx, "start": 0, "extent": None, "gate_idx": None}

        # Handle offset arg: fold const ints into start
        if len(src_indices) > 1:
          off_idx = src_indices[1]
          off_val = values.get(off_idx)
          if isinstance(off_val, int):
            ptr["start"] = ptr.get("start", 0) + off_val

        # Handle gating arg
        if len(src_indices) > 2:
          g_idx = src_indices[2]
          ptr["gate_idx"] = g_idx

        values[i] = ptr
      
      elif op is Ops.LOAD:
        # original path, locate buffer by scanning src_indices
        buf_idx = None
        for j, (buf_op, _, buf_src_indices, _) in enumerate(self.uops_data[:i]):
          if buf_op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL} and j in src_indices:
            buf_idx = len([(op, _, _, _) for op, _, _, _ in self.uops_data[:j] if op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}])
            break
        if buf_idx is not None and buf_idx < len(bufs):
          shape = (bufs[buf_idx].size // dtype.itemsize,)
          values[i] = self._ensure_ttnn_tensor(bufs[buf_idx], shape, dtype)
      
      elif op is Ops.CONST:
        # Numeric scalar constants remain scalars for scheduler math; tensors go to TTNN
        const_val = arg
        if isinstance(const_val, (int, float, bool)):
          values[i] = const_val
        else:
          torch_tensor = torch.tensor(const_val, dtype=torch.float32)
          values[i] = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
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
      elif op in {Ops.ADD, Ops.MUL, Ops.SUB, Ops.FDIV, Ops.MAX, Ops.POW, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.AND, Ops.OR, Ops.XOR, Ops.SHL, Ops.SHR, Ops.MOD}:
        a = src_values[0]
        b = src_values[1]
        if a is None and src_indices[0] in values: a = values[src_indices[0]]
        if b is None and src_indices[1] in values: b = values[src_indices[1]]
        # If both are non-TTNN (Python scalars), compute with Python
        non_ttnn = (a is not None and b is not None and not hasattr(a, 'shape') and not hasattr(b, 'shape'))
        if non_ttnn:
          if op is Ops.ADD: values[i] = a + b
          elif op is Ops.MUL: values[i] = a * b
          elif op is Ops.SUB: values[i] = a - b
          elif op is Ops.FDIV: values[i] = a / b
          elif op is Ops.MAX: values[i] = a if a >= b else b
          elif op is Ops.POW: values[i] = a ** b
          elif op is Ops.CMPLT: values[i] = int(a < b)
          elif op is Ops.CMPEQ: values[i] = int(a == b)
          elif op is Ops.CMPNE: values[i] = int(a != b)
          elif op is Ops.AND: values[i] = a & b
          elif op is Ops.OR: values[i] = a | b
          elif op is Ops.XOR: values[i] = a ^ b
          elif op is Ops.SHL: values[i] = a << b
          elif op is Ops.SHR: values[i] = a >> b
          elif op is Ops.MOD: values[i] = a % b
        else:
          assert a is not None and b is not None, f"TTNN binary op missing inputs at {i}: {self.uops_data[i]}"
          # Use TTNN for tensor ops
          if op is Ops.ADD:
            values[i] = trace(ttnn.add(a, b))
          elif op is Ops.MUL:
            values[i] = trace(ttnn.multiply(a, b))
            # detect matmul-like pattern (products from buf 1 and 2 in any order)
            ai, bi = src_indices[0], src_indices[1]
            if ai in value_source_buf and bi in value_source_buf:
              sa, sb = value_source_buf[ai], value_source_buf[bi]
              if {sa, sb} == {1, 2}: saw_mul_from_inputs = True
          elif op is Ops.SUB:
            values[i] = ttnn.subtract(a, b)
          elif op is Ops.FDIV:
            values[i] = ttnn.div(a, b)
          elif op is Ops.MAX:
            values[i] = ttnn.maximum(a, b)
          elif op is Ops.POW:
            values[i] = ttnn.pow(a, b)
          elif op is Ops.CMPLT:
            values[i] = ttnn.lt(a, b)
          elif op is Ops.CMPEQ:
            values[i] = ttnn.eq(a, b)
          elif op is Ops.CMPNE:
            values[i] = ttnn.ne(a, b)
          else:
            raise NotImplementedError(f"TTNN backend doesn't support tensor op: {op}")
      
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

      # Movement operations
      elif op is Ops.VIEW:
        base = src_values[0]
        try:
          new_shape = tuple(arg.shape)
        except Exception:
          new_shape = None
        values[i] = ttnn.reshape(base, new_shape) if new_shape is not None else base
      elif op is Ops.RESHAPE:
        new_shape = arg
        values[i] = ttnn.reshape(src_values[0], new_shape)
      elif op is Ops.PERMUTE:
        dims = arg
        values[i] = ttnn.permute(src_values[0], dims)
      
      # Store operation
      elif op is Ops.STORE:
        # STORE: write back to output buffer; if dest is pointer descriptor with slicing, update that slice
        result_tensor = src_values[1] if len(src_values) > 1 else (src_values[0] if src_values else values.get(i - 1))
        if result_tensor is not None:
          out_buf_idx = None
          for j, (buf_op, _, buf_src_indices, _) in enumerate(self.uops_data[:i]):
            if buf_op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL} and j in src_indices:
              out_buf_idx = len([(op, _, _, _) for op, _, _, _ in self.uops_data[:j] if op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}])
              break
          if out_buf_idx is not None and out_buf_idx < len(bufs):
            bufs[out_buf_idx]["ttnn_tensor"] = result_tensor
        else:
          raise RuntimeError("Bad result tensor")
      
      # Skip operations that don't produce values
      elif op in {Ops.BARRIER, Ops.SINK, Ops.NOOP, Ops.ENDIF, Ops.IF}:
        pass
      
      else:
        # For unimplemented operations, raise an error for now
        raise NotImplementedError(f"TTNN backend doesn't support operation: {op}")
      
      _emit_debug()
    
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
