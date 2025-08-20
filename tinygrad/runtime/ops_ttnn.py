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
  return s

if TYPE_CHECKING:
  from tinygrad.device import Buffer

try:
  import os
  import ttnn
  import torch
except ImportError as e:
  raise ImportError("TTNN backend requires ttnn and torch to be installed") from e

CACHED_DEVICE = None



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
      torch_tensor = ttnn.to_torch(row_major_tensor).detach().cpu().contiguous()
      arr = torch_tensor.numpy()
      expected_nbytes = len(dest)
      if arr.nbytes != expected_nbytes:
        numel = arr.size
        # try to coerce dtype to match expected buffer size
        if expected_nbytes == numel * 4:
          torch_tensor = torch_tensor.to(torch.float32 if torch.is_floating_point(torch_tensor) else torch.int32)
          arr = torch_tensor.numpy()
        elif expected_nbytes == numel * 2:
          # prefer float16 fallback
          torch_tensor = torch_tensor.to(torch.float16)
          arr = torch_tensor.numpy()
        elif expected_nbytes == numel:
          torch_tensor = torch_tensor.to(torch.uint8)
          arr = torch_tensor.numpy()
      out_bytes = arr.tobytes()
      if len(out_bytes) != len(dest):
        if len(out_bytes) > len(dest):
          out_bytes = out_bytes[:len(dest)]
        else:
          out_bytes = out_bytes + b'\x00'*(len(dest)-len(out_bytes))
      dest[:] = out_bytes
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
      host_data = meta["host_buffer"]

      # Map tinygrad dtype to torch and ttnn dtypes
      if dtype == dtypes.float32:
        unpack_fmt, torch_dt, ttnn_dt, bytes_per = 'f', torch.float32, ttnn.float32, 4
      elif dtype == dtypes.float16:
        unpack_fmt, torch_dt, ttnn_dt, bytes_per = 'e', torch.float16, ttnn.float16, 2
      elif dtype == dtypes.int32:
        unpack_fmt, torch_dt, ttnn_dt, bytes_per = 'i', torch.int32, ttnn.int32, 4
      elif dtype == dtypes.uchar:
        unpack_fmt, torch_dt, ttnn_dt, bytes_per = 'B', torch.uint8, ttnn.uint8, 1
      else:
        unpack_fmt, torch_dt, ttnn_dt, bytes_per = 'f', torch.float32, ttnn.float32, 4

      count = len(host_data) // bytes_per
      data_list = list(struct.unpack(f'{count}{unpack_fmt}', host_data[:count*bytes_per]))
      torch_tensor = torch.tensor(data_list, dtype=torch_dt).reshape(shape)

      meta["ttnn_tensor"] = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dt,
        layout=ttnn.TILE_LAYOUT,
        device=self._get_ttnn_device()
      )

    return meta["ttnn_tensor"]
  
  # torch tensors are not used for compute; only TTNN tensors are produced
  
  def __call__(self, *bufs, global_size: tuple[int,int,int]=(1,1,1), local_size: tuple[int,int,int]=(1,1,1), vals: tuple[int, ...]=(), wait=False):
    """Execute the UOps by interpreting them directly with TTNN operations"""
    values = {}  # Store intermediate results by UOp index

    # Precompute supported op maps based on available ttnn symbols (lowercase)
    def _mk_map(pairs):
      out = {}
      for op_const, ttnn_name in pairs:
        if hasattr(ttnn, ttnn_name):
          out[op_const] = getattr(ttnn, ttnn_name)
      return out
    unary_map = _mk_map([
      (Ops.EXP2, 'exp2'),
      (Ops.LOG2, 'log2'),
      (Ops.SQRT, 'sqrt'),
      (Ops.NEG, 'neg'),
      (Ops.SIN, 'sin'),
    ])
    binary_map = _mk_map([
      (Ops.ADD, 'add'),
      (Ops.MUL, 'mul'),
      (Ops.SUB, 'sub'),
      (Ops.FDIV, 'divide'),
      (Ops.MAX, 'max'),
      (Ops.POW, 'pow'),
    ])

    # Map DEFINE_GLOBAL uops to runtime buffer indices and base dtypes
    define_global_order: list[int] = [idx for idx,(oop,_,_,_) in enumerate(self.uops_data) if oop is Ops.DEFINE_GLOBAL]
    define_to_runtime_idx: dict[int,int] = {uop_idx: runtime_idx for runtime_idx, uop_idx in enumerate(define_global_order)}
    define_base_dtype: dict[int, DType] = {}
    for uop_idx in define_global_order:
      _, dt, _, _ = self.uops_data[uop_idx]
      base_dt = dt.base if hasattr(dt, 'base') else dt
      define_base_dtype[uop_idx] = base_dt.scalar()

    # Helpers to resolve DEFINE_* buffer indices even when wrapped in VIEWs
    def _resolve_define_uop_idx(uop_idx: int) -> int | None:
      seen = set()
      while uop_idx is not None and uop_idx not in seen:
        seen.add(uop_idx)
        opx, _dtx, srcx, _argx = self.uops_data[uop_idx]
        if opx in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}: return uop_idx
        if opx is Ops.VIEW and srcx:
          uop_idx = srcx[0]
          continue
        return None
      return None

    def _runtime_buf_index_from_any(src_uop_idx: int) -> int | None:
      def_uop = _resolve_define_uop_idx(src_uop_idx)
      if def_uop is None: return None
      return define_to_runtime_idx.get(def_uop)

    def _shape_from_view_uop(src_uop_idx: int) -> tuple[int, ...] | None:
      opx, _dtx, srcx, argx = self.uops_data[src_uop_idx]
      if opx is Ops.VIEW and hasattr(argx, 'shape'):
        try:
          return tuple(argx.shape)
        except Exception:
          return None
      return None

    def _shape_strides_from_view_uop(src_uop_idx: int) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
      # best-effort extraction of (shape, strides) from ShapeTracker View
      opx, _dtx, srcx, argx = self.uops_data[src_uop_idx]
      if opx is not Ops.VIEW: return None
      # common API: argx.views is a tuple of View objects
      try:
        views = getattr(argx, 'views', None)
        if isinstance(views, (list, tuple)) and len(views) > 0:
          v = views[-1]
          shp = tuple(getattr(v, 'shape'))
          std = tuple(getattr(v, 'strides'))
          if isinstance(shp, tuple) and isinstance(std, tuple):
            return shp, std
      except Exception:
        return None
      return None
    print(f"uops_data: {list(enumerate(self.uops_data))}")
    for i, (op, dtype, src_indices, arg) in enumerate(self.uops_data):
      # Helper to lazily materialize trivial sources (CONST scalars). SPECIAL won't be present for TTNN.
      def _maybe_materialize_src(idx: int):
        if idx in values: return values[idx]
        s_op, _, _, s_arg = self.uops_data[idx]
        if s_op is Ops.CONST and isinstance(s_arg, (int, float, bool)): return s_arg
        return None
      # Get source values (preserve arity; some ops can handle Nones for special cases)
      src_values = [_maybe_materialize_src(idx) for idx in src_indices]
      
      # (optional) debug removed for cleanliness
      
      if op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL}:
        # These define buffer access - we'll handle them when loading
        continue

      if op is Ops.RANGE:
        # Single-worker: loop index is 0
        values[i] = 0
        continue
      if op is Ops.ENDRANGE:
        # No value to produce
        continue

      if op is Ops.INDEX:
        # Not used in the simple eltwise-add test; implement later as needed
        continue
      
      if op is Ops.LOAD:
        # Resolve runtime buffer index even when wrapped in VIEWs
        buf_idx = None
        shape_hint = None
        shape_strides = None
        for src_uop_idx in src_indices:
          # prefer shapes from VIEW nodes if present
          shape_hint = _shape_from_view_uop(src_uop_idx) or shape_hint
          shape_strides = _shape_strides_from_view_uop(src_uop_idx) or shape_strides
          maybe_idx = _runtime_buf_index_from_any(src_uop_idx)
          if maybe_idx is not None:
            buf_idx = maybe_idx
            break
        if buf_idx is None or buf_idx >= len(bufs):
          raise RuntimeError("TTNN LOAD: unable to resolve source buffer")
        # Choose shape hint only if it matches buffer size; else fallback to flat
        buf_obj = bufs[buf_idx]
        try:
          byte_size = buf_obj["size"] if isinstance(buf_obj, dict) else buf_obj.size
        except Exception:
          # last resort: try to get underlying meta
          meta = getattr(buf_obj, "_buf", None)
          byte_size = meta.get("size", 0) if isinstance(meta, dict) else 0
        numel = byte_size // dtype.itemsize
        # Build TTNN tensor in two steps: load flat, then apply reshape/permute to realize VIEW
        base_tensor = self._ensure_ttnn_tensor(bufs[buf_idx], (numel,), dtype)
        final_tensor = base_tensor
        if shape_strides is not None:
          shp, std = shape_strides
          nz = [(d, s) for d, s in enumerate(std) if s != 0]
          if len(nz) == 0:
            # all broadcast, just make a scalar then expand (will broadcast later in ops)
            final_tensor = ttnn.reshape(base_tensor, (1,)*len(shp))
          elif len(nz) == len(shp):
            # no broadcasting, reshape directly to hint if valid
            if shape_hint is not None:
              final_tensor = ttnn.reshape(base_tensor, tuple(shape_hint))
            else:
              final_tensor = base_tensor
          elif len(nz) == 2:
            # common matmul lowering pattern: two data dims and one broadcast dim
            # order non-zero dims by stride (major -> minor)
            nz_sorted = sorted(nz, key=lambda x: x[1], reverse=True)
            base2d_shape = (int(shp[nz_sorted[0][0]]), int(shp[nz_sorted[1][0]]))
            # reshape flat to base2d
            tmp = ttnn.reshape(base_tensor, base2d_shape)
            # extend to N dims by appending ones
            tmp = ttnn.reshape(tmp, base2d_shape + (1,)*(len(shp)-2))
            # build perm to place axes at correct dims
            # mapping from view dim -> axis index in base2d
            axis_map = {nz_sorted[0][0]: 0, nz_sorted[1][0]: 1}
            # target positions for axis 0 and 1
            pos0 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 0)
            pos1 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 1)
            # current axes are [0,1,2,3,...] where 2.. are singleton dims
            perm = [None]*len(shp)
            perm[pos0] = 0
            perm[pos1] = 1
            # fill remaining with the singleton axes in order
            single_axes = [ax for ax in range(2, len(shp))]
            for idx in range(len(shp)):
              if perm[idx] is None:
                perm[idx] = single_axes.pop(0)
            final_tensor = ttnn.permute(tmp, tuple(perm))
          else:
            # fallback: try to reshape to shape hint
            final_tensor = ttnn.reshape(base_tensor, tuple(shape_hint) if shape_hint is not None else (numel,))
        else:
          # no view info, reshape to hint if valid
          final_tensor = ttnn.reshape(base_tensor, tuple(shape_hint)) if shape_hint is not None else base_tensor
        values[i] = final_tensor
        continue
      
      elif op is Ops.CONST:
        # Numeric scalar constants remain scalars for scheduler math; tensors go to TTNN
        const_val = arg
        if isinstance(const_val, (int, float, bool)):
          values[i] = const_val
        else:
          # map dtype to torch/ttnn
          if dtype == dtypes.float32:
            torch_dt, ttnn_dt = torch.float32, ttnn.float32
          elif dtype == dtypes.float16:
            torch_dt, ttnn_dt = torch.float16, ttnn.float16
          elif dtype == dtypes.int32:
            torch_dt, ttnn_dt = torch.int32, ttnn.int32
          else:
            torch_dt, ttnn_dt = torch.float32, ttnn.float32
          torch_tensor = torch.tensor(const_val, dtype=torch_dt)
          values[i] = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn_dt,
            layout=ttnn.TILE_LAYOUT,
            device=self._get_ttnn_device()
          )
        continue
      
      if op in unary_map:
        x = src_values[0] if src_values and src_values[0] is not None else values[src_indices[0]]
        values[i] = unary_map[op](x)
        continue
      
      if op in binary_map:
        a = src_values[0] if src_values and src_values[0] is not None else values[src_indices[0]]
        b = src_values[1] if len(src_values) > 1 and src_values[1] is not None else values[src_indices[1]]
        values[i] = binary_map[op](a, b)
        continue
      
      # Comparison operations not supported by current ttnn python API
      
      # Ternary operations
      if op is Ops.WHERE:
        c = src_values[0] if src_values and src_values[0] is not None else values[src_indices[0]]
        x = src_values[1] if len(src_values) > 1 and src_values[1] is not None else values[src_indices[1]]
        y = src_values[2] if len(src_values) > 2 and src_values[2] is not None else values[src_indices[2]]
        values[i] = ttnn.where(c, x, y)
        continue
      if op is Ops.WMMA:
        # Generic matmul via TTNN
        a_t = src_values[0] if src_values and src_values[0] is not None else values[src_indices[0]]
        b_t = src_values[1] if len(src_values) > 1 and src_values[1] is not None else values[src_indices[1]]
        # ttnn.matmul exists
        if hasattr(ttnn, 'matmul'):
          values[i] = ttnn.matmul(a_t, b_t)
        else:
          # fallback using add/mul reductions
          raise NotImplementedError("TTNN matmul not available")
        continue
      
      # Reduction operations
      if op is Ops.REDUCE_AXIS:
        # arg is (reduce_op, axis_tuple)
        if not arg or len(arg) < 2:
          reduce_op, axis = Ops.ADD, -1
        else:
          reduce_op, axis = arg[0], arg[1]
        in_tensor = src_values[0] if src_values and src_values[0] is not None else values[src_indices[0]]
        # compute dims natively for ttnn tensor in_tensor
        try:
          shape = tuple(in_tensor.shape)
          rank = len(shape)
        except Exception:
          # fallback: assume last dim
          rank = None
        dims = axis if isinstance(axis, (list, tuple)) else (axis,)
        if rank is not None:
          ndims = []
          for a in dims:
            a = int(a)
            if a < 0: a += rank
            if 0 <= a < rank: ndims.append(a)
          if not ndims:
            ndims = [rank-1]
          dims = tuple(sorted(set(ndims)))
        dim_arg = (dims[0] if len(dims) == 1 else list(dims))
        if reduce_op is Ops.ADD and hasattr(ttnn, 'sum'):
          values[i] = ttnn.sum(in_tensor, dim=dim_arg)
          continue
        if reduce_op is Ops.MAX and hasattr(ttnn, 'max'):
          values[i] = ttnn.max(in_tensor, dim=dim_arg)
          continue
        raise NotImplementedError(f"Reduction op {reduce_op} not implemented for TTNN")

      # Movement operations
      if op is Ops.VIEW:
        # VIEW on a tensor -> reshape; on a pointer -> no-op (handled by LOAD/STORE)
        base = src_values[0] if len(src_values) > 0 else None
        if base is None and src_indices and src_indices[0] in values:
          base = values[src_indices[0]]
        if base is None:
          # source not materialized yet, skip
          continue
        if hasattr(base, 'shape'):
          try:
            new_shape = tuple(arg.shape)
          except Exception:
            new_shape = None
          values[i] = base.reshape(new_shape) if new_shape is not None else base
        # Do not materialize pointers here
        continue
      if op is Ops.RESHAPE:
        new_shape = arg
        base = src_values[0] if len(src_values) > 0 else None
        if base is None and src_indices and src_indices[0] in values:
          base = values[src_indices[0]]
        if base is None:
          continue
        values[i] = ttnn.reshape(base, new_shape)
        continue
      if op is Ops.PERMUTE:
        dims = arg
        base = src_values[0] if len(src_values) > 0 else None
        if base is None and src_indices and src_indices[0] in values:
          base = values[src_indices[0]]
        if base is None:
          continue
        values[i] = ttnn.permute(base, dims)
        continue
      
      # Store operation
      if op is Ops.STORE:
        # STORE result tensor into resolved destination buffer
        result_tensor = src_values[1] if len(src_values) > 1 else (src_values[0] if src_values else None)
        if result_tensor is None:
          raise RuntimeError("TTNN STORE: missing result tensor")
        out_buf_idx = None
        for src_uop_idx in src_indices:
          maybe_idx = _runtime_buf_index_from_any(src_uop_idx)
          if maybe_idx is not None:
            out_buf_idx = maybe_idx
            break
        if out_buf_idx is None or out_buf_idx >= len(bufs):
          raise RuntimeError("TTNN STORE: unable to resolve destination buffer")
        # Store TTNN tensor directly
        bufs[out_buf_idx]["ttnn_tensor"] = result_tensor
        continue
      
      # Skip operations that don't produce values
      if op in {Ops.BARRIER, Ops.SINK, Ops.NOOP, Ops.ENDIF, Ops.IF}:
        continue
      
      # For unimplemented operations, raise an error for now
      raise NotImplementedError(f"TTNN backend doesn't support operation: {op}")
      
      # end per-uop
    
    # Synchronize device to ensure all operations complete
    if self.device:
      self.device.synchronize()

# ---- Main Device Class --------------------------------------------------------
class TTNNDevice(Compiled):
  """Tenstorrent TTNN device implementation"""

  def __init__(self, device: str = "TTNN:0"):
    global CACHED_DEVICE

    # Extract device ID from device string
    device_id = int(device.split(":")[1]) if ":" in device else 0

    # Initialize TTNN device
    # Disable block reordering to avoid sorting non-orderable args in TTNN pipeline
    os.environ["BLOCK_REORDER"] = "0"
    if not CACHED_DEVICE:
      CACHED_DEVICE = ttnn.open_device(device_id=device_id)
      print(f"TTNN device created: {CACHED_DEVICE}")
    self.ttnn_device = CACHED_DEVICE

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
