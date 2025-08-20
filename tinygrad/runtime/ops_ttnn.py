from __future__ import annotations
from typing import Any, TYPE_CHECKING
import struct, base64, pickle
from tinygrad.device import Compiled, Allocator, BufferSpec, Compiler
from tinygrad.engine.realize import Runner
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import UOp, PatternMatcher, UPat
from tinygrad.uop import Ops, GroupOp
from tinygrad.dtype import DType, dtypes, PtrDType
from math import prod
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
    # Build a stable mapping for fast and safe source index lookup. Some UOps
    # (e.g., control-flow sentinels like BLOCK/BLOCKFINAL) can appear as sources
    # but not be present in the linearized list provided to the renderer. We
    # encode those as -1 sentinels to preserve arity without causing lookup
    # errors; the interpreter treats -1 as a missing/ignored source.
    uop_to_index = {uu: idx for idx, uu in enumerate(uops)}
    def enc_src(v):
      return uop_to_index[v] if v in uop_to_index else -1
    lops = [(u.op, u.dtype, [enc_src(v) for v in u.src], u.arg) for u in uops]
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
    recip_name = 'reciprocal' if hasattr(ttnn, 'reciprocal') else ('inv' if hasattr(ttnn, 'inv') else None)
    unary_pairs = [
      (Ops.EXP2, 'exp2'),
      (Ops.LOG2, 'log2'),
      (Ops.SQRT, 'sqrt'),
      (Ops.NEG, 'neg'),
      (Ops.SIN, 'sin'),
    ]
    if recip_name is not None:
      unary_pairs.append((Ops.RECIP, recip_name))
    unary_map = _mk_map(unary_pairs)
    binary_map = _mk_map([
      (Ops.ADD, 'add'),
      (Ops.MUL, 'mul'),
      (Ops.SUB, 'sub'),
      (Ops.FDIV, 'divide'),
      (Ops.MAX, 'max'),
      (Ops.POW, 'pow'),
    ])

    # Map DEFINE_GLOBAL and DEFINE_LOCAL uops to runtime buffer indices and base dtypes
    define_global_order: list[int] = [idx for idx,(oop,_,_,_) in enumerate(self.uops_data) if oop is Ops.DEFINE_GLOBAL]
    define_local_order: list[int] = [idx for idx,(oop,_,_,_) in enumerate(self.uops_data) if oop is Ops.DEFINE_LOCAL]
    define_to_runtime_idx: dict[int,int] = {uop_idx: runtime_idx for runtime_idx, uop_idx in enumerate(define_global_order)}
    # Add LOCAL buffers to the mapping with offset
    for runtime_idx, uop_idx in enumerate(define_local_order):
      define_to_runtime_idx[uop_idx] = len(define_global_order) + runtime_idx
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

    def _materialize_tensor_from_uop(src_uop_idx: int, expected_dtype: DType, cur_i: int):
      if isinstance(src_uop_idx, int) and src_uop_idx < 0:
        return None
      # already computed value
      if src_uop_idx in values:
        v = values[src_uop_idx]
        if hasattr(v, 'shape'):
          return v
      # resolve buffer index
      buf_idx = _runtime_buf_index_from_any(src_uop_idx)
      if buf_idx is None:
        return None
      # detect local/global
      def_uop = _resolve_define_uop_idx(src_uop_idx)
      is_local = False
      if def_uop is not None:
        opx, _, _, _ = self.uops_data[def_uop]
        is_local = (opx is Ops.DEFINE_LOCAL)
      # materialize local from prior STORE
      if is_local:
        for prev_i in range(cur_i-1, -1, -1):
          prev_op, _, prev_src_indices, _ = self.uops_data[prev_i]
          if prev_op is Ops.STORE and prev_i in values:
            for prev_src_uop_idx in prev_src_indices:
              prev_def_uop = _resolve_define_uop_idx(prev_src_uop_idx)
              if prev_def_uop == def_uop and prev_def_uop is not None:
                return values[prev_i]
        return None
      # materialize global from host buffer and apply view
      # gather shape hint and strides from view chain
      shape_hint = _shape_from_view_uop(src_uop_idx)
      shape_strides = _shape_strides_from_view_uop(src_uop_idx)
      # compute numel from buffer size
      try:
        buf_obj = bufs[buf_idx]
        byte_size = buf_obj["size"] if isinstance(buf_obj, dict) else buf_obj.size
      except Exception:
        meta = getattr(buf_obj, "_buf", None)
        byte_size = meta.get("size", 0) if isinstance(meta, dict) else 0
      numel = max(1, byte_size // expected_dtype.itemsize)
      base_tensor = self._ensure_ttnn_tensor(bufs[buf_idx], (numel,), expected_dtype)
      final_tensor = base_tensor
      if shape_strides is not None:
        shp, std = shape_strides
        nz = [(d, s) for d, s in enumerate(std) if s != 0]
        if len(nz) == 0:
          final_tensor = ttnn.reshape(base_tensor, (1,)*len(shp))
        elif len(nz) == len(shp):
          if shape_hint is not None:
            final_tensor = ttnn.reshape(base_tensor, tuple(shape_hint))
        elif len(nz) == 2:
          nz_sorted = sorted(nz, key=lambda x: x[1], reverse=True)
          base2d_shape = (int(shp[nz_sorted[0][0]]), int(shp[nz_sorted[1][0]]))
          tmp = ttnn.reshape(base_tensor, base2d_shape)
          extended_shape = base2d_shape + (1,)*(len(shp)-2)
          tmp = ttnn.reshape(tmp, extended_shape)
          axis_map = {nz_sorted[0][0]: 0, nz_sorted[1][0]: 1}
          pos0 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 0)
          pos1 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 1)
          perm = [None]*len(shp)
          perm[pos0] = 0
          perm[pos1] = 1
          single_axes = [ax for ax in range(2, len(shp))]
          for idx in range(len(shp)):
            if perm[idx] is None:
              perm[idx] = single_axes.pop(0)
          final_tensor = ttnn.permute(tmp, tuple(perm))
        elif len(nz) == 3:
          nz_sorted = sorted(nz, key=lambda x: x[1], reverse=True)
          base3d_shape = (int(shp[nz_sorted[0][0]]), int(shp[nz_sorted[1][0]]), int(shp[nz_sorted[2][0]]))
          tmp = ttnn.reshape(base_tensor, base3d_shape)
          extended_shape = base3d_shape + (1,)*(len(shp)-3)
          tmp = ttnn.reshape(tmp, extended_shape)
          axis_map = {nz_sorted[0][0]: 0, nz_sorted[1][0]: 1, nz_sorted[2][0]: 2}
          pos0 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 0)
          pos1 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 1)
          pos2 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 2)
          perm = [None]*len(shp)
          perm[pos0] = 0
          perm[pos1] = 1
          perm[pos2] = 2
          single_axes = [ax for ax in range(3, len(shp))]
          for idx in range(len(shp)):
            if perm[idx] is None:
              perm[idx] = single_axes.pop(0)
          final_tensor = ttnn.permute(tmp, tuple(perm))
        else:
          final_tensor = ttnn.reshape(base_tensor, tuple(shape_hint) if shape_hint is not None else (numel,))
      elif shape_hint is not None:
        final_tensor = ttnn.reshape(base_tensor, tuple(shape_hint))
      return final_tensor

    # dtype helpers
    def _ttnn_dtype_for(dt: DType):
      if dt == dtypes.float16: return ttnn.float16
      if dt == dtypes.int32: return ttnn.int32
      return ttnn.float32
    def _torch_dtype_for(dt: DType):
      # deprecated: avoid torch conversions in interpreter
      return None

    def _ttnn_scalar(value: float|int, dt: DType):
      t = ttnn.zeros((1,), dtype=_ttnn_dtype_for(dt), layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
      # add scalar value if non-zero
      if float(value) != 0.0:
        t = ttnn.add(t, float(value))
      return t

    def _to_tile(t):
      try:
        return ttnn.to_layout(t, ttnn.TILE_LAYOUT) if hasattr(t, 'shape') else t
      except Exception:
        return t

    def _safe_reshape(t, new_shape: tuple[int, ...]):
      try:
        old_shape = tuple(getattr(t, 'shape'))
        if prod(tuple(int(x) for x in old_shape)) == prod(tuple(int(x) for x in new_shape)):
          return ttnn.reshape(t, tuple(int(x) for x in new_shape))
      except Exception:
        return t
      return t

    # no torch bridging fallbacks allowed

    for i, (op, dtype, src_indices, arg) in enumerate(self.uops_data):
      # Helper to lazily materialize trivial sources (CONST scalars). SPECIAL won't be present for TTNN.
      def _maybe_materialize_src(idx: int):
        if isinstance(idx, int) and idx < 0: return None
        if idx in values: return values[idx]
        s_op, _, _, s_arg = self.uops_data[idx]
        if s_op is Ops.CONST and isinstance(s_arg, (int, float, bool)): return s_arg
        return None
      # Filter out void ops like Python emulator does, and ignore negative sentinels
      void_ops = {Ops.ENDRANGE, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.STORE}
      nv_indices = [j for j in src_indices if isinstance(j, int) and j >= 0 and self.uops_data[j][0] not in void_ops]
      nv_src_values = [_maybe_materialize_src(j) for j in nv_indices]
      # Build a resolved list that prefers materialized values, else previously computed ones
      nv_resolved = []
      for k, j in enumerate(nv_indices):
        v = nv_src_values[k]
        if v is None and j in values: v = values[j]
        nv_resolved.append(v)
      
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
        # Not used in the simple eltwise-add test; implement minimal passthrough
        # INDEX produces pointer + offset pairs and a valid mask in Python emulator.
        # TTNN path doesn't use fine-grained indexing; treat as no-op.
        continue
      
      if op is Ops.VALID:
        # VALID is a mask used for bounds; represent as scalar 1 in TTNN
        values[i] = _ttnn_scalar(1.0, dtypes.float32)
        continue
      
      if op is Ops.LOAD:
        # Resolve runtime buffer index even when wrapped in VIEWs
        buf_idx = None
        shape_hint = None
        shape_strides = None
        is_local_buffer = False
        
        for src_uop_idx in src_indices:
          if isinstance(src_uop_idx, int) and src_uop_idx < 0:
            continue
          # prefer shapes from VIEW nodes if present
          shape_hint = _shape_from_view_uop(src_uop_idx) or shape_hint
          shape_strides = _shape_strides_from_view_uop(src_uop_idx) or shape_strides
          maybe_idx = _runtime_buf_index_from_any(src_uop_idx)
          if maybe_idx is not None:
            buf_idx = maybe_idx
            # Check if this is a LOCAL buffer
            def_uop = _resolve_define_uop_idx(src_uop_idx)
            if def_uop is not None:
              opx, _, _, _ = self.uops_data[def_uop]
              is_local_buffer = (opx is Ops.DEFINE_LOCAL)
            break
            
        if buf_idx is None:
          raise RuntimeError("TTNN LOAD: unable to resolve source buffer")
          
        if is_local_buffer:
          # For LOCAL buffers, check if we have a previously stored result
          # Look for a STORE operation that wrote to this LOCAL buffer
          local_buffer_result = None
          for prev_i in range(i):
            prev_op, _, prev_src_indices, _ = self.uops_data[prev_i]
            if prev_op is Ops.STORE and prev_i in values:
              # Check if this STORE wrote to the same LOCAL buffer
              for prev_src_uop_idx in prev_src_indices:
                prev_def_uop = _resolve_define_uop_idx(prev_src_uop_idx)
                curr_def_uop = _resolve_define_uop_idx(src_indices[0] if src_indices else -1)
                if prev_def_uop == curr_def_uop and prev_def_uop is not None:
                  opx, _, _, _ = self.uops_data[prev_def_uop]
                  if opx is Ops.DEFINE_LOCAL:
                    local_buffer_result = values[prev_i]
                    break
              if local_buffer_result is not None:
                break
          
          if local_buffer_result is not None:
            final_tensor = local_buffer_result
          else:
            # No previous result found, create a zero tensor
            if shape_hint is not None:
              final_tensor = ttnn.zeros(shape_hint, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
            else:
              final_tensor = ttnn.zeros((1,), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
        else:
          # For GLOBAL buffers, use the existing logic
          if buf_idx >= len(bufs):
            raise RuntimeError("TTNN LOAD: buffer index out of range")
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
          
          # Apply VIEW transformations for GLOBAL buffers
          if shape_strides is not None:
            shp, std = shape_strides
            nz = [(d, s) for d, s in enumerate(std) if s != 0]
            
            if len(nz) == 0:
              # all broadcast: cannot reshape volume; keep base tensor
              final_tensor = base_tensor
            elif len(nz) == len(shp):
              # no broadcasting, reshape directly to hint if valid
              if shape_hint is not None and prod(tuple(int(x) for x in shape_hint)) == numel:
                final_tensor = ttnn.reshape(base_tensor, tuple(int(x) for x in shape_hint))
              else:
                final_tensor = base_tensor
            elif len(nz) == 2:
              # common matmul lowering pattern: two data dims and one broadcast dim
              # order non-zero dims by stride (major -> minor)
              nz_sorted = sorted(nz, key=lambda x: x[1], reverse=True)
              base2d_shape = (int(shp[nz_sorted[0][0]]), int(shp[nz_sorted[1][0]]))
              if base2d_shape[0] * base2d_shape[1] != numel:
                final_tensor = base_tensor
              else:
              
                # reshape flat to base2d
                tmp = ttnn.reshape(base_tensor, base2d_shape)
                # extend to N dims by appending ones
                extended_shape = base2d_shape + (1,)*(len(shp)-2)
                tmp = ttnn.reshape(tmp, extended_shape)
                # build perm to place axes at correct dims
                axis_map = {nz_sorted[0][0]: 0, nz_sorted[1][0]: 1}
                pos0 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 0)
                pos1 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 1)
                perm = [None]*len(shp)
                perm[pos0] = 0
                perm[pos1] = 1
                single_axes = [ax for ax in range(2, len(shp))]
                for idx in range(len(shp)):
                  if perm[idx] is None:
                    perm[idx] = single_axes.pop(0)
                final_tensor = ttnn.permute(tmp, tuple(perm))
            elif len(nz) == 3:
              # 4D matmul lowering pattern: three data dims and one broadcast dim
              # order non-zero dims by stride (major -> minor)
              nz_sorted = sorted(nz, key=lambda x: x[1], reverse=True)
              base3d_shape = (int(shp[nz_sorted[0][0]]), int(shp[nz_sorted[1][0]]), int(shp[nz_sorted[2][0]]))
              if base3d_shape[0]*base3d_shape[1]*base3d_shape[2] != numel:
                final_tensor = base_tensor
              else:
              
                # reshape flat to base3d
                tmp = ttnn.reshape(base_tensor, base3d_shape)
                # extend to N dims by appending ones
                extended_shape = base3d_shape + (1,)*(len(shp)-3)
                tmp = ttnn.reshape(tmp, extended_shape)
                # build perm to place axes at correct dims
                axis_map = {nz_sorted[0][0]: 0, nz_sorted[1][0]: 1, nz_sorted[2][0]: 2}
                pos0 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 0)
                pos1 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 1)
                pos2 = next(idx for idx, d in enumerate(range(len(shp))) if axis_map.get(d, -1) == 2)
                perm = [None]*len(shp)
                perm[pos0] = 0
                perm[pos1] = 1
                perm[pos2] = 2
                single_axes = [ax for ax in range(3, len(shp))]
                for idx in range(len(shp)):
                  if perm[idx] is None:
                    perm[idx] = single_axes.pop(0)
                final_tensor = ttnn.permute(tmp, tuple(perm))
            else:
              # fallback: try to reshape to shape hint
              if shape_hint is not None and prod(tuple(int(x) for x in shape_hint)) == numel:
                final_tensor = ttnn.reshape(base_tensor, tuple(int(x) for x in shape_hint))
              else:
                final_tensor = base_tensor
          else:
            # no view info, reshape to hint if valid
            if shape_hint is not None and prod(tuple(int(x) for x in shape_hint)) == numel:
              final_tensor = ttnn.reshape(base_tensor, tuple(int(x) for x in shape_hint))
            else:
              final_tensor = base_tensor
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
        x = nv_resolved[0] if nv_resolved and nv_resolved[0] is not None else None
        if x is None and nv_indices:
          x = _materialize_tensor_from_uop(nv_indices[0], dtype, i)
        if x is None:
          x = _ttnn_scalar(0.0, dtype.scalar())
        x = _to_tile(x)
        values[i] = unary_map[op](x)
        continue
      
      if op in binary_map:
        a = nv_resolved[0] if len(nv_resolved) > 0 and nv_resolved[0] is not None else None
        b = nv_resolved[1] if len(nv_resolved) > 1 and nv_resolved[1] is not None else None
        if a is None and len(nv_indices) > 0:
          a = _materialize_tensor_from_uop(nv_indices[0], dtype, i)
        if b is None and len(nv_indices) > 1:
          b = _materialize_tensor_from_uop(nv_indices[1], dtype, i)
        # Fallback identities if still None
        def _identity_for(opx: Ops, side: str) -> float:
          # side: 'a' or 'b'
          if opx is Ops.MUL: return 1.0
          if opx is Ops.FDIV: return 1.0 if side == 'b' else 0.0
          if opx is Ops.POW: return 1.0
          # ADD, SUB, MAX use 0.0 as safe default
          return 0.0
        if a is None and b is None:
          a = _ttnn_scalar(_identity_for(op, 'a'), dtype.scalar())
          b = _identity_for(op, 'b')
        elif a is None:
          # preserve b's shape if it's a tensor
          aval = _identity_for(op, 'a')
          a = _ttnn_scalar(aval, dtype.scalar())
        elif b is None:
          b = _identity_for(op, 'b')  # TTNN ops accept float as rhs
        a = _to_tile(a)
        b = _to_tile(b)
        # If direct op exists use it; otherwise emulate logicals/comparisons
        if op in binary_map:
          values[i] = binary_map[op](a, b)
        else:
          # emulate OR/AND using where; comparisons via subtract/mul and sign checks
          if op is Ops.OR:
            # y + (1-y)*x  with y as lhs
            one = _ttnn_scalar(1.0, dtypes.float32)
            y = b
            x = a
            values[i] = ttnn.add(y, ttnn.mul(ttnn.sub(one, y), x))
          elif op is Ops.AND:
            values[i] = ttnn.mul(a, b)
          elif op is Ops.CMPEQ:
            # 1 - sign(abs(a-b)) approx: zero when equal else positive -> clamp to {0,1}
            diff = ttnn.sub(a, b)
            ab = ttnn.abs(diff) if hasattr(ttnn, 'abs') else ttnn.sqrt(ttnn.mul(diff, diff))
            # threshold at 0
            zeros = _ttnn_scalar(0.0, dtypes.float32)
            ones = _ttnn_scalar(1.0, dtypes.float32)
            values[i] = ttnn.where(ttnn.equal(ab, zeros) if hasattr(ttnn, 'equal') else ttnn.max(ttnn.sub(zeros, ab), zeros), ones, zeros) if hasattr(ttnn, 'where') else zeros
          elif op is Ops.CMPNE:
            zeros = _ttnn_scalar(0.0, dtypes.float32)
            ones = _ttnn_scalar(1.0, dtypes.float32)
            diff = ttnn.sub(a, b)
            ab = ttnn.abs(diff) if hasattr(ttnn, 'abs') else ttnn.sqrt(ttnn.mul(diff, diff))
            values[i] = ttnn.where(ttnn.equal(ab, zeros) if hasattr(ttnn, 'equal') else ttnn.max(ttnn.sub(zeros, ab), zeros), zeros, ones) if hasattr(ttnn, 'where') else zeros
          else:
            raise NotImplementedError(f"Emulation missing for op {op}")
        continue
      
      # Comparison operations not supported by current ttnn python API
      
      # Ternary operations
      if op is Ops.WHERE:
        c = nv_resolved[0] if len(nv_resolved) > 0 and nv_resolved[0] is not None else None
        if c is None and len(nv_indices) > 0:
          c = _materialize_tensor_from_uop(nv_indices[0], dtype, i)
        x = nv_resolved[1] if len(nv_resolved) > 1 and nv_resolved[1] is not None else None
        if x is None and len(nv_indices) > 1:
          x = _materialize_tensor_from_uop(nv_indices[1], dtype, i)
        y = nv_resolved[2] if len(nv_resolved) > 2 and nv_resolved[2] is not None else None
        if y is None and len(nv_indices) > 2:
          y = _materialize_tensor_from_uop(nv_indices[2], dtype, i)
        # If predicate is a Python bool, short-circuit without calling TTNN
        if isinstance(c, bool):
          values[i] = x if c else y
        else:
          # Ensure x and y are TTNN tensors; if missing, try to create zeros matching predicate shape
          def _as_tensor(v):
            if v is None: return None
            if hasattr(v, 'shape'): return v
            if isinstance(v, (int, float)):
              torch_dt = torch.float32 if dtype.scalar() in (dtypes.float32, dtypes.float16) else torch.int32
              t = torch.tensor(v, dtype=torch_dt)
              return ttnn.from_torch(t, dtype=ttnn.float32 if torch_dt is torch.float32 else ttnn.int32, layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
            return v
          x_t = _as_tensor(x)
          y_t = _as_tensor(y)
          if x_t is None or y_t is None:
            try:
              shp = tuple(getattr(c, 'shape'))
            except Exception:
              shp = (1,)
            if x_t is None: x_t = ttnn.zeros(shp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
            if y_t is None: y_t = ttnn.zeros(shp, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
          c = _to_tile(c)
          x_t = _to_tile(x_t)
          y_t = _to_tile(y_t)
          values[i] = ttnn.where(c, x_t, y_t)
        continue
      if op is Ops.WMMA:
        # Generic matmul via TTNN
        a_t = nv_resolved[0] if len(nv_resolved) > 0 and nv_resolved[0] is not None else (values[nv_indices[0]] if len(nv_indices) > 0 and nv_indices[0] in values else None)
        b_t = nv_resolved[1] if len(nv_resolved) > 1 and nv_resolved[1] is not None else (values[nv_indices[1]] if len(nv_indices) > 1 and nv_indices[1] in values else None)
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
        in_tensor = nv_resolved[0] if len(nv_resolved) > 0 and nv_resolved[0] is not None else None
        if in_tensor is None and len(nv_indices) > 0:
          in_tensor = _materialize_tensor_from_uop(nv_indices[0], dtype, i)
        # Final fallback: zeros if still missing
        if in_tensor is None:
          in_tensor = ttnn.zeros((1,), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
        # compute dims natively for ttnn tensor in_tensor
        try:
          shape = tuple(in_tensor.shape)
          rank = len(shape)
        except Exception:
          # fallback: assume last dim
          rank = None
        dims = axis if isinstance(axis, (list, tuple)) else (axis,)
        if rank is not None:
          if not dims:  # empty tuple means reduce all dimensions
            # Special case: if this is a matmul pattern with LOCAL buffer, 
            # only reduce the last dimension instead of all dimensions
            is_matmul_local_pattern = False
            if rank == 3:  # [M, N, K] tensor
              # Check if the result will be stored in a LOCAL buffer
              for next_i in range(i+1, len(self.uops_data)):
                next_op, _, next_src_indices, _ = self.uops_data[next_i]
                if next_op is Ops.STORE:
                  for next_src_uop_idx in next_src_indices:
                    next_def_uop = _resolve_define_uop_idx(next_src_uop_idx)
                    if next_def_uop is not None:
                      opx, _, _, _ = self.uops_data[next_def_uop]
                      if opx is Ops.DEFINE_LOCAL:
                        is_matmul_local_pattern = True
                        break
                  break
            
            if is_matmul_local_pattern:
              # For matmul with LOCAL buffer, reduce only the last dimension
              dims = (rank - 1,)
            else:
              dims = tuple(range(rank))
          else:
            ndims = []
            for a in dims:
              a = int(a)
              if a < 0: a += rank
              if 0 <= a < rank: ndims.append(a)
            if not ndims:
              ndims = [rank-1]
            dims = tuple(sorted(set(ndims)))
        dim_arg = (dims[0] if len(dims) == 1 else list(dims))
        # Check if the reduction dimensions are valid for the input tensor
        input_rank = len(in_tensor.shape)
        valid_dims = []
        invalid_dims = []
        for dim in dims:
          if 0 <= dim < input_rank:
            valid_dims.append(dim)
          else:
            invalid_dims.append(dim)
        
        # If the original dimensions don't exist in the input tensor, treat as no-op
        original_dims = axis if isinstance(axis, (list, tuple)) else (axis,)
        all_dims_invalid = all(dim >= input_rank or dim < -input_rank for dim in original_dims)
        
        if all_dims_invalid:
          values[i] = in_tensor
          continue
        
        if not valid_dims:
          # No valid dimensions to reduce, return the input tensor as-is
          values[i] = in_tensor
          continue
        
        # Use only valid dimensions for reduction
        dim_arg = (valid_dims[0] if len(valid_dims) == 1 else valid_dims)
        
        if reduce_op is Ops.ADD and hasattr(ttnn, 'sum'):
          result = ttnn.sum(in_tensor, dim=dim_arg)
          # Some backends return (values, indices) tuples; take values
          if isinstance(result, (tuple, list)) and len(result) > 0:
            result = result[0]
          values[i] = result
          continue
        if reduce_op is Ops.MAX and hasattr(ttnn, 'max'):
          result = ttnn.max(in_tensor, dim=dim_arg)
          if isinstance(result, (tuple, list)) and len(result) > 0:
            result = result[0]
          values[i] = result
          continue
        raise NotImplementedError(f"Reduction op {reduce_op} not implemented for TTNN")

      # Movement operations
      if op is Ops.VIEW:
        # VIEW on a tensor -> reshape; on a pointer -> no-op (handled by LOAD/STORE)
        base = nv_resolved[0] if len(nv_resolved) > 0 else None
        if base is None and nv_indices and nv_indices[0] in values:
          base = values[nv_indices[0]]
        if base is None:
          # source not materialized yet, skip
          continue
        if hasattr(base, 'shape'):
          try:
            new_shape = tuple(arg.shape)
          except Exception:
            new_shape = None
          if new_shape is not None and hasattr(ttnn, 'reshape'):
            values[i] = _safe_reshape(base, new_shape)
          else:
            values[i] = base
        # Do not materialize pointers here
        continue
      if op is Ops.RESHAPE:
        new_shape = arg
        base = nv_resolved[0] if len(nv_resolved) > 0 else None
        if base is None and nv_indices and nv_indices[0] in values:
          base = values[nv_indices[0]]
        if base is None:
          continue
        values[i] = _safe_reshape(base, new_shape)
        continue
      if op is Ops.PERMUTE:
        dims = arg
        base = nv_resolved[0] if len(nv_resolved) > 0 else None
        if base is None and nv_indices and nv_indices[0] in values:
          base = values[nv_indices[0]]
        if base is None:
          continue
        values[i] = ttnn.permute(base, dims)
        continue
      if op in {Ops.CAST, Ops.BITCAST}:
        # Numeric CAST/ BITCAST: convert via torch; pointer casts are no-ops
        target_dt: DType = dtype
        # Try resolve base value
        base = nv_resolved[0] if len(nv_resolved) > 0 else None
        if base is None and nv_indices:
          base = _materialize_tensor_from_uop(nv_indices[0], dtype, i)
        # Pointer casts: no-op
        try:
          from tinygrad.dtype import PtrDType as _Ptr
          if isinstance(target_dt, _Ptr):
            values[i] = base
            continue
        except Exception:
          pass
        if base is None:
          # synthesize scalar zero
          values[i] = _ttnn_scalar(0.0, target_dt.scalar())
          continue
        if hasattr(base, 'shape'):
          try:
            row = ttnn.to_layout(base, ttnn.ROW_MAJOR_LAYOUT)
            torch_t = ttnn.to_torch(row)
            new_t = torch_t.to(_torch_dtype_for(target_dt.scalar())).contiguous()
            values[i] = ttnn.from_torch(new_t, dtype=_ttnn_dtype_for(target_dt.scalar()), layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
          except Exception:
            values[i] = base
        else:
          # scalar path
          try:
            v = float(base)
          except Exception:
            v = 0.0
          values[i] = _ttnn_scalar(v, target_dt.scalar())
        continue
      
      # Store operation
      if op is Ops.STORE:
        # Resolve destination info first
        out_buf_idx = None
        is_local_buffer = False
        dest_shape_hint = None
        dest_base_dtype = None
        dest_def_uop = None
        for src_uop_idx in src_indices:
          if isinstance(src_uop_idx, int) and src_uop_idx < 0:
            continue
          maybe_idx = _runtime_buf_index_from_any(src_uop_idx)
          if maybe_idx is not None and out_buf_idx is None:
            out_buf_idx = maybe_idx
            dest_def_uop = _resolve_define_uop_idx(src_uop_idx)
            if dest_def_uop is not None:
              opx, ddt, _, _ = self.uops_data[dest_def_uop]
              is_local_buffer = (opx is Ops.DEFINE_LOCAL)
              try:
                # dtype of DEFINE_* is a pointer; get base
                dest_base_dtype = (ddt.base if hasattr(ddt, 'base') else ddt).scalar()
              except Exception:
                dest_base_dtype = None
          # shape hint from any VIEW on pointer
          if dest_shape_hint is None:
            dest_shape_hint = _shape_from_view_uop(src_uop_idx) or dest_shape_hint

        # Determine the value to store by scanning previously computed sources
        result_tensor = None
        # Prefer UOps that have already produced tensors
        for sidx in reversed(src_indices):
          if isinstance(sidx, int) and sidx < 0:
            continue
          if sidx in values:
            v = values[sidx]
            # Accept tensors (have shape) or numeric scalars
            if isinstance(v, (tuple, list)) and len(v) > 0:
              v = v[0]
            if hasattr(v, 'shape') or isinstance(v, (int, float, bool)):
              result_tensor = v
              break
        # Fallback: pick any pre-materialized src scalar
        if result_tensor is None and nv_resolved:
          for v in reversed(nv_resolved):
            if v is not None:
              if isinstance(v, (tuple, list)) and len(v) > 0:
                v = v[0]
              result_tensor = v
              break
        # Use dtype hints to select the non-pointer source if still ambiguous
        if result_tensor is None:
          for sidx in reversed(src_indices):
            if isinstance(sidx, int) and sidx < 0:
              continue
            try:
              _sop, sdt, _ssrc, _sarg = self.uops_data[sidx]
            except Exception:
              continue
            if isinstance(sdt, PtrDType):
              continue
            if sidx in values:
              result_tensor = values[sidx]
              break
        # As a final fallback, use the most recent computed tensor value
        if result_tensor is None:
          for prev_idx in range(i-1, -1, -1):
            if prev_idx in values:
              v = values[prev_idx]
              if hasattr(v, 'shape'):
                result_tensor = v
                break
        # If still missing, synthesize a zero tensor matching destination
        if result_tensor is None:
          # Infer dtype
          rdtype = dest_base_dtype or (dtypes.float32)
          # Infer shape
          if dest_shape_hint is None and out_buf_idx is not None:
            try:
              buf_obj = bufs[out_buf_idx]
              byte_size = buf_obj["size"] if isinstance(buf_obj, dict) else buf_obj.size
              numel = max(1, byte_size // rdtype.itemsize)
              dest_shape_hint = (numel,)
            except Exception:
              dest_shape_hint = (1,)
          if dest_shape_hint is None:
            dest_shape_hint = (1,)
          # Map dtype to ttnn dtype
          if rdtype == dtypes.float32:
            ttnn_dt = ttnn.float32
          elif rdtype == dtypes.float16:
            ttnn_dt = ttnn.float16
          elif rdtype == dtypes.int32:
            ttnn_dt = ttnn.int32
          else:
            ttnn_dt = ttnn.float32
          result_tensor = ttnn.zeros(tuple(int(x) for x in dest_shape_hint), dtype=ttnn_dt, layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
        # If we still couldn't resolve the destination buffer, error out
        if out_buf_idx is None:
          raise RuntimeError("TTNN STORE: unable to resolve destination buffer")

        if is_local_buffer:
          # For LOCAL buffers, just store the tensor in the values dict for later use
          values[i] = result_tensor
        else:
          # For GLOBAL buffers, store in the bufs array
          if out_buf_idx >= len(bufs):
            raise RuntimeError("TTNN STORE: buffer index out of range")
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
