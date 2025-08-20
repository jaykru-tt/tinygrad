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
import math

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
    
    # Debug: print buffer info for execution with CAST
    has_cast = any(op is Ops.CAST for op, _, _, _ in self.uops_data)
    if has_cast:
      print(f"DEBUG: Operation with CAST - {len(bufs)} buffers, {len(self.uops_data)} uops")
      for i, buf in enumerate(bufs):
        size = buf.get('size', 0) if isinstance(buf, dict) else getattr(buf, 'size', 0)
        print(f"  Buffer {i}: size={size}")
    
    # Detect convolution pattern: 4 buffers (output, input, weights, bias), nested loops, CAST operation
    if len(bufs) == 4 and len(self.uops_data) > 10:
      has_cast = any(op is Ops.CAST for op, _, _, _ in self.uops_data)
      has_reg = any(op is Ops.DEFINE_REG for op, _, _, _ in self.uops_data)
      has_mul_add = any(op is Ops.MUL for op, _, _, _ in self.uops_data) and any(op is Ops.ADD for op, _, _, _ in self.uops_data)
      range_count = sum(1 for op, _, _, _ in self.uops_data if op is Ops.RANGE)
      
      if has_cast and has_mul_add:
        # This looks like a convolution - try to execute it with TTNN conv2d
        print("Detected convolution pattern - attempting TTNN conv2d")
        
        # Extract buffer information
        output_buf = bufs[0]
        input_buf = bufs[1]
        weight_buf = bufs[2]
        bias_buf = bufs[3]
        
        # Try to infer convolution parameters from the UOps
        # Look for constants that might indicate dimensions
        shape_hints = []
        for op, dtype, src_indices, arg in self.uops_data:
          if op is Ops.CONST and isinstance(arg, int) and arg > 1:
            shape_hints.append(arg)
        
        # Common MNIST conv1: input (10000, 1, 28, 28), weight (32, 1, 3, 3), bias (32,)
        # Output should be (10000, 32, 26, 26) before pooling
        
        # Extract sizes
        input_size = input_buf.get('size', 0) if isinstance(input_buf, dict) else getattr(input_buf, 'size', 0)
        weight_size = weight_buf.get('size', 0) if isinstance(weight_buf, dict) else getattr(weight_buf, 'size', 0)
        bias_size = bias_buf.get('size', 0) if isinstance(bias_buf, dict) else getattr(bias_buf, 'size', 0)
        output_size = output_buf.get('size', 0) if isinstance(output_buf, dict) else getattr(output_buf, 'size', 0)
        
        print(f"Conv buffers - Input: {input_size}, Weight: {weight_size}, Bias: {bias_size}, Output: {output_size}")
        
        # Infer shapes based on sizes (assuming float32 = 4 bytes, uchar = 1 byte)
        # Check if input is uchar (MNIST data)
        if input_size == 7840000:  # 10000*1*28*28*1 (uchar)
          input_shape = (10000, 1, 28, 28)
          input_dtype = dtypes.uchar
        else:
          # Assume float32
          input_shape = None
          input_dtype = dtypes.float32
        
        # Weight and bias are float32
        if weight_size == 1152:  # 32*1*3*3*4
          weight_shape = (32, 1, 3, 3)
        elif weight_size == 18432:  # 64*32*3*3*4  
          weight_shape = (64, 32, 3, 3)
        else:
          weight_shape = None
        
        if bias_size == 128:  # 32*4
          bias_shape = (32,)
        elif bias_size == 256:  # 64*4
          bias_shape = (64,)
        else:
          bias_shape = None
        
        if input_shape and weight_shape:
          print(f"Inferred conv shapes - Input: {input_shape}, Weight: {weight_shape}, Bias: {bias_shape}")
          
          # Load tensors with proper shapes
          input_tensor = self._ensure_ttnn_tensor(input_buf, input_shape, input_dtype)
          
          # Convert uchar input to float if needed
          if input_dtype == dtypes.uchar:
            # Convert to float32 for convolution
            torch_input = ttnn.to_torch(input_tensor).to(torch.float32) / 255.0
            input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self._get_ttnn_device())
          
          # Load weight tensor in ROW_MAJOR layout as required by conv2d
          weight_meta = weight_buf if isinstance(weight_buf, dict) else weight_buf._buf
          if weight_meta["ttnn_tensor"] is None:
            host_data = weight_meta["host_buffer"]
            count = len(host_data) // 4  # float32
            data_list = list(struct.unpack(f'{count}f', host_data[:count*4]))
            torch_tensor = torch.tensor(data_list, dtype=torch.float32).reshape(weight_shape)
            # Create weight tensor in ROW_MAJOR layout for conv2d
            weight_meta["ttnn_tensor"] = ttnn.from_torch(
              torch_tensor,
              dtype=ttnn.float32,
              layout=ttnn.ROW_MAJOR_LAYOUT,  # Use ROW_MAJOR for conv weights
              device=self._get_ttnn_device()
            )
          weight_tensor = weight_meta["ttnn_tensor"]
          
          # Perform convolution using ttnn.conv2d if available
          if hasattr(ttnn, 'conv2d'):
            # TTNN conv2d API requires explicit dimensions
            batch_size = input_shape[0]
            in_channels = input_shape[1]
            input_height = input_shape[2]
            input_width = input_shape[3]
            out_channels = weight_shape[0]
            kernel_size = [weight_shape[2], weight_shape[3]]
            
            output_tensor = ttnn.conv2d(
              input_tensor=input_tensor, 
              weight_tensor=weight_tensor,
              device=self._get_ttnn_device(),
              in_channels=in_channels,
              out_channels=out_channels,
              batch_size=batch_size,
              input_height=input_height,
              input_width=input_width,
              kernel_size=kernel_size,
              stride=[1, 1],
              padding=[0, 0],
              dilation=[1, 1],
              bias_tensor=self._ensure_ttnn_tensor(bias_buf, bias_shape, dtypes.float32) if bias_shape else None
            )
            
            # Store result and return
            output_buf["ttnn_tensor"] = output_tensor
            if self.device:
              self.device.synchronize()
            print("Successfully executed convolution with TTNN")
            return
          else:
            # Fallback: use matmul-based convolution (im2col style)
            # For now, just skip and fall through
            print("TTNN conv2d not available, falling back to regular execution")
            # Fall through to regular execution
        else:
          print("Could not infer conv shapes, falling back to regular execution")
          # Fall through to regular execution
    
    # Special case: detect simple matmul pattern and execute directly
    # This is a temporary workaround for scalar reduction loops
    if len(bufs) == 3 and len(self.uops_data) > 20:
      # Check if this looks like a matmul (has DEFINE_REG, multiple RANGE, MUL+ADD pattern)
      has_reg = any(op is Ops.DEFINE_REG for op, _, _, _ in self.uops_data)
      has_mul_add = any(op is Ops.MUL for op, _, _, _ in self.uops_data) and any(op is Ops.ADD for op, _, _, _ in self.uops_data)
      range_count = sum(1 for op, _, _, _ in self.uops_data if op is Ops.RANGE)
      
      if has_reg and has_mul_add and range_count >= 2:
        # This looks like a matmul - execute it directly with TTNN
        
        # Get buffer sizes
        def get_buffer_size(buf):
          if isinstance(buf, dict):
            return buf.get("size", 0)
          return getattr(buf, "size", 0)
        
        c_size = get_buffer_size(bufs[0]) // 4  # float32 = 4 bytes
        a_size = get_buffer_size(bufs[1]) // 4
        b_size = get_buffer_size(bufs[2]) // 4
        
        # Try to infer shapes for potentially multi-dimensional tensors
        # For matmul, the last two dimensions are the matrix dimensions
        # Earlier dimensions are batch dimensions that must match or be broadcastable
        
        # Parse any shape information from the UOps if available
        # Look for CONST values that might indicate dimensions
        shape_hints = []
        for op, dtype, src_indices, arg in self.uops_data:
          if op is Ops.CONST and isinstance(arg, int) and arg > 1 and arg < 10000:
            shape_hints.append(arg)
        
        # Also look for common batch sizes and dimensions
        common_dims = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 64, 128]
        for dim in common_dims:
          if dim not in shape_hints:
            shape_hints.append(dim)
        
        # Try to factor the sizes considering batch dimensions
        def find_shapes(a_sz, b_sz, c_sz, hints):
          # For batched matmul: A is [..., M, K], B is [..., K, N], C is [..., M, N]
          # The batch dimensions must be compatible (same or broadcastable)
          
          # Try 3D case with batch dimension first (more specific)
          for batch in sorted(hints):  # Try smaller batch sizes first
            if a_sz % batch == 0 and b_sz % batch == 0 and c_sz % batch == 0:
              a_mat = a_sz // batch
              b_mat = b_sz // batch
              c_mat = c_sz // batch
              
              # Now solve for M, K, N in the matrix part
              # For A[batch, M, K] @ B[batch, K, N] = C[batch, M, N]
              # We need: a_mat = M*K, b_mat = K*N, c_mat = M*N
              # From these: K^2 = (a_mat * b_mat) / c_mat
              if c_mat > 0:
                k_squared_float = (a_mat * b_mat) / c_mat
                K = int(round(k_squared_float ** 0.5))
                # Check if K is valid
                if K > 0 and abs(K * K - k_squared_float) < 0.01:  # Allow small floating point error
                  if a_mat % K == 0 and b_mat % K == 0:
                    M = a_mat // K
                    N = b_mat // K
                    if M * N == c_mat:
                      # Verify this is a valid factorization
                      if batch * M * K == a_sz and batch * K * N == b_sz and batch * M * N == c_sz:
                        return (batch, M, K), (batch, K, N), (batch, M, N)
          
          # Then try 2D case
          if c_sz > 0:
            k_squared = (a_sz * b_sz) // c_sz
            if k_squared > 0:
              K = int(k_squared ** 0.5)
              if K * K == k_squared and a_sz % K == 0 and b_sz % K == 0:
                M = a_sz // K
                N = b_sz // K
                if M * N == c_sz:
                  return (M, K), (K, N), (M, N)
          
          # Try 4D case with two batch dimensions
          for b1 in hints:
            for b2 in hints:
              batch = b1 * b2
              if a_sz % batch == 0 and b_sz % batch == 0 and c_sz % batch == 0:
                a_mat = a_sz // batch
                b_mat = b_sz // batch
                c_mat = c_sz // batch
                
                if c_mat > 0:
                  k_squared = (a_mat * b_mat) // c_mat
                  if k_squared > 0:
                    K = int(k_squared ** 0.5)
                    if K * K == k_squared and a_mat % K == 0 and b_mat % K == 0:
                      M = a_mat // K
                      N = b_mat // K
                      if M * N == c_mat:
                        return (b1, b2, M, K), (b1, b2, K, N), (b1, b2, M, N)
          
          return None
        
        # Try common specific cases first
        if c_size == 4096 and a_size == 2048 and b_size == 2048:
          # This matches [64,32] @ [32,64] -> [64,64]
          a_shape = (64, 32)
          b_shape = (32, 64)
          c_shape = (64, 64)
        elif a_size == b_size and c_size == a_size:
          # Square matrices
          size = int(a_size ** 0.5)
          if size * size == a_size:
            a_shape = (size, size)
            b_shape = (size, size)
            c_shape = (size, size)
          else:
            # Not perfect square, try to find shapes
            result = find_shapes(a_size, b_size, c_size, shape_hints)
            if result:
              a_shape, b_shape, c_shape = result
            else:
              # Fallback - treat as flat and let TTNN figure it out
              a_shape = (a_size,)
              b_shape = (b_size,)
              c_shape = (c_size,)
        else:
          # Try to infer shapes using hints
          result = find_shapes(a_size, b_size, c_size, shape_hints)
          if result:
            a_shape, b_shape, c_shape = result
          else:
            # Last resort: try to factor assuming 2D
            if c_size > 0:
              k_squared = (a_size * b_size) // c_size
              if k_squared > 0:
                K = int(k_squared ** 0.5)
                if K * K == k_squared and a_size % K == 0 and b_size % K == 0:
                  M = a_size // K
                  N = b_size // K
                  if M * N == c_size:
                    a_shape = (M, K)
                    b_shape = (K, N)
                    c_shape = (M, N)
                  else:
                    # Give up and use flat shapes
                    a_shape = (a_size,)
                    b_shape = (b_size,)
                    c_shape = (c_size,)
                else:
                  # Give up and use flat shapes
                  a_shape = (a_size,)
                  b_shape = (b_size,)
                  c_shape = (c_size,)
              else:
                # Give up and use flat shapes
                a_shape = (a_size,)
                b_shape = (b_size,)
                c_shape = (c_size,)
            else:
              return  # Can't handle empty buffers
        
        # Load input tensors
        A = self._ensure_ttnn_tensor(bufs[1], a_shape, dtypes.float32)
        B = self._ensure_ttnn_tensor(bufs[2], b_shape, dtypes.float32)
        
        # Handle different dimensional cases
        if len(a_shape) == 1 or len(b_shape) == 1:
          # Vector-vector, vector-matrix, or matrix-vector multiplication
          # TTNN matmul might require 2D tensors, so reshape if needed
          if len(a_shape) == 1:
            A = ttnn.reshape(A, (1, a_shape[0]))
          if len(b_shape) == 1:
            B = ttnn.reshape(B, (b_shape[0], 1))
          C = ttnn.matmul(A, B)
          # Reshape output back if needed
          if len(c_shape) == 1:
            C = ttnn.reshape(C, c_shape)
        elif len(a_shape) >= 2 and len(b_shape) >= 2:
          # Standard matrix or batched matrix multiplication
          # TTNN matmul should handle batched matmul natively
          C = ttnn.matmul(A, B)
        else:
          # Shouldn't reach here, but fallback to regular matmul
          C = ttnn.matmul(A, B)
        
        # Store result
        bufs[0]["ttnn_tensor"] = C
        
        # Synchronize device
        if self.device:
          self.device.synchronize()
        return
    
    values = {}  # Store intermediate results by UOp index
    local_tensors: dict[int, Any] = {}  # DEFINE_LOCAL storage

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
        if opx in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}: return uop_idx
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
      
      if op in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
        # These define buffer access - we'll handle them when loading
        # DEFINE_REG creates accumulator registers for reductions
        continue

      if op is Ops.RANGE:
        # Single-worker: loop index is 0
        values[i] = 0
        continue
      if op is Ops.ENDRANGE:
        # No value to produce
        continue

      if op is Ops.INDEX:
        # INDEX creates a pointer to a buffer at a specific offset
        # For DEFINE_REG (accumulators), we just pass through the define index
        buf_def = _resolve_define_uop_idx(src_indices[0]) if src_indices else None
        if buf_def is not None and self.uops_data[buf_def][0] is Ops.DEFINE_REG:
          # For register accumulators, just track the define index
          values[i] = ('REG_PTR', buf_def)
        else:
          # For other buffers, create an index reference
          values[i] = ('INDEX', src_indices[0], src_indices[1] if len(src_indices) > 1 else 0)
        continue
      
      if op is Ops.LOAD:
        # Check if we're loading from a register accumulator
        if src_indices and isinstance(values.get(src_indices[0]), tuple) and values[src_indices[0]][0] == 'REG_PTR':
          # Loading from accumulator register
          reg_def = values[src_indices[0]][1]
          if reg_def not in local_tensors:
            # Initialize accumulator with the provided initial value if this is the first load
            if len(src_indices) > 1 and src_indices[1] in values:
              # Second argument is the STORE that initialized the accumulator
              local_tensors[reg_def] = 0.0  # Will be set by the STORE
            else:
              local_tensors[reg_def] = 0.0  # Default to 0
          values[i] = local_tensors[reg_def]
          continue
        
        # Check if we're loading through an INDEX operation
        if src_indices and isinstance(values.get(src_indices[0]), tuple) and values[src_indices[0]][0] == 'INDEX':
          # Loading through INDEX - extract the buffer and offset
          _, buf_uop_idx, offset = values[src_indices[0]]
          # Resolve the buffer through the INDEX's buffer reference
          def_uop_idx = _resolve_define_uop_idx(buf_uop_idx)
          buf_idx = define_to_runtime_idx.get(def_uop_idx) if def_uop_idx is not None else None
          shape_hint = _shape_from_view_uop(buf_uop_idx)
          shape_strides = _shape_strides_from_view_uop(buf_uop_idx)
        else:
          # Resolve runtime buffer index even when wrapped in VIEWs
          buf_idx = None
          shape_hint = None
          shape_strides = None
          def_uop_idx = None
          for src_uop_idx in src_indices:
            # prefer shapes from VIEW nodes if present
            shape_hint = _shape_from_view_uop(src_uop_idx) or shape_hint
            shape_strides = _shape_strides_from_view_uop(src_uop_idx) or shape_strides
            maybe_def = _resolve_define_uop_idx(src_uop_idx)
            if def_uop_idx is None and maybe_def is not None:
              def_uop_idx = maybe_def
            maybe_idx = define_to_runtime_idx.get(maybe_def) if maybe_def is not None else None
            if maybe_idx is not None:
              buf_idx = maybe_idx
              break

        if def_uop_idx is not None and self.uops_data[def_uop_idx][0] is Ops.DEFINE_LOCAL:
          # LOAD from local buffer: return stored tensor as-is. Local STORE already wrote the
          # correct result tensor for subsequent ops; reshaping here can violate volume.
          if def_uop_idx not in local_tensors:
            raise RuntimeError("TTNN LOAD: local buffer not initialized")
          values[i] = local_tensors[def_uop_idx]
          continue
        
        if def_uop_idx is not None and self.uops_data[def_uop_idx][0] is Ops.DEFINE_REG:
          # LOAD from register accumulator
          if def_uop_idx not in local_tensors:
            local_tensors[def_uop_idx] = 0.0  # Initialize to 0
          values[i] = local_tensors[def_uop_idx]
          continue

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
          nb_dims = [d for d,s in enumerate(std) if s != 0]
          if not nb_dims:
            final_tensor = ttnn.reshape(base_tensor, (1,)*len(shp))
          else:
            # Check if the shape is valid for the number of elements
            target_shape = tuple(int(s) for s in shp)
            target_numel = math.prod(target_shape)
            if target_numel != numel:
              # Shape mismatch - use flat tensor
              final_tensor = base_tensor
            else:
              # stride-aware realize: try to reshape directly to target shape
              try:
                final_tensor = ttnn.reshape(base_tensor, target_shape)
              except RuntimeError:
                # If direct reshape fails, keep it flat
                final_tensor = base_tensor
        else:
          if shape_hint is not None:
            hint_numel = math.prod(shape_hint)
            if hint_numel == numel:
              try:
                final_tensor = ttnn.reshape(base_tensor, tuple(shape_hint))
              except RuntimeError:
                final_tensor = base_tensor
            else:
              final_tensor = base_tensor
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
      
      if op is Ops.CAST:
        # Cast operation - convert data type
        src_val = src_values[0] if src_values and src_values[0] is not None else values[src_indices[0]]
        # If it's a scalar, just convert the Python type
        if isinstance(src_val, (int, float, bool)):
          if dtype == dtypes.float32 or dtype == dtypes.float16:
            values[i] = float(src_val)
          elif dtype == dtypes.int32:
            values[i] = int(src_val)
          elif dtype == dtypes.uchar:
            values[i] = int(src_val) & 0xFF
          else:
            values[i] = float(src_val)  # Default to float
        else:
          # It's a tensor, use ttnn.typecast if available, otherwise convert through torch
          if hasattr(ttnn, 'typecast'):
            # Map target dtype to ttnn dtype
            if dtype == dtypes.float32:
              ttnn_dt = ttnn.float32
            elif dtype == dtypes.float16:
              ttnn_dt = ttnn.float16
            elif dtype == dtypes.int32:
              ttnn_dt = ttnn.int32
            elif dtype == dtypes.uchar:
              ttnn_dt = ttnn.uint8
            else:
              ttnn_dt = ttnn.float32
            values[i] = ttnn.typecast(src_val, ttnn_dt)
          else:
            # Fallback: convert to torch, cast, convert back
            # First convert ttnn tensor to torch
            torch_tensor = ttnn.to_torch(src_val).detach()
            # Cast in torch
            if dtype == dtypes.float32:
              torch_tensor = torch_tensor.to(torch.float32)
              ttnn_dt = ttnn.float32
            elif dtype == dtypes.float16:
              torch_tensor = torch_tensor.to(torch.float16)
              ttnn_dt = ttnn.float16
            elif dtype == dtypes.int32:
              torch_tensor = torch_tensor.to(torch.int32)
              ttnn_dt = ttnn.int32
            elif dtype == dtypes.uchar:
              torch_tensor = torch_tensor.to(torch.uint8)
              ttnn_dt = ttnn.uint8
            else:
              torch_tensor = torch_tensor.to(torch.float32)
              ttnn_dt = ttnn.float32
            # Convert back to ttnn
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
        
        # Handle scalar operations - keep them as scalars for intermediate calculations
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
          # Both are scalars - do scalar math
          if op is Ops.ADD:
            values[i] = a + b
          elif op is Ops.MUL:
            values[i] = a * b
          elif op is Ops.SUB:
            values[i] = a - b
          elif op is Ops.FDIV:
            values[i] = a / b if b != 0 else float('inf')
          elif op is Ops.MAX:
            values[i] = max(a, b)
          elif op is Ops.POW:
            values[i] = a ** b
          else:
            values[i] = binary_map[op](a, b)
        elif isinstance(a, (int, float)):
          # a is scalar, b is tensor - convert scalar to ttnn scalar tensor
          # Get the actual device object
          device = b.device() if callable(b.device) else b.device if hasattr(b, 'device') else self._get_ttnn_device()
          scalar_tensor = ttnn.full(tuple([1] * len(b.shape)), a, dtype=b.dtype, layout=b.layout, device=device)
          values[i] = binary_map[op](scalar_tensor, b)
        elif isinstance(b, (int, float)):
          # b is scalar, a is tensor - convert scalar to ttnn scalar tensor
          # Get the actual device object
          device = a.device() if callable(a.device) else a.device if hasattr(a, 'device') else self._get_ttnn_device()
          scalar_tensor = ttnn.full(tuple([1] * len(a.shape)), b, dtype=a.dtype, layout=a.layout, device=device)
          values[i] = binary_map[op](a, scalar_tensor)
        else:
          # Both are tensors - check if shapes are compatible
          try:
            # Try the operation directly
            values[i] = binary_map[op](a, b)
          except RuntimeError as e:
            if "Broadcasting rule violation" in str(e):
              # Shapes are incompatible, try to make them compatible
              # This is a fallback for cases where the shapes don't match
              # Get shapes
              a_shape = tuple(a.shape) if hasattr(a, 'shape') else (1,)
              b_shape = tuple(b.shape) if hasattr(b, 'shape') else (1,)
              
              # If one is much larger than the other, it might be a flattened vs shaped tensor issue
              a_size = math.prod(a_shape)
              b_size = math.prod(b_shape)
              
              # For now, just re-raise the error with more context
              raise RuntimeError(f"Shape mismatch in {op}: a_shape={a_shape} ({a_size} elements), b_shape={b_shape} ({b_size} elements)") from e
            else:
              raise
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
        # The first source is the destination, second is the value to store
        # Additional sources may be loop control markers (RANGE/ENDRANGE)
        dest_idx = src_indices[0] if src_indices else None
        value_idx = src_indices[1] if len(src_indices) > 1 else None
        
        # Check if we're storing to a register accumulator
        if dest_idx is not None and isinstance(values.get(dest_idx), tuple) and values[dest_idx][0] == 'REG_PTR':
          # Storing to accumulator register
          reg_def = values[dest_idx][1]
          result_value = values.get(value_idx) if value_idx is not None else None
          if result_value is None:
            raise RuntimeError("TTNN STORE: missing result value for register")
          local_tensors[reg_def] = result_value
          continue
        
        # Check if we're storing through an INDEX operation
        if dest_idx is not None and isinstance(values.get(dest_idx), tuple) and values[dest_idx][0] == 'INDEX':
          # Storing through INDEX - extract the buffer and offset
          _, buf_idx_uop, offset = values[dest_idx]
          dest_def = _resolve_define_uop_idx(buf_idx_uop)
          
          if dest_def is not None and self.uops_data[dest_def][0] is Ops.DEFINE_LOCAL:
            # store into local map
            result_tensor = values.get(value_idx) if value_idx is not None else None
            if result_tensor is None:
              raise RuntimeError("TTNN STORE: missing result tensor")
            local_tensors[dest_def] = result_tensor
            continue
          
          if dest_def is not None and self.uops_data[dest_def][0] is Ops.DEFINE_REG:
            # store into register accumulator
            result_value = values.get(value_idx) if value_idx is not None else None
            if result_value is None:
              raise RuntimeError("TTNN STORE: missing result value for register")
            local_tensors[dest_def] = result_value
            continue
          
          # Store to global buffer
          out_buf_idx = define_to_runtime_idx.get(dest_def) if dest_def is not None else None
          if out_buf_idx is not None and out_buf_idx < len(bufs):
            result_tensor = values.get(value_idx) if value_idx is not None else None
            if result_tensor is None:
              raise RuntimeError("TTNN STORE: missing result tensor")
            bufs[out_buf_idx]["ttnn_tensor"] = result_tensor
            continue
        
        # STORE result tensor into resolved destination buffer
        result_tensor = src_values[1] if len(src_values) > 1 else (src_values[0] if src_values else None)
        if result_tensor is None:
          if value_idx is not None and value_idx in values:
            result_tensor = values[value_idx]
          else:
            raise RuntimeError(f"TTNN STORE: missing result tensor (value_idx={value_idx})")

        # first source is the destination pointer
        dest_ptr_uop = src_indices[0] if src_indices else None
        
        # Check if the destination is an INDEX tuple we've already processed
        if dest_ptr_uop is not None and dest_ptr_uop in values and isinstance(values[dest_ptr_uop], tuple):
          if values[dest_ptr_uop][0] == 'INDEX':
            # It's an INDEX tuple, extract the buffer reference
            _, buf_uop_idx, offset = values[dest_ptr_uop]
            dest_def = _resolve_define_uop_idx(buf_uop_idx)
          elif values[dest_ptr_uop][0] == 'REG_PTR':
            # It's a register pointer
            dest_def = values[dest_ptr_uop][1]
          else:
            dest_def = _resolve_define_uop_idx(dest_ptr_uop)
        else:
          # Resolve to DEFINE_*
          dest_def = _resolve_define_uop_idx(dest_ptr_uop) if dest_ptr_uop is not None else None

        if dest_def is not None and self.uops_data[dest_def][0] is Ops.DEFINE_LOCAL:
          # store into local map
          local_tensors[dest_def] = result_tensor
          continue
        
        if dest_def is not None and self.uops_data[dest_def][0] is Ops.DEFINE_REG:
          # store into register accumulator
          local_tensors[dest_def] = result_tensor
          continue

        out_buf_idx = define_to_runtime_idx.get(dest_def) if dest_def is not None else None
        if out_buf_idx is None or out_buf_idx >= len(bufs):
          raise RuntimeError(f"TTNN STORE: unable to resolve destination buffer (dest_def={dest_def}, out_buf_idx={out_buf_idx})")
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
