# Add rewrite step for specials
_Exported on 8/20/2025 at 14:45:19 PDT from Cursor (1.4.5)_

---

**User**

Let's add a rewrite step that replaces specials with a RANGE that records the SPECIAL's extent. Put it into the rewrite pipeline if we're on TTNN. Do it at the very end.

---

**Cursor**

I'll scan the codebase to locate the rewrite pipeline, how SPECIAL and RANGE UOps are modeled, and where TTNN-specific rewrites are activated, so I can append a final rewrite step only for TTNN.

I'll open the rewrite pipeline definition and relevant files to see the exact structure and where to insert a TTNN-only final step, plus check how SPECIAL and RANGE are used so the transformation is correct.

I'll add a TTNN-only final rewrite that converts `Ops.SPECIAL` to `Ops.RANGE` with the SPECIAL's size as the RANGE extent, and wire it into the end of the rewrite pipeline (before linearizer).

```python
from typing import Any, Callable
import functools
from dataclasses import dataclass
from tinygrad.helpers import QUANTIZE, DEVECTORIZE, TRANSCENDENTAL
from tinygrad.uop.ops import PatternMatcher, graph_rewrite, UOp, UPat
from tinygrad.uop import Ops
from tinygrad.uop.spec import type_verify
from tinygrad.renderer import Renderer

# import all pattern matchers here
from tinygrad.codegen.lowerer import pm_lowerer, get_index
from tinygrad.codegen.quantize import pm_quant
from tinygrad.codegen.gpudims import pm_add_gpudims
from tinygrad.uop.symbolic import sym, symbolic_simple, gep_pushing
from tinygrad.uop.decompositions import get_late_rewrite_patterns
from tinygrad.codegen.expander import migrate_indexing, expander
from tinygrad.codegen.devectorizer import load_store_folding, load_store_indexing, devectorize, pm_reduce, \
  ReduceContext, correct_load_store, pm_render
from tinygrad.codegen.linearize import block_create, pm_blockend_merge, block_merge, pm_finalize, BlockContext
from tinygrad.codegen.opt import pm_optimize
from tinygrad.codegen.opt.swizzler import view_left, view_right, fix_kernel_ops

@dataclass
class RewriteStep:
  pm: PatternMatcher
  ctx: Callable[[UOp], Any]|None = None
  name: str|None = None
  bottom_up: bool = False
  def __call__(self, sink:UOp):
    return graph_rewrite(sink, self.pm, ctx=self.ctx(sink) if self.ctx is not None else None, name=self.name, bottom_up=self.bottom_up)

def apply_rewrites(sink:UOp, rewrites:list[RewriteStep]): return functools.reduce(lambda x,f: f(x), rewrites, sink)

rewrites_for_views = [
  RewriteStep(view_left, name="Main View Left"),
  RewriteStep(view_right, name="Main View Right"),
  RewriteStep(view_left+fix_kernel_ops, bottom_up=True, name="Finalize Kernel"),
]

rewrites_for_linearizer = [
  RewriteStep(block_create, ctx=BlockContext.from_sink, name="Linearizer: Create Blocks", bottom_up=True),
  RewriteStep(pm_blockend_merge, name="Linearizer: Merge Blockends"),
  RewriteStep(block_merge, name="Linearizer: Merge Blocks"),
  RewriteStep(pm_finalize, name="Linearizer: Finalize")]

# Replace SPECIAL with RANGE carrying the SPECIAL's extent as RANGE end (src[0])
pm_special_to_range = PatternMatcher([
  (UPat(Ops.SPECIAL, name="x"), lambda x: UOp.range(x.dtype, x.arg[1], id(x)))
])

def get_rewrites_for_renderer(opts:Renderer, linearizer:bool=True) -> list[RewriteStep]:
  # cache with the values of the context vars
  return _get_rewrites_for_renderer(opts, linearizer, QUANTIZE.value, DEVECTORIZE.value, TRANSCENDENTAL.value)

@functools.cache
def _get_rewrites_for_renderer(opts:Renderer, linearizer:bool, _QUANTIZE, _DEVECTORIZE, _TRANSCENDENTAL) -> list[RewriteStep]:
  # ** lowerer (rewrite_shapetracker_with_index) **
  ret: list[RewriteStep] = []

  # view pushing
  ret.extend(rewrites_for_views)

  # this is kernel.py
  ret.append(RewriteStep(pm_optimize, ctx=lambda _: opts, name="optimize ast"))

  if getattr(opts, 'device', None) != "TTNN":
    if _QUANTIZE and opts.device in {"CPU", "DSP"}: ret.append(RewriteStep(pm_quant, name="quantize"))
    ret.append(RewriteStep(pm_lowerer, get_index, name="lowerer", bottom_up=True))

    # ** expander (expand_rewrite) **
    ret.append(RewriteStep(sym+migrate_indexing, name="initial symbolic"))

    # expand
    ret.append(RewriteStep(sym+expander, name="expander"))

    # ** devectorizer (full_graph_rewrite) **
    # remove reduce
    ret.append(RewriteStep(pm_reduce+gep_pushing, lambda _: ReduceContext(), name="remove_reduce"))

    # add gpu dims (late) — skip for TTNN which runs single-worker without SPECIAL

    ret.append(RewriteStep(pm_add_gpudims, lambda _: opts, name="add gpudims"))

    # devectorize (TODO: does this need opts?)
    if _DEVECTORIZE >= 2: pm_devectorize = sym+load_store_folding+load_store_indexing
    elif _DEVECTORIZE: pm_devectorize = sym+devectorize+load_store_folding+correct_load_store+load_store_indexing
    else: pm_devectorize = sym+load_store_folding+correct_load_store+load_store_indexing
    ret.append(RewriteStep(pm_devectorize, lambda _: opts, name="devectorize"))

    supported_ops = tuple(opts.code_for_op.keys())
    extra_matcher = opts.extra_matcher if opts.extra_matcher is not None else PatternMatcher([])

    # optional pre matcher
    if opts.pre_matcher is not None: ret.append(RewriteStep(opts.pre_matcher, name="pre_matcher"))

    # decompositions
    pm_decomp = symbolic_simple+get_late_rewrite_patterns(supported_ops, _TRANSCENDENTAL>=2)
    ret.append(RewriteStep(pm_decomp, name="decompositions"))

    # final rules for the renderer (without sym)
    pm_final_rewrite = pm_decomp+pm_render+extra_matcher
    ret.append(RewriteStep(pm_final_rewrite, lambda _: opts.device, name="final rewrite"))

  # TTNN-specific: at the very end, replace SPECIAL with RANGE recording extent
  if getattr(opts, 'device', None) == "TTNN":
    ret.append(RewriteStep(pm_special_to_range, name="ttnn: specials to ranges"))

  # return the list (with optional linearizer)
  return ret + (rewrites_for_linearizer if linearizer else [])

def full_rewrite_to_sink(sink:UOp, opts:Renderer|None=None, linearizer:bool=False) -> UOp:
  return apply_rewrites(sink, get_rewrites_for_renderer(opts if opts is not None else Renderer(), linearizer))

def full_rewrite(sink:UOp, opts:Renderer|None=None) -> list[UOp]:
  """
  Function to transform the Kernel UOp graph into a linearized program.

  Args:
    sink: The Ops.SINK rooting the Kernel graph.
    opts: The Renderer (can change how things are processed, fix this).

  Returns:
    Linear program in UOps.
  """

  lst = list(full_rewrite_to_sink(sink, opts, linearizer=True).arg.lst)
  if __debug__: type_verify(lst)
  return lst

```

