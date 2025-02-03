"""
This just changes the Unsloth kernel to not use a batch size. 
This can easily be circumvented by applying a view on the tensors before calling the Unsloth kernels, the reason
I'm using a custom kernel is that including Unsloth in the benchmark script has some strange effects on the benchmark results. The rankings are not changed and the performance differences are minimally impacted, but for certain matrix dimensions both cuBLAS+Unsloth and the custom kernel exhibit weird dips in performance.
It is unclear to me if this is related to a stranger hardware issue in our cloud or something that Unsloth does at initialization.
"""

import triton.language as tl
import triton 
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE : tl.constexpr,):
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask = mask, other = 0).to(tl.float32)
    g_row = tl.load(g + offsets, mask = mask, other = 0)#.to(tl.float32)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row) # e_row / (1 + tl.exp(-e_row))
    f_row = f_row.to(g_row.dtype) # Exact copy from HF
    # h = f * g
    h_row = f_row * g_row

    # Store h
    tl.store(h + offsets, h_row, mask = mask)
pass


def swiglu_fg_kernel(e, g):
    seq_len, hd = e.shape
    n_elements = e.numel()
    # h = torch.empty((seq_len, hd), dtype = e.dtype, device = "cuda:0")
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fg_kernel[grid](e, g, g, n_elements, BLOCK_SIZE = 1024,)
    return g
pass