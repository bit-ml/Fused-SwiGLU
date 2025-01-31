import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import torch.nn as nn
import fused_mlp_module
from unsloth_no_bs import swiglu_fg_kernel

def bit_fwd(A, B, C, M, N, K):
    fused_mlp_module.swiglu_fwd_bf16(M, N, K, A, K, B, K, C, N)

@torch.inference_mode()
def vanilla_gemm_fwd(A, B, C):
    torch.mm(A, B, out=C[0])

@torch.inference_mode()
def module_swig_fwd(swig, C):
    return swig(C)

class Swiglu(nn.Module):
    def __init__(self, M, N, K):
        super().__init__()
        self.M = M
        self.N = N
        self.x = torch.empty((M, K), dtype=torch.bfloat16, device=0)
        self.w = torch.empty((K, N), dtype=torch.bfloat16, device=0)

    def forward(self, C):
        torch.matmul(self.x, self.w, out=C)
        return swiglu_fg_kernel(C[:, :self.N//2], C[:, self.N//2:])

@torch.inference_mode()
def eager_swiglu_fwd(A, B, C, M, N):
    torch.mm(A, B, out=C)
    return swiglu_fg_kernel(C[:, :N//2], C[:, N//2:])  


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['Tokens'], 
        x_vals=[1024,
                2048,
                4096,
                8192,
                16384,
                32768,
                49152,
                65536],
        line_arg='provider',
        line_vals=[
            'cuBLAS+Unsloth',
            'Fused',
        ],
        line_names=[
            "cuBLAS+Unsloth",
            "Fused",
        ],
        styles=[('#17becf', '-', 'o'), ('#d62728', '-', 's')], 
        ylabel="TFLOP/s", 
        plot_name="Gated MLP for Llama 70B",
        args={}
    ))
def benchmark(Tokens, provider):
    K = 8192
    M = N = Tokens
    N = 28672*2
    device = 0
    A = torch.randn((M, K), dtype=torch.bfloat16, device=device) 
    B = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    BT = B.T
    C = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    C2 = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    # torch.cuda.empty_cache()
    swig = Swiglu(M, N, K)
    swig.x = A.view(1, M, K)
    swig.w = BT

    quantiles = [0.5, 0.2, 0.8]
    if provider == "cuBLAS+Unsloth":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: eager_swiglu_fwd(A, BT, C2, M, N), quantiles=quantiles)
    elif provider == "Fused":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_mlp_module.swiglu_fwd_bf16(M, N, K, A, K, B, K, C, N), quantiles=quantiles)
    # flops for matmul are MN(2K-1), and then the glu is
    # sigmoid(C) = (1+exp(-C))^-1 - this is 4 flops
    # silu(C) = C * sigmoid(C) - 1 flop
    # C * silu(C) - 1 flop
    # therefore, 6 flops per element for swiglu
    flops = lambda ms: ((M*N*(2*K-1)*1e-12)+(6*M*(N//2)*1e-12))/(ms *1e-3)
    print(ms, provider, Tokens)
    return flops(ms), flops(max_ms), flops(min_ms)
    

benchmark.run(show_plots=True, print_data=True, save_path='.')