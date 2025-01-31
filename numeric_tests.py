import torch
import math
from torch.nn import init
import torch.nn.functional as F
import fused_mlp_module
import numpy as np

def initialize_weight(weight: torch.Tensor) -> torch.Tensor:
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    return weight

# Configuration
MNK_values = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
ITERATIONS = 100 

header = (
    f"{'MNK':>8} | "
    f"{'Metric':>15} | "
    f"{'20th Percentile':>15} | "
    f"{'80th Percentile':>15} | "
    f"{'Mean':>15} | "
    f"{'Std Dev':>15}"
)
print(header)
print("-" * len(header))

for MNK in MNK_values:
    M = N = K = MNK

    max_abs_diffs = []
    mean_abs_diffs = []
    relative_diffs = []
    
    for _ in range(ITERATIONS):
        A = torch.randn((M, K), device=0, dtype=torch.bfloat16)
        B = torch.randn((N, K), device=0, dtype=torch.bfloat16)
        A = initialize_weight(A)
        B = initialize_weight(B)

        # Torch computation
        C_torch = A @ B.T
        silu = F.silu(C_torch[:, 1::2])
        gated_torch = C_torch[:, ::2] * silu 

        # Custom kernel computation
        gated_mine = fused_mlp_module.fused_swiglu_bf16(A, B)

        # Compute differences
        diff = gated_mine - gated_torch
        max_abs_diffs.append(diff.abs().max().item())
        mean_abs_diffs.append(diff.abs().mean().item())
        relative_diffs.append(diff.norm(p='fro').item() / gated_torch.norm(p='fro').item())
    
    max_abs_diffs = np.array(max_abs_diffs)
    mean_abs_diffs = np.array(mean_abs_diffs)
    relative_diffs = np.array(relative_diffs)
    
    stats = {
        "Max Abs Diff": max_abs_diffs,
        "Mean Abs Diff": mean_abs_diffs,
        "Relative Diff": relative_diffs
    }
    
    for metric, values in stats.items():
        percentile_20 = np.percentile(values, 20)
        percentile_80 = np.percentile(values, 80)
        mean_val = values.mean()
        std_val = values.std()
        print(
            f"{MNK:>8} | "
            f"{metric:>15} | "
            f"{percentile_20:15.6e} | "
            f"{percentile_80:15.6e} | "
            f"{mean_val:15.6e} | "
            f"{std_val:15.6e}"
        )
