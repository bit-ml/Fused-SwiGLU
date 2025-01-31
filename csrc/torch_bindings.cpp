#ifndef OPS_H
#define OPS_H


#include <torch/all.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>

#include "constants.cuh"


template<typename TA, typename TB, typename TC>
void swiglu_fwd(int64_t m, int64_t n, int64_t k, 
                torch::Tensor const& A, int64_t ldA,
                torch::Tensor const& B, int64_t ldB,
                torch::Tensor      & C, int64_t ldC);
extern template void swiglu_fwd<cute::bfloat16_t, cute::bfloat16_t, C_TYPE>(
    int64_t m, int64_t n, int64_t k, 
    torch::Tensor const& A, int64_t ldA,
    torch::Tensor const& B, int64_t ldB,
    torch::Tensor      & C, int64_t ldC);

at::Tensor fused_swiglu_bf16(torch::Tensor const& X, torch::Tensor const& W1W3); 

namespace Fused_MLP {

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("fused_swiglu_bf16", &fused_swiglu_bf16, "fused SwiGLU for bf16");

        // templates
        m.def("swiglu_fwd_bf16", &swiglu_fwd<cute::bfloat16_t, cute::bfloat16_t, C_TYPE>, "SwiGLU fwd bf16");
    }

    TORCH_LIBRARY(ASMLLM, m) {
        m.def("fused_swiglu_bf16(Tensor X, Tensor W1W3) -> Tensor");

        // templates
        m.def("swiglu_fwd_bf16(int m, int n, int k, Tensor A, int ldA, Tensor B, int ldB, Tensor C, int ldC) -> ()");
    } 

    TORCH_LIBRARY_IMPL(ASMLLM, CUDA, m) {
        m.impl("fused_swiglu_bf16", &fused_swiglu_bf16);

        // templates 
        m.impl("swiglu_fwd_bf16", &swiglu_fwd<cute::bfloat16_t, cute::bfloat16_t, C_TYPE>);
    }
}
#endif // OPS_H