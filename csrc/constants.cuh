#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <type_traits>
#include <cute/numeric/numeric_types.hpp>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#define C_TYPE cute::bfloat16_t
using C_AT_TYPE = std::conditional_t<
    std::is_same<C_TYPE, cute::bfloat16_t>::value,
    at::BFloat16,
    C_TYPE
>;

#endif // CONSTANTS_H