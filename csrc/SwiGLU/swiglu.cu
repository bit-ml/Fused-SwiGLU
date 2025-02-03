#ifndef SWIGLU_H
#define SWIGLU_H

/***
 * The code, mostly in the GEMM part, is heavily based on the CuTe GEMM tutorial and 
 * several very helpful issues on the CUTLASS repo. A non-exhaustive list of the tutorials and repos used is:
 * - CuTe tutorials: https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute
 * - A100 GEMM half precision issue: https://github.com/NVIDIA/cutlass/issues/1905
 * - issue on predicates: https://github.com/NVIDIA/cutlass/issues/1886
*/

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cute/tensor.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits.hpp>

#include "../constants.cuh"
#include "../gated_mma_16x4_atom.cuh"

namespace Fused_MLP {

  template <class ElementA, class ElementB,
            class SmemLayoutA, class SmemLayoutB>
  struct SharedStorage
  {
    cute::array<ElementA, cute::cosize_v<SmemLayoutA>> A;
    cute::array<ElementB, cute::cosize_v<SmemLayoutB>> B;
  };


  // inlined conversion from fp32, fp16 or bf16 to either one
  template<typename To_type, typename From_type>
  __forceinline__ __device__ To_type cast_mmaresult(const From_type el) {
    bool constexpr no_cast = std::is_same<From_type, To_type>::value;
    if constexpr(no_cast) {
      return el; 
    }
    bool constexpr to_float = std::is_same<To_type, float>::value;
    bool constexpr to_fp16 = std::is_same<To_type, cute::half_t>::value;
    bool constexpr to_bf16 = std::is_same<To_type, cute::bfloat16_t>::value;

    bool constexpr from_float = std::is_same<From_type, float>::value;
    bool constexpr from_fp16 = std::is_same<From_type, cute::half_t>::value;
    bool constexpr from_bf16 = std::is_same<From_type, cute::bfloat16_t>::value;

    if constexpr (to_float) {
      return static_cast<float>(el);
    }
    else if constexpr (to_fp16) {
      if constexpr (from_float) {
        return static_cast<cute::half_t>(el);
      }
      if constexpr(from_bf16) {
        return static_cast<cute::half_t>(__half(static_cast<const __nv_bfloat16>(el)));
      }
    }
    else if constexpr(to_bf16) {
      if constexpr (from_float) {
        // return static_cast<cute::bfloat16_t>(el);
        return cute::bfloat16_t(el);
      }
      if constexpr (from_fp16) {
        return static_cast<cute::bfloat16_t>(__nv_bfloat16(static_cast<const __half>(el)));
      }
    }
  
  } 

  template<typename Precision>
  __forceinline__ __device__ Precision silu(Precision el) {
    bool constexpr is_float = std::is_same<Precision, float>::value;
    float local_copy;
    if constexpr(!is_float) { // on A100 this won't happen anyway, all bf16 mmas accumulate in fp32
      local_copy = cast_mmaresult<float>(el);
    }
    else {
      local_copy = el;
    }

    // not sure if an overflow check is warranted here?
    local_copy = __fdividef(local_copy, 1 + __expf(-local_copy));
    if constexpr (is_float) {
      return local_copy;
    }
    return cast_mmaresult<Precision>(local_copy);
  }

  template <class ProblemShape, class ProblemHalfShape, class CtaTiler, class CtaHalfTiler,
            class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
            class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
            class TC, class CStride, class CStride_half, class CSmemLayout, class TiledMma, class TiledCopyC, class TiledMmaC,
            int L2_SWIZZLE>
  __global__ static
  __launch_bounds__(decltype(size(TiledMma{}))::value)
  void
  gemm_device(ProblemShape shape_MNK, ProblemHalfShape shape_MNK_half, CtaTiler cta_tiler, CtaHalfTiler half_tiler,
              TA const* __restrict__ A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_A,
              TB const* __restrict__ B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_B,
              TC      * __restrict__ C, CStride dC, CStride_half dC_half, CSmemLayout sC_layout, TiledMma mma,      TiledCopyC copy_c, TiledMmaC mma_copy,
              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<L2_SWIZZLE> block_swizzler, int log_tile)
  {
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_c) == size(mma));

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(half_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(half_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK_half), dA));         // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK_half), dB));         // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK_half), dC_half));         // dC strides for shape MN

    //
    // Full and Tiled Tensors
    //

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

    // threadblock swizzling\rasterization
    cutlass::gemm::GemmCoord block_idx = block_swizzler.get_tile_offset(log_tile);

    // get the appropriate blocks for this thread block
    auto cta_coord = make_coord(block_idx.m(), block_idx.n(), _);              // (m,n,k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    // Shared memory buffers
    __align__(128) extern __shared__ char byte_smem[];
    using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(byte_smem);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.data()), sA_layout);            // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.data()), sB_layout);            // (BLK_N,BLK_K,PIPE)

    //
    // Partition the copying of A and B tiles across the threads
    //

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

    Tensor mcA = make_identity_tensor(shape(mA));                        // predicate tensors
    Tensor mcB = make_identity_tensor(shape(mB));

    Tensor cA = local_tile(mcA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor cB = local_tile(mcB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

    // apply same partitioning to predicate
    Tensor tAcA = thr_copy_a.partition_S(cA); // (CPY, CPY_M, CPY_K) 
    Tensor tBcB = thr_copy_b.partition_S(cB); // (CPY, CPY_N, CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tAcA) == size<1>(tAsA));                // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAcA) == size<2>(tAsA));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBcB) == size<1>(tBsB));                // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBcB) == size<2>(tBsB));                // CPY_K
    //
    // PREFETCH
    //

    auto K_PIPE_MAX = size<3>(tAsA);

    // Total count of tiles
    int k_tile_count = size<3>(tAgA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;

    // Start async loads for all pipes but the last
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
      // predicates have this weird thing
      // for copy_if calls that use atoms, the copy_if actually skips iterating over the first rank of source and destionation
      // this makes sense in the context of CuTe, since the first rank is the Atom
      // this is why we index with Int<0>{}, otherwise the predicate would have more "elements" than the tensors in the copy_if
      // it is very important for the predicate and tensors to have the same amount of elements they can acces
      // smaller predicates will fail (i think even at compile time)
      // larger predicates lead to hard to debug illegal memory access errors
      auto pred_a = [&](auto... coords) {
        auto predicate = tAcA(Int<0>{}, _, _, k_tile_next);
        return elem_less(predicate(coords...), shape(mA));
      };
      auto pred_b = [&](auto... coords) {
        auto predicate = tBcB(Int<0>{}, _, _, k_tile_next);
        return elem_less(predicate(coords...), shape(mB));
      };
      
      copy_if(copy_a, pred_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
      copy_if(copy_b, pred_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) { ++k_tile_next; }
    }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));                // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));                // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)


  CUTE_STATIC_ASSERT_V((shape(tCrC) == take<0,3>(shape(tCgC))));     // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));              // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));              // MMA_N
  // Clear the accumulators
  clear(tCrC);
  
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_A, mma);

  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_B, mma);
  ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)

  // Current pipe index in smem to read from
  int smem_pipe_read  = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX-1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_REG_MAX = size<2>(tCrA);

  // PREFETCH register pipeline
  if (K_BLOCK_REG_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX-2>();    
    __syncthreads(); // <-- very important! the cp.async will write using the swizlled gmem->smem mapping, but smem->rmem is done with ldmatrix. so for example, given thread i, ldmatrix might request smem written by thread j at the gmem->smem stage. so cp.async.wait doesn't guarantee that thread j is ready at this point, only that thread i wrote something. 

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_A, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_B, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL
    for (int k_block_reg = 0; k_block_reg < K_BLOCK_REG_MAX; ++k_block_reg)
    {

      if (k_block_reg == K_BLOCK_REG_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_,_,_,smem_pipe_read); 
        tXsB_p = tXsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_reg_next = (k_block_reg + Int<1>{}) % K_BLOCK_REG_MAX;      // static
      #if 1
      copy(s2r_atom_A, tXsA_p(_,_,k_block_reg_next), tXrA(_,_,k_block_reg_next));
      copy(s2r_atom_B, tXsB_p(_,_,k_block_reg_next), tXrB(_,_,k_block_reg_next));
      #endif

      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block_reg == 0)
      {
        if(k_tile_count > 0) {
          auto pred_a = [&](auto... coords) {
            auto predicate = tAcA(Int<0>{}, _, _, k_tile_next);
            return elem_less(predicate(coords...), shape(mA));
          };
          auto pred_b = [&](auto... coords) {
            auto predicate = tBcB(Int<0>{}, _, _, k_tile_next);
            return elem_less(predicate(coords...), shape(mB));
          };
          copy_if(copy_a, pred_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
          copy_if(copy_b, pred_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        }
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) * 0 + (smem_pipe_read != K_PIPE_MAX) * smem_pipe_read;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block_reg), tCrB(_,_,k_block_reg), tCrC);
    }

  }

    //
    // Gating
    //

    // reuse allocated shared memory for C
    TC* __restrict__ smem_C = reinterpret_cast<TC*>(byte_smem);
    Tensor sC = make_tensor(make_smem_ptr(smem_C), sC_layout); // applies sC_layout (128x64 row-major) to shared memory
               
    
    Tensor mC_half = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK_half), dC_half); // (M,N//2)
  
    ThrCopy thr_copy_c = copy_c.get_slice(threadIdx.x);  // this feller actually uses the same memory for efficiency
    Tensor tCsC = thr_copy_c.partition_S(sC);                     // (CPY,CPY_M,CPY_N)
    Tensor mcC_half = make_identity_tensor(shape(mC_half));
    Tensor cC_half = local_tile(mcC_half, half_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N//2)
    Tensor gC_half = local_tile(mC_half,  half_tiler, cta_coord, Step<_1, _1, X>{});  // (BLK_M,BLK_N//2)
    Tensor tCcC      = thr_copy_c.partition_D(cC_half);
    Tensor tCgC_half = thr_copy_c.partition_D(gC_half);                // (CPY,CPY_M,CPY_N)     
    
    CUTE_STATIC_ASSERT_V(size<1>(tCsC) == size<1>(tCgC_half)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsC) == size<2>(tCgC_half)); // CPY_N/2
    CUTE_STATIC_ASSERT_V(size<1>(tCcC) == size<1>(tCgC_half));
    CUTE_STATIC_ASSERT_V(size<2>(tCcC) == size<2>(tCgC_half));


    ThrMMA thr_mma_copy = mma_copy.get_slice(threadIdx.x);
    Tensor tXsC      = thr_mma_copy.partition_C(sC);    // (MMA, MMA_M, MMA_N)
    Tensor tCrC_copy = thr_mma_copy.make_fragment_C(thr_mma_copy.partition_C(gC_half));  // (MMA, MMA_M, MMA_N)

    auto MMA_M = size<1>(tCrC);
    auto MMA_N = size<2>(tCrC);
    CUTE_UNROLL
    for(int j = 0; j < MMA_N; ++j) {
      CUTE_UNROLL
      for(int i = 0; i < MMA_M; ++i) {
        tCrC_copy[make_coord(0, i, j)] = cast_mmaresult<TC>(tCrC[make_coord(make_coord(0, 0), i, j)] * silu(tCrC[make_coord(make_coord(1, 0), i, j)]));
        tCrC_copy[make_coord(1, i, j)] = cast_mmaresult<TC>(tCrC[make_coord(make_coord(0, 1), i, j)] * silu(tCrC[make_coord(make_coord(1, 1), i, j)])); 
      }
    }


    copy(tCrC_copy, tXsC); // copy registers to shared memory, not vectorized tho (don't think it makes much of a difference anyway?)
    __syncthreads(); 

    auto pred_c = [&](auto... coords) {
      auto predicate = tCcC(Int<0>{}, _, _);
      return elem_less(predicate(coords...), shape(mC_half));
    };
    copy_if(copy_c, pred_c, tCsC, tCgC_half); // smem->gmem is vectorized now
  }
}

template<typename TA, typename TB, typename TC>
void swiglu_fwd(int64_t m, int64_t n, int64_t k,  
                torch::Tensor const& TsrA, int64_t ldA,
                torch::Tensor const& TsrB, int64_t ldB,
                torch::Tensor      & TsrC, int64_t ldC) {
    using namespace cute;

    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);
    auto prob_half_shape = make_shape(M, N/2, K);

    // strides
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(ldC, Int<1>{});
    auto dC_half = make_stride(ldC/2, Int<1>{});

    // CTA tile sizes
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bN_half = Int<64>{};
    auto bK = Int< 64>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto half_tiler = make_shape(bM, bN_half, bK);
    auto bP = Int<  3>{}; // pipeline

    auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,_64>,
                                         Stride<_64, _1>>{});
    auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
    auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
    auto sC = tile_to_shape(swizzle_atom, make_shape(bM, bN_half));


    // thread layouts
    TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, TA>{},
                                      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // thr layout 16x8
                                      Layout<Shape<_1, _8>>{}); // val layout 1x8 -> K-major

    TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, TB>{},
                                      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // thr layout 16x8
                                      Layout<Shape<_1, _8>>{}); // val layout 1x8 -> K-major

    // this will be used for the vectorized SMEM->GMEM store at the epilogue
    TiledCopy copyC = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t, uint128_t>, TC>{},
                                      Layout<Shape<_16, _8>, Stride<_8, _1>>{}, // thr layout 16x8
                                      Layout<Shape<_1, _8>>{}); // val layout 1x8 -> K-major

    TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{}, // 16x8x16 mma
                                 Layout<Shape<_2, _2>>{},    // 2x2x1 MMA Atoms
                                 Tile<_64,_16,_16>{});      // 64x16x16 Tiled MMA for LDSM

    
    TiledMMA mmaC_gating = make_tiled_mma(SM80_16x4x16_F32BF16BF16F32_TN_GATED_COPY{}, // 16x4x16 mma atom for gate
                                 Layout<Shape<_2, _2>>{},    // 2x2x1 MMA Atoms
                                 Tile<_64,_8,_16>{});      // 64x8x16 Tiled MMA for copying RMEM->SMEM and reducing for gating
  
    Copy_Atom<SM75_U32x4_LDSM_N, TA> s2r_atom_A;
    Copy_Atom<SM75_U32x2_LDSM_N, TB> s2r_atom_B; // this seems to work better than U32x4 for B


    // launch kernel
    dim3 dimBlock(size(mmaC));
    constexpr int L2_SWIZZLE = 4;
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<L2_SWIZZLE> block_swizzler; 
    auto tiled_shape = block_swizzler.get_tiled_shape(cutlass::gemm::GemmCoord({M, N, 0}),
                                                     cutlass::gemm::GemmCoord({bM, bN, 0}),
                                                     0);
    const int log_tile = block_swizzler.get_log_tile(tiled_shape);
    dim3 dimGrid = block_swizzler.get_grid_shape(tiled_shape);
    dimGrid = dim3(dimGrid.x, dimGrid.y);

    TA const* __restrict__ A = reinterpret_cast<TA*>(TsrA.data_ptr<at::BFloat16>());
    TB const* __restrict__ B = reinterpret_cast<TB*>(TsrB.data_ptr<at::BFloat16>());
    TC      * __restrict__ C = reinterpret_cast<TC*>(TsrC.data_ptr<C_AT_TYPE>());

    const at::cuda::OptionalCUDAGuard device_guard(device_of(TsrA));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const size_t smem_size = int(sizeof(Fused_MLP::SharedStorage<TA, TB, decltype(sA), decltype(sB)>)); // ~93 KB for 128x64 tiles

    using ProblemShape = decltype(prob_shape);
    using ProblemHalfShape = decltype(prob_half_shape);
    using CtaTiler = decltype(cta_tiler);
    using CtaHalfTiler = decltype(half_tiler);
    using AStride = decltype(dA);
    using ASmemLayout = decltype(sA);
    using TiledCopyA = decltype(copyA);
    using S2RAtomA = decltype(s2r_atom_A);
    using BStride = decltype(dB);
    using BSmemLayout = decltype(sB);
    using TiledCopyB = decltype(copyB);
    using S2RAtomB = decltype(s2r_atom_B);
    using TiledCopyC = decltype(copyC);
    using CStride = decltype(dC);
    using CStride_half = decltype(dC_half);
    using CSmemLayout = decltype(sC);
    using TiledMma = decltype(mmaC);
    using TiledMmaC = decltype(mmaC_gating);

    auto GemmDeviceKernel = Fused_MLP::gemm_device<
        ProblemShape, ProblemHalfShape, CtaTiler, CtaHalfTiler,
        TA , AStride, ASmemLayout, TiledCopyA, S2RAtomA,
        TB , BStride, BSmemLayout, TiledCopyB, S2RAtomB,
        TC , CStride, CStride_half, CSmemLayout, TiledMma, TiledCopyC, TiledMmaC,
        L2_SWIZZLE
    >;

    cudaFuncSetAttribute(GemmDeviceKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);


    GemmDeviceKernel<<<dimGrid, dimBlock, smem_size, stream>>>
        (prob_shape, prob_half_shape, cta_tiler, half_tiler,
        A, dA, sA, copyA, s2r_atom_A,
        B, dB, sB, copyB, s2r_atom_B,
        C, dC, dC_half, sC, mmaC, copyC, mmaC_gating,
        block_swizzler, log_tile);
}

template void swiglu_fwd<cute::bfloat16_t, cute::bfloat16_t, C_TYPE>(
    int64_t m, int64_t n, int64_t k, 
    torch::Tensor const& A, int64_t ldA,
    torch::Tensor const& B, int64_t ldB,
    torch::Tensor      & C, int64_t ldC);
  
/**
* shapes assumed:
* - X: Seq_len x Hidd
* - W1W3: Up_hidd x hidd
*/
at::Tensor fused_swiglu_bf16(torch::Tensor const& X,
                             torch::Tensor const& W1W3) {
  auto M = X.size(0);
  auto N = W1W3.size(0);
  auto K = X.size(1);
  auto options_c_type = X.options().dtype(c10::CppTypeToScalarType<C_AT_TYPE>::value);
  auto out = at::empty({M, N/2}, options_c_type);

  swiglu_fwd<cute::bfloat16_t, cute::bfloat16_t, C_TYPE>(
    M, N, K,
    X, K,
    W1W3, K,
    out, N
  );
  return out;
}

#endif // SWIGLU_H


 
