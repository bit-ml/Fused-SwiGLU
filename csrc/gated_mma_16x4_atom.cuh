#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_traits.hpp>

namespace cute {
  // MMA 16x4x16 TN
  // as far as I can tell this is not relevant for any part in offset and layout computation, but I need this object for the template to be valid (some compilation issue)
  struct SM80_16x4x16_F32BF16BF16F32_TN_GATED_COPY
  {
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];

    CUTE_HOST_DEVICE static void
    fma(float         & d0, float         & d1, float         & d2, float         & d3,
        uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
        uint32_t const& b0, uint32_t const& b1,
        float const   & c0, float const   & c1, float const   & c2, float const   & c3)
    {
  #if defined(CUTE_ARCH_MMA_SM80_ENABLED)
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
          "r"(b0),  "r"(b1),
          "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
  #else
      CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x4x16_F32BF16BF16F32_TN_GATED_COPY without CUTE_ARCH_MMA_SM80_ENABLED");
  #endif
    }
  };

  template <>
  struct MMA_Traits<SM80_16x4x16_F32BF16BF16F32_TN_GATED_COPY>
        : MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>
  {
    // not entirely sure what effect these types have on offset computation
    using ValTypeD = float;
    using ValTypeA = half_t;
    using ValTypeB = half_t;
    using ValTypeC = float;

    // 4 is very important here, since I reduce the atoms on the feature (N) dimension
    using Shape_MNK = Shape<_16,_4,_16>;
    using ThrID   = Layout<_32>;
    // A and B don't really matter, this atom is useful only for creating the right shapes for the C layout
    using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
                          Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
    using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                          Stride<Stride<_16,_1>,Stride<_8,_64>>>;
    // the C layout suggest the (2) atoms used for the merging
    // and the strides are changed to make sense in terms of the new numel
    using CLayout = Layout<Shape <Shape < _4,_8>, _2>,
                             Stride<Stride<_16,_1>,_8>>;
  };
}