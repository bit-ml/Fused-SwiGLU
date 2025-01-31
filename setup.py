from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

cutlass_dir        = os.path.join(this_dir, 'csrc', 'cutlass', 'include')
cutlass_tools_dir  = os.path.join(this_dir, 'csrc', 'cutlass', 'tools', 'library', 'include')
cutlass_utils_dir  = os.path.join(this_dir, 'csrc', 'cutlass', 'utils', 'include')

setup(
    name='fused_mlp_module',
    ext_modules=[
        CUDAExtension(
            name='fused_mlp_module',
            sources=[
                'csrc/SwiGLU/swiglu.cu',
                'csrc/torch_bindings.cpp',
            ],
            include_dirs=[cpp_extension.include_paths(),
                cutlass_dir,
                cutlass_tools_dir,
                cutlass_utils_dir
                          ],
            extra_compile_args={
                'cxx': ['-O3', '-Wall', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '--expt-relaxed-constexpr',
                    '--generate-code=arch=compute_80,code=sm_80',
                    '--compiler-options', "'-fPIC'",
                    "--generate-line-info",
                    "-Xptxas", "-dlcm=cg",
                    # '--ptxas-options=-v',
                ]
            },
            extra_link_args=['-lcudart']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)