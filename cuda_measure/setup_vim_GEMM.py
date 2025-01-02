from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cutlass_path = "cutlass/include"
file_path = os.path.join(os.getcwd(), cutlass_path)

setup(
    name='vim_GEMM',
    ext_modules=[
        CUDAExtension(name='vim_GEMM', sources=[
            'vim_GEMM.cpp',
            'vim_GEMM_kernel.cu',
        ], include_dirs=[file_path],
        extra_compile_args=["-std=c++17"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })