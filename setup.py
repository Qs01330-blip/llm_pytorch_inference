import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mini_vllm_ext",
    ext_modules=[
        CUDAExtension(
            name="mini_vllm._C",
            sources=[
                "csrc/bindings/ops_binding.cpp",
                "csrc/kernels/rmsnorm/rmsnorm_kernel.cu",
                "csrc/kernels/rope/rope_kernel.cu",
                "csrc/kernels/activation/activation_kernel.cu",
                "csrc/kernels/sampling/sampling_kernel.cu",
                "csrc/kernels/embedding/embedding_kernel.cu",
                "csrc/kernels/softmax/softmax_kernel.cu",
                "csrc/kernels/mlp/mlp_kernel.cu",
            ],
            include_dirs=[
                "csrc",
                "csrc/include",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
