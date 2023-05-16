this is early but under full time development.

an extremely simple interface to SIMT-style (CUDA, HIP, compute shaders) computation.

current focus is CUDA + ROCm/HIP support.

kernels are written in C++. this makes it easy to port existing code and also saves a lot of headaches dealing with emitting NVVM/AMDGPU code.