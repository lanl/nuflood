#pragma once

//#ifdef CUDA
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#define GpuErrChk(ans) { GpuAssert((ans), __FILE__, __LINE__); }
void GpuAssert(cudaError_t code, std::string file, int line);
//#endif
