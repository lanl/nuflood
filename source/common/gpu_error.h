#pragma once

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#define GpuErrChk(ans)                                                         \
    {                                                                          \
        GpuAssert((ans), __FILE__, __LINE__);                                  \
    }
void GpuAssert(cudaError_t code, std::string file, int line);

#endif
