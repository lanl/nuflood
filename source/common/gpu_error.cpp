#ifdef __CUDACC__

#include "gpu_error.h"
#include "error.h"

void GpuAssert(cudaError_t code, std::string file, int line) {
    if (code != cudaSuccess) {
        std::string error_string = cudaGetErrorString(code) + std::string(" ") +
                                   file + std::string(" ") +
                                   std::to_string(line);
        PrintErrorAndExit(error_string);
    }
}

#endif
