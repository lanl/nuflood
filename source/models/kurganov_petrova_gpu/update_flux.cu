#include <cuda.h>
#include <cuda_runtime.h>
#include <common/gpu_error.h>
#include "compute_time_step.h"

template <typename T>
__inline__ __device__
T* get_element(T *base, size_t pitch, int col, int row) {
	return (T*)((char*)base + row*pitch) + col;
}

__global__ void UpdateFluxKernel(double* B, double* w) {
}

void UpdateFlux(GpuRaster<double>* topographic_elevation,
                GpuRaster<double>* water_surface_elevation) {
	dim3 grid_dim = topographic_elevation->gpu_grid_dim();
	dim3 block_dim = topographic_elevation->gpu_block_dim();
	cudaFuncSetCacheConfig(UpdateFluxKernel, cudaFuncCachePreferShared);

	UpdateFluxKernel <<< grid_dim, block_dim >>> (topographic_elevation->gpu_array(),
	                                              water_surface_elevation->gpu_array());

	GpuErrChk(cudaPeekAtLastError());
	GpuErrChk(cudaDeviceSynchronize());
}
