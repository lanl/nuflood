#include <cuda.h>
#include <cuda_runtime.h>
#include "compute_time_step.h"

template<int block_size>
__global__ void ComputeTimeStepKernel(double* max_speed, int n) {
	__shared__ double s_max[block_size];

	int tid = threadIdx.x;
	s_max[tid] = 0.0;

	for (int i = tid; i < n; i += block_size) {
		s_max[tid] = fmax(s_max[tid], max_speed[i]);
	}

	__syncthreads();

	if (block_size >= 512) {
		if (tid < 256) s_max[tid] = fmax(s_max[tid], s_max[tid+256]);
		__syncthreads();
	}

	if (block_size >= 256) {
		if (tid < 128) s_max[tid] = fmax(s_max[tid], s_max[tid+128]);
		__syncthreads();
	}

	if (block_size >= 128) {
		if (tid < 64) s_max[tid] = fmax(s_max[tid], s_max[tid+64]);
		__syncthreads();
	}

	if (tid < 32) {
		volatile double *s_mem = s_max;

		if (block_size >= 64) s_mem[tid] = fmax(s_mem[tid], s_mem[tid+32]);

		if (tid < 16) {
			if (block_size >= 32) s_mem[tid] = fmax(s_mem[tid], s_mem[tid+16]);
			if (block_size >= 16) s_mem[tid] = fmax(s_mem[tid], s_mem[tid+8]);
			if (block_size >= 8) s_mem[tid] = fmax(s_mem[tid], s_mem[tid+4]);
			if (block_size >= 4) s_mem[tid] = fmax(s_mem[tid], s_mem[tid+2]);
			if (block_size >= 2) s_mem[tid] = fmax(s_mem[tid], s_mem[tid+1]);
		}

		if (tid == 0) max_speed[0] = s_mem[0];
	}
}

double ComputeTimeStep(GpuRaster<double>* max_speed, double desingularization) {
	int block_size = 1;
	int num_elements = max_speed->gpu_grid_dim().x *
	                   max_speed->gpu_grid_dim().y;

	for (int k = 1; k <= 512; k *= 2) {
		block_size = (num_elements >= k) ? k : block_size;
	}

	size_t shared_mem_size = block_size*sizeof(double) * (block_size <= 32) ? 2 : 1;

	switch (block_size) {
		case 512:
			cudaFuncSetCacheConfig(ComputeTimeStepKernel<512>, cudaFuncCachePreferShared);
			ComputeTimeStepKernel<512> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
		case 256:
			ComputeTimeStepKernel<256> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
		case 128:
			ComputeTimeStepKernel<128> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
		case 64:
			ComputeTimeStepKernel<64> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
		case 32:
			ComputeTimeStepKernel<32> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
		case 16:
			ComputeTimeStepKernel<16> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
		case 8:
			ComputeTimeStepKernel<8> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
		case 4:
			ComputeTimeStepKernel<4> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
		case 2:
			ComputeTimeStepKernel<2> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
		case 1:
			ComputeTimeStepKernel<1> <<< 1, block_size, shared_mem_size, 0 >>>
				(max_speed->gpu_array(), num_elements);
			break;
	}

	GpuErrChk(cudaPeekAtLastError());
	GpuErrChk(cudaDeviceSynchronize());

	double domain_max_speed;
	cudaMemcpy(&domain_max_speed, max_speed->gpu_array(), sizeof(double), cudaMemcpyDeviceToHost);
	return max_speed->cellsize_x() / fmax(fmax(4.0 * domain_max_speed, desingularization), 10.0);
}
