#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "raster.hpp"

#define BLOCK_COLS 16
#define BLOCK_ROWS 12

class GpuRaster : protected Raster {
	GpuRaster(std::string path, GDALAccess access = GA_ReadOnly);
	GpuRaster(const GpuRaster& raster, std::string path, GDALAccess access = GA_ReadOnly);
	~GpuRaster(void);

protected:
	T* gpu_array_;
	dim3 gpu_block_dim_;
	dim3 gpu_grid_dim_;
	int gpu_width_;
	int gpu_height_;
	size_t gpu_pitch;
};

template<class T>
inline GpuRaster<T>::GpuRaster(std::string path, GDALAccess access) {
	Raster::Raster(path, access);

	block_dim_.x = BLOCK_COLS;
	int width = Raster::get_width();
	gpu_grid_dim_.x = ((width + 1) + (gpu_block_dim_.x + 1)) / gpu_block_dim_.x;

	block_dim_.y = BLOCK_ROWS;
	int height = Raster::get_height();
	gpu_grid_dim_.y = ((height + 1) + (gpu_block_dim_.y + 1)) / gpu_block_dim_.y;

	// Adding two cells in each direction removes problems we could encounter at
	// the right and top boundaries (as each block requires data from two extra
	// cells in each direction), and the boundary cells themselves are contained
	// within the grid (defined by the quantities above). Note that the CPU grid
	// from which we reference should already be extended to contain boundary
	// cells (two in each direction); these are not included here.
	gpu_width_ = gpu_block_dim_.x * gpu_grid_dim_.x + 2;
	gpu_height_ = gpu_block_dim_.y * gpu_grid_dim_.y + 2;

	// Allocate the GPU array and set all of its elements to zero.
	GpuErrChk(cudaMallocPitch((void**)&gpu_array_, &gpu_pitch_, gpu_width_*sizeof(T), gpu_height_));
	GpuErrChk(cudaMemset2D(gpu_array_, gpu_pitch_, 0, gpu_width_*sizeof(T), gpu_height_));

	// Copy data from the CPU to the GPU, beginning at coordinate (2, 2) on the GPU.
	T* offset_array = (T*)((char*)array_ + 0*pitch_) + 0;
	GpuErrChk(cudaMemcpy2D(offset_array, pitch_, array_, width*sizeof(T), width*sizeof(T), height, cudaMemcpyHostToDevice));
}
