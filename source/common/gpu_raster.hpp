#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_error.h"
#include "raster.hpp"

#define BLOCK_COLS 16
#define BLOCK_ROWS 12

template<class T>
class GpuRaster : public Raster<T> {
public:
	GpuRaster(std::string path, GDALAccess access = GA_ReadOnly);
	GpuRaster(const GpuRaster& raster, std::string path, GDALAccess access = GA_ReadOnly);
	~GpuRaster(void);

private:
	T* gpu_array_;
	dim3 gpu_block_dim_;
	dim3 gpu_grid_dim_;
	int gpu_width_;
	int gpu_height_;
	size_t gpu_pitch_;
};

template<class T>
inline GpuRaster<T>::GpuRaster(std::string path, GDALAccess access) {
	Raster<T>::Raster(path, access);

	gpu_block_dim_.x = BLOCK_COLS;
	int width = Raster<T>::get_width();
	gpu_grid_dim_.x = ((width + 1) + (gpu_block_dim_.x + 1)) / gpu_block_dim_.x;

	gpu_block_dim_.y = BLOCK_ROWS;
	int height = Raster<T>::get_height();
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
	T* offset_array = (T*)((char*)Raster<T>::array_ + 0*gpu_pitch_) + 0;
	GpuErrChk(cudaMemcpy2D(offset_array, gpu_pitch_, Raster<T>::array_, width*sizeof(T), width*sizeof(T), height, cudaMemcpyHostToDevice));
}

template<class T>
inline GpuRaster<T>::GpuRaster(const GpuRaster& raster, std::string path, GDALAccess access) {
	// Construct the CPU-based raster object.
	Raster<T>::Raster(raster, path, access);

	// Populate GPU-based raster metadata.
	gpu_block_dim_ = raster.get_gpu_block_dim();
	gpu_grid_dim_ = raster.get_gpu_grid_dim();
	gpu_width_ = raster.get_gpu_width();
	gpu_height_ = raster.get_gpu_height();

	// Allocate the GPU array and set all of its elements to zero.
	GpuErrChk(cudaMallocPitch((void**)&gpu_array_, &gpu_pitch_, gpu_width_*sizeof(T), gpu_height_));
	GpuErrChk(cudaMemset2D(gpu_array_, gpu_pitch_, 0, gpu_width_*sizeof(T), gpu_height_));

	// Copy data from the CPU to the GPU, beginning at coordinate (2, 2) on the GPU.
	T* offset_array = (T*)((char*)Raster<T>::array_ + 0*gpu_pitch_) + 0;
	int width = Raster<T>::get_width();
	int height = Raster<T>::get_height();
	GpuErrChk(cudaMemcpy2D(offset_array, gpu_pitch_, Raster<T>::array_, width*sizeof(T), width*sizeof(T), height, cudaMemcpyHostToDevice));
}

template<class T>
inline GpuRaster<T>::~GpuRaster(void) {
	cudaFree(gpu_array_);
	Raster<T>::~Raster();
}
