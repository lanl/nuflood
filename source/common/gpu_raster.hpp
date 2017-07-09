#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_error.h"
#include "raster.hpp"

#define BLOCK_COLS 16
#define BLOCK_ROWS 12

//! Default GpuRaster implementation.
/*! Expands Raster functionality for graphics processing units (GPUs).
    \tparam Type of raster data. Default is double.
*/
template<class T = double>
class GpuRaster : public Raster<T> {
public:
	GpuRaster(std::string path, GDALAccess access = GA_ReadOnly);
	GpuRaster(const GpuRaster& raster, std::string path, GDALAccess access = GA_ReadOnly);
	~GpuRaster(void);

	T* gpu_array(void) const { return gpu_array_; }
	dim3 gpu_block_dim(void) const { return gpu_block_dim_; }
	dim3 gpu_grid_dim(void) const { return gpu_grid_dim_; }
	int gpu_width(void) const { return gpu_width_; }
	int gpu_height(void) const { return gpu_height_; }
	size_t gpu_pitch(void) const { return gpu_pitch_; }

	void CopyFromHostRaster(const Raster<T>& raster);
	void Fill(T value);

private:
	T* gpu_array_;
	dim3 gpu_block_dim_;
	dim3 gpu_grid_dim_;
	int gpu_width_;
	int gpu_height_;
	size_t gpu_pitch_;
};

template<class T>
inline GpuRaster<T>::GpuRaster(std::string path, GDALAccess access) : Raster<T>(path, access) {
	gpu_block_dim_.x = BLOCK_COLS;
	int width = Raster<T>::width();
	gpu_grid_dim_.x = ((width + 1) + (gpu_block_dim_.x + 1)) / gpu_block_dim_.x;

	gpu_block_dim_.y = BLOCK_ROWS;
	int height = Raster<T>::height();
	gpu_grid_dim_.y = ((height + 1) + (gpu_block_dim_.y + 1)) / gpu_block_dim_.y;

	// Adding two cells in each direction removes problems we could encounter at
	// the right and top boundaries (as each block requires data from two extra
	// cells in each direction), and the boundary cells themselves are contained
	// within the grid (defined by the quantities above). Note that the CPU grid
	// from which we reference should already be extended to contain boundary
	// cells (two in each direction); these are not included here.
	gpu_width_ = gpu_block_dim_.x * gpu_grid_dim_.x + 2;
	gpu_height_ = gpu_block_dim_.y * gpu_grid_dim_.y + 2;

	// Allocate memory for the GPU array and copy from the host array.
	GpuErrChk(cudaMallocPitch((void**)&gpu_array_, &gpu_pitch_,
	                          gpu_width_*sizeof(T), gpu_height_));
	GpuRaster<T>::CopyFromHostRaster(*this);
}

template<class T>
inline GpuRaster<T>::GpuRaster(const GpuRaster& raster, std::string path, GDALAccess access) : Raster<T>(raster, path, access) {
	// Populate GPU-based raster metadata.
	gpu_block_dim_ = raster.gpu_block_dim();
	gpu_grid_dim_ = raster.gpu_grid_dim();
	gpu_width_ = raster.gpu_width();
	gpu_height_ = raster.gpu_height();

	// Allocate the new GPU array and perform a copy from the reference.
	GpuErrChk(cudaMallocPitch((void**)&gpu_array_, &gpu_pitch_,
	                          gpu_width_*sizeof(T), gpu_height_));
	GpuErrChk(cudaMemcpy2D(gpu_array_, gpu_pitch_, raster.gpu_array(),
	                       gpu_pitch_, gpu_width_ * sizeof(T), gpu_height_,
                          cudaMemcpyDeviceToDevice));
}

template<class T>
inline void GpuRaster<T>::CopyFromHostRaster(const Raster<T>& raster) {
	if (GpuRaster<T>::EqualDimensions(raster)) {
		T* offset_gpu_array = (T*)((char*)gpu_array_);
		GpuErrChk(cudaMemcpy2D(offset_gpu_array, gpu_pitch_, raster.array(),
		                       raster.width()*sizeof(T),
		                       raster.width()*sizeof(T),
		                       raster.height(), cudaMemcpyHostToDevice));
	} else {
		std::string error_message = "Host/device raster dimension mismatch "
		                            "between \"" + GpuRaster<T>::path() +
		                            "\" and \"" + raster.path() + "\".";
		std::cerr << "ERROR: " << error_message << std::endl;
		std::exit(2);
	}
}

template<class T>
inline void GpuRaster<T>::Fill(T value) {
	Raster<T>::Fill(value);
	GpuRaster<T>::CopyFromHostRaster(*this);
}

template<class T>
inline GpuRaster<T>::~GpuRaster(void) {
	GpuErrChk(cudaFree(gpu_array_));
}
