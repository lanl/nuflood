#pragma once

#define BLOCK_ROWS 12
#define BLOCK_COLS 16

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_error.h"
#include "grid.h"

template<class T>
class GpuGrid {
public:
	GpuGrid(void);
	~GpuGrid(void);

	void CopyFromCpuGrid(const Grid<T>& reference);
	void CopyToCpuGrid(const Grid<T>& reference) const;
	bool IsEmpty(void) const;
	void set_name(const std::string name) { name_ = name; }

	T* data(void) const { return data_; }
	size_t pitch(void) const { return pitch_; }
	unsigned int num_columns(void) const { return num_columns_; }
	unsigned int num_rows(void) const { return num_rows_; }
	double x_lower_left(void) const { return x_lower_left_; }
	double y_lower_left(void) const { return y_lower_left_; }
	double cellsize(void) const { return cellsize_; }
	double nodata_value(void) const { return nodata_value_; }
	std::string name(void) const { return name_; }
	dim3 block_dim(void) const { return block_dim_; }
	dim3 grid_dim(void) const { return grid_dim_; }

private:
	T* data_;
	size_t pitch_;
	unsigned int num_columns_, num_rows_;
	double x_lower_left_, y_lower_left_, cellsize_, nodata_value_;
	std::string name_;
	dim3 block_dim_, grid_dim_;
};

template<class T>
GpuGrid<T>::GpuGrid(void) {
	data_ = nullptr;
	pitch_ = 0;
	num_columns_ = 0;
	num_rows_ = 0;
	x_lower_left_ = 0.0;
	y_lower_left_ = 0.0;
	cellsize_ = 0.0;
	nodata_value_ = 0.0;
	name_ = "";
	block_dim_ = dim3(1, 1, 1);
	grid_dim_ = dim3(1, 1, 1);
}

template<class T>
inline void GpuGrid<T>::CopyFromCpuGrid(const Grid<T>& reference_grid) {
	num_columns_ = reference_grid.num_columns();
	num_rows_ = reference_grid.num_rows();
	x_lower_left_ = reference_grid.x_lower_left();
	y_lower_left_ = reference_grid.y_lower_left();
	cellsize_ = reference_grid.cellsize();
	nodata_value_ = reference_grid.nodata_value();
	name_ = reference_grid.name() + std::string("Gpu");

	if (!IsEmpty()) {
		cudaFree(data_);
		data_ = nullptr;
	}

	if (!reference_grid.IsEmpty()) {
		block_dim_.x = BLOCK_COLS;
		block_dim_.y = BLOCK_ROWS;

		grid_dim_.x = ((reference_grid.num_columns()+1) + (block_dim_.x + 1)) / block_dim_.x;
		grid_dim_.y = ((reference_grid.num_rows()+1) + (block_dim_.y + 1)) / block_dim_.y;

		// Adding two cells in each direction removes problems we could encounter
		// at the right and top boundaries (since each block requires data from
		// two extra cells in each direction), and the boundary cells themselves
		// are contained within the grid (defined by the quantities above). Note
		// that the CPU grid from which we reference should already be extended
		// to contain boundary cells (two in each direction), and these are not
		// included here.
		num_columns_ = block_dim_.x*grid_dim_.x + 2;
		num_rows_    = block_dim_.y*grid_dim_.y + 2;

		// Allocate the array and set all of its data elements to zero
		GpuErrChk(cudaMallocPitch((void**)&data_, &pitch_, num_columns_*sizeof(T), num_rows_));
		GpuErrChk(cudaMemset2D(data_, pitch_, 0, num_columns_*sizeof(T), num_rows_));

		// Copy data from the CPU to the GPU, beginning at coordinate (2, 2) on the GPU grid.
		T* offset_data = (T*)((char*)data_ + 0*pitch_) + 0;
		GpuErrChk(cudaMemcpy2D(offset_data, pitch_, reference_grid.data(),
		                       reference_grid.num_columns()*sizeof(T),
		                       reference_grid.num_columns()*sizeof(T),
		                       reference_grid.num_rows(), cudaMemcpyHostToDevice));
	}
}

template<class T>
inline void GpuGrid<T>::CopyToCpuGrid(const Grid<T>& cpu_grid) const {
	if (!cpu_grid.IsEmpty()) {
		size_t CPU_pitch = cpu_grid.num_columns()*sizeof(T);
		T* CPU_ptr = cpu_grid.data();
		T* GPU_ptr = (T*)((char*)data_ + 0*pitch_) + 0;
		GpuErrChk(cudaMemcpy2D(CPU_ptr, CPU_pitch, GPU_ptr, pitch_,
		                       cpu_grid.num_columns()*sizeof(T),
		                       cpu_grid.num_rows(), cudaMemcpyDeviceToHost));
	}
}

template<class T>
inline GpuGrid<T>::~GpuGrid(void) {
	if (!IsEmpty()) {
		cudaFree(data_);
		data_ = nullptr;
	}
}

template<class T>
inline bool GpuGrid<T>::IsEmpty(void) const {
	return data_ == nullptr;
}
