#pragma once

#ifdef CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_error.h"

template<class T>
class GpuVector {
public:
	GpuVector(void);
	~GpuVector(void);

	void Initialize(unsigned int num_rows);
	void CopyFromCpu(const T* cpu_ptr, const unsigned int num_rows);

	T* data(void) const { return data_; }
	unsigned int num_rows(void) const { return num_rows_; }

protected:
	T* data_;
	unsigned int num_rows_;
};

template<class T>
inline GpuVector<T>::GpuVector(void) {
	data_ = nullptr;
	num_rows_ = 0;
}

template<class T>
inline void GpuVector<T>::Initialize(unsigned int num_rows) {
	if (data_ != nullptr) {
		GpuErrChk(cudaFree(data_));
		data_ = nullptr;
		num_rows_ = 0;
	}

	num_rows_ = num_rows;

	if (num_rows_ > 0) {
		GpuErrChk(cudaMalloc((void**)&data_, num_rows_*sizeof(T)));
		GpuErrChk(cudaMemset(data_, 0, num_rows_*sizeof(T)));
	}
}

template<class T>
inline void GpuVector<T>::CopyFromCpu(const T* cpu_ptr, const unsigned int num_rows) {
	GpuErrChk(cudaMemcpy(data_, cpu_ptr, num_rows*sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
inline GpuVector<T>::~GpuVector(void) {
	GpuErrChk(cudaFree(data_));
	data_ = nullptr;
}

#endif
