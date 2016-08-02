#pragma once

#include <stdlib.h>
#include "grid.h"
#include "point_source.h"

template<class T>
class PointSourceList {
public:
	PointSourceList(void) : num_points_(0), x_(nullptr), y_(nullptr),
	                        x_id_(nullptr), y_id_(nullptr), rate_(nullptr) {}
	~PointSourceList(void);
	void Scale(const T scalar);
	void Add(const PointSource<T>& source);
	void SetCoordinates(const Grid<T>& grid);
	void Update(const T current_time);
	T SumOfRates(void);

	unsigned int& num_points(void) { return num_points_; }
	double* x(void) const { return x_; }
	double* y(void) const { return y_; }
	unsigned int* x_id(void) const { return x_id_; }
	unsigned int* y_id(void) const { return y_id_; }
	T* rate(void) const { return rate_; }

protected:
	std::vector< PointSource<T> > sources_;
	unsigned int num_points_;
	double* x_;
	double* y_;
	unsigned int* x_id_;
	unsigned int* y_id_;
	T* rate_;
};

template<class T>
void PointSourceList<T>::Scale(const T scalar) {
	for (unsigned int i = 0; i < num_points_; i++) {
		sources_[i].Scale(scalar);
	}
}

template <class T>
void PointSourceList<T>::Add(const PointSource<T>& source) {
	sources_.push_back(source);
	num_points_ += 1;

	x_ = (double*)realloc(x_, num_points_*sizeof(double));
	y_ = (double*)realloc(y_, num_points_*sizeof(double));
	x_id_ = (unsigned int*)realloc(x_id_, num_points_*sizeof(unsigned int));
	y_id_ = (unsigned int*)realloc(y_id_, num_points_*sizeof(unsigned int));
	rate_ = (T*)realloc(rate_, num_points_*sizeof(T));

	// Array ID of the element we are adding.
	unsigned int id = num_points_ - 1;

	x_[id] = source.x();
	y_[id] = source.y();
	x_id_[id] = 0;
	y_id_[id] = 0;
	rate_[id] = (T)0;
}

template<class T>
void PointSourceList<T>::SetCoordinates(const Grid<T>& grid) {
	for (unsigned int i = 0; i < num_points_; i++) {
		unsigned int column = (unsigned int)((x_[i] - grid.x_lower_left()) / grid.cellsize() + 0.5);
		unsigned int row = (unsigned int)((y_[i] - grid.y_lower_left()) / grid.cellsize() + 0.5);

		x_id_[i] = column;
		y_id_[i] = row;
	}
}

template<class T>
void PointSourceList<T>::Update(const T current_time) {
	for (unsigned int i = 0; i < num_points_; i++) {
		rate_[i] = sources_[i].interpolated_rate(current_time);
	}
}

template<class T>
T PointSourceList<T>::SumOfRates(void) {
	T sum = (T)0;
	for (unsigned int i = 0; i < num_points_; i++) {
		sum += rate_[i];
	}
	return sum;
}

template<class T>
PointSourceList<T>::~PointSourceList(void) {
	free(x_);
	free(y_);
	free(x_id_);
	free(y_id_);
	free(rate_);

	x_ = nullptr;
	y_ = nullptr;
	y_id_ = nullptr;
	x_id_ = nullptr;
	rate_ = nullptr;
}
