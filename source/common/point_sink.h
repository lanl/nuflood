#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "file.h"
#include "grid.h"

template<class T>
class PointSink {
public:
	PointSink(void) : x_(0.0), y_(0.0), rate_((T)(0)), depth_((T)(0), name_) {}
	PointSink(const double x, const double y, const T rate,
	          const T depth = (T)0, const std::string name = "");

	void Scale(const T scalar);

	void set_x(const double x) { x_ = x; }
	void set_y(const double y) { y_ = y; }
	void set_depth(const T depth) { depth_ = depth; }
	void set_rate(const T rate) { rate_ = rate; }
	void set_name(const std::string name) { name_ = name; }

	double x(void) const { return x_; }
	double y(void) const { return y_; }
	double depth(void) const { return depth_; }
	T rate(void) const { return rate_; }
	std::string name(void) const { return name_; }

	INT_TYPE x_index(const Grid<T>& grid) const;
	INT_TYPE y_index(const Grid<T>& grid) const;

protected:
	double x_;
	double y_;
	T depth_;
	T rate_;
	std::string name_;
};

template<class T>
INT_TYPE PointSink<T>::x_index(const Grid<T>& grid) const {
	return (INT_TYPE)grid.GetXIndex(x_);
}

template<class T>
INT_TYPE PointSink<T>::y_index(const Grid<T>& grid) const {
	return (INT_TYPE)grid.GetYIndex(y_);
}

template<class T>
PointSink<T>::PointSink(const double x, const double y, const T rate,
                        const T depth, const std::string name) {
	x_ = x;
	y_ = y;
	rate_ = rate;
	depth_ = depth;
	name_ = name;
}

template<class T>
void PointSink<T>::Scale(const T scalar) {
	rate_ *= scalar;
}
