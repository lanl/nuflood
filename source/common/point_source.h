#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include "file.h"

template<class T>
class PointSource {
public:
	PointSource(void) : x_(0.0), y_(0.0) {}
	PointSource(const double x, const double y, const File& file);
	void Load(const File& file);
	void Scale(const T scalar);

	T interpolated_rate(const T current_time) const;

	void set_x(const double x) { x_ = x; }
	void set_y(const double y) { y_ = y; }

	double x(void) const { return x_; }
	double y(void) const { return y_; }

	INT_TYPE x_index(const Grid<T>& grid) const;
	INT_TYPE y_index(const Grid<T>& grid) const;

protected:
	std::vector<T> time_;
	std::vector<T> rate_;
	double x_, y_;
};

template<class T>
INT_TYPE PointSource<T>::x_index(const Grid<T>& grid) const {
	return (INT_TYPE)grid.GetXIndex(x_);
}

template<class T>
INT_TYPE PointSource<T>::y_index(const Grid<T>& grid) const {
	return (INT_TYPE)grid.GetYIndex(y_);
}

template<class T>
PointSource<T>::PointSource(const double x, const double y, const File& file) {
	Load(file);
	x_ = x;
	y_ = y;
}

template<class T>
void PointSource<T>::Load(const File& file) {
	std::fstream input(file.path().c_str()); 
	std::string line;

	while (std::getline(input, line)) {
		// Remove white space from the beginning of the string.
		line.erase(line.begin(), std::find_if(line.begin(), line.end(), 
		std::not1(std::ptr_fun<int, int>(std::isspace))));

		// If the line of the file begins with '#', skip it.
		if (line[0] == '#') {
			continue;
		}

		// Read the comma-delimited source file
		std::string data;
		std::stringstream linestream(line);

		unsigned int count = 0;
		T value;
		while (std::getline(linestream, data, ',')) {
			std::stringstream(data) >> value;
			if (count == 0) {
				time_.push_back(value);
			} else {
				rate_.push_back(value);
			}

			count++;
		}
	}
}

template<class T>
void PointSource<T>::Scale(const T scalar) {
	for (unsigned int i = 0; i < time_.size(); i++) {
		rate_[i] *= scalar;
	}
}

template<class T>
T PointSource<T>::interpolated_rate(const T current_time) const {
	T rate_i = (T)0;
	// TODO: Make this linear interpolation faster. The for loop can become
	// computationally expensive if the number of time steps is large.
	for (unsigned int i = 0; i < time_.size()-1; i++) {
		if (current_time >= time_[i] && current_time < time_[i+1]) {
			rate_i = rate_[i] + (rate_[i+1] - rate_[i]) / (time_[i+1] - time_[i]) *
			                    (current_time - time_[i]);
			break;
		}
	}

	return rate_i;
}
