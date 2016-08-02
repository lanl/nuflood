#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include "file.h"

template<class T>
class TimeSeries {
public:
	TimeSeries(void);
	TimeSeries(const File& file);

	bool IsEmpty(void) const;
	void Load(const File& file);
	void Scale(const T scalar);
	void Update(const T current_time);
	T end_time(void) const { return time_.back(); }
	T* p_current_value(void) { return &current_value_; }
	T interpolated_value(const T current_time) const;

protected:
	std::vector<T> time_;
	std::vector<T> value_;
	T current_value_;
	T end_time_;
};

template<class T>
bool TimeSeries<T>::IsEmpty(void) const {
	return time_.size() == 0;
}

template<class T>
TimeSeries<T>::TimeSeries(void) {
	current_value_ = (T)0;
}

template<class T>
TimeSeries<T>::TimeSeries(const File& file) {
	Load(file);
}

template<class T>
void TimeSeries<T>::Load(const File& file) {
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

		T value;
		unsigned int count = 0;
		while (std::getline(linestream, data, ',')) {
			std::stringstream(data) >> value;
			if (count == 0) {
				time_.push_back(value);
			} else {
				value_.push_back(value);
			}

			count++;
		}
	}
}

template<class T>
void TimeSeries<T>::Scale(const T scalar) {
	for (unsigned int i = 0; i < time_.size(); i++) {
		value_[i] *= scalar;
	}
}

template<class T>
void TimeSeries<T>::Update(const T current_time) {
	current_value_ = interpolated_value(current_time);
}

template<class T>
T TimeSeries<T>::interpolated_value(const T current_time) const {
	if (time_.size() == 0) {
		return (T)0;
	}

	float value_i = (T)0;
	for (unsigned int i = 0; i < time_.size()-1; i++) {
		if (current_time >= time_[i] && current_time < time_[i+1]) {
			value_i = value_[i] + (value_[i+1] - value_[i]) /
			          (time_[i+1] - time_[i]) * (current_time - time_[i]);
			break;
		}
	}

	return value_i;
}
