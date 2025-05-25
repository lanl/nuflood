#pragma once

#include "file.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <class T> class PointSource {
  public:
    PointSource() : x_(0.0), y_(0.0) {}
    PointSource(double x_coord, double y_coord, const File &file);
    void Load(const File &file);
    void Scale(T scalar);

    auto interpolated_rate(T current_time) const -> T;

    void set_x(double x_coord) { x_ = x_coord; }
    void set_y(double y_coord) { y_ = y_coord; }

    auto x() const -> double { return x_; }
    auto y() const -> double { return y_; }

    auto x_index(const Grid<T> &grid) const -> INT_TYPE;
    auto y_index(const Grid<T> &grid) const -> INT_TYPE;

  private:
    std::vector<T> time_;
    std::vector<T> rate_;
    double x_, y_;
};

template <class T>
auto PointSource<T>::x_index(const Grid<T> &grid) const -> INT_TYPE {
    return (INT_TYPE)grid.GetXIndex(x_);
}

template <class T>
auto PointSource<T>::y_index(const Grid<T> &grid) const -> INT_TYPE {
    return (INT_TYPE)grid.GetYIndex(y_);
}

template <class T>
PointSource<T>::PointSource(double x_coord, double y_coord, const File &file)
    : x_(x_coord), y_(y_coord) {
    Load(file);
}

template <class T> void PointSource<T>::Load(const File &file) {
    std::fstream input(file.path().c_str());
    std::string line;

    while (std::getline(input, line)) {
        // Remove white space from the beginning of the string.
        line.erase(line.begin(),
                   std::find_if(line.begin(), line.end(),
                                [](unsigned char character) -> bool {
                                    return !std::isspace(character);
                                }));

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

template <class T> void PointSource<T>::Scale(T scalar) {
    for (unsigned int i = 0; i < time_.size(); i++) {
        rate_[i] *= scalar;
    }
}

template <class T>
auto PointSource<T>::interpolated_rate(T current_time) const -> T {
    T rate_i = (T)0;
    // TODO: Make this linear interpolation faster. The for loop can become
    // computationally expensive if the number of time steps is large.
    for (unsigned int i = 0; i < time_.size() - 1; i++) {
        if (current_time >= time_[i] && current_time < time_[i + 1]) {
            rate_i = rate_[i] + (rate_[i + 1] - rate_[i]) /
                                    (time_[i + 1] - time_[i]) *
                                    (current_time - time_[i]);
            break;
        }
    }

    return rate_i;
}
