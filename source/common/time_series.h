#pragma once

#include "file.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <class T> class TimeSeries {
  public:
    TimeSeries();
    TimeSeries(const File &file);

    auto IsEmpty() const -> bool;
    void Load(const File &file);
    void Scale(T scalar);
    void Update(T current_time);
    auto end_time() const -> T { return time_.back(); }
    auto p_current_value() -> T * { return &current_value_; }
    auto interpolated_value(T current_time) const -> T;

  private:
    std::vector<T> time_;
    std::vector<T> value_;
    T current_value_;
    T end_time_;
};

template <class T> auto TimeSeries<T>::IsEmpty() const -> bool {
    return time_.empty();
}

template <class T> TimeSeries<T>::TimeSeries() : current_value_{T{}} {}

template <class T> TimeSeries<T>::TimeSeries(const File &file) { Load(file); }

template <class T> void TimeSeries<T>::Load(const File &file) {
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

template <class T> void TimeSeries<T>::Scale(const T scalar) {
    for (unsigned int i = 0; i < time_.size(); i++) {
        value_[i] *= scalar;
    }
}

template <class T> void TimeSeries<T>::Update(const T current_time) {
    current_value_ = interpolated_value(current_time);
}

template <class T>
auto TimeSeries<T>::interpolated_value(const T current_time) const -> T {
    if (time_.empty()) {
        return T{};
    }

    T value_i = T{};
    for (unsigned int i = 0; i < time_.size() - 1; i++) {
        if (current_time >= time_[i] && current_time < time_[i + 1]) {
            value_i = value_[i] + (value_[i + 1] - value_[i]) /
                                      (time_[i + 1] - time_[i]) *
                                      (current_time - time_[i]);
            break;
        }
    }

    return value_i;
}
