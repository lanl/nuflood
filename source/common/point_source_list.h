#pragma once

#include "grid.h"
#include "point_source.h"
#include <stdlib.h>
#include <utility>
#include <vector>

template <class T> class PointSourceList {
  public:
    PointSourceList() = default;
    ~PointSourceList();

    // Copy constructor
    PointSourceList(const PointSourceList &other);

    // Copy assignment operator
    auto operator=(const PointSourceList &other) -> PointSourceList &;

    // Move constructor
    PointSourceList(PointSourceList &&other) noexcept;

    // Move assignment operator
    auto operator=(PointSourceList &&other) noexcept -> PointSourceList &;

    void Scale(T scalar);
    void Add(const PointSource<T> &source);
    void SetCoordinates(const Grid<T> &grid);
    void Update(T current_time);
    auto SumOfRates() -> T;

    auto num_points() -> unsigned int & { return num_points_; }
    auto x() const -> double * { return x_; }
    auto y() const -> double * { return y_; }
    auto x_id() const -> unsigned int * { return x_id_; }
    auto y_id() const -> unsigned int * { return y_id_; }
    auto rate() const -> T * { return rate_; }

  private:
    static constexpr double ROUNDING_OFFSET = 0.5;

    std::vector<PointSource<T>> sources_;
    unsigned int num_points_{0};
    double *x_{nullptr};
    double *y_{nullptr};
    unsigned int *x_id_{nullptr};
    unsigned int *y_id_{nullptr};
    T *rate_{nullptr};
};

// Copy constructor
template <class T>
PointSourceList<T>::PointSourceList(const PointSourceList &other)
    : sources_(other.sources_), num_points_(other.num_points_) {
    if (num_points_ > 0) {
        x_ = static_cast<double *>(malloc(num_points_ * sizeof(double)));
        y_ = static_cast<double *>(malloc(num_points_ * sizeof(double)));
        x_id_ = static_cast<unsigned int *>(
            malloc(num_points_ * sizeof(unsigned int)));
        y_id_ = static_cast<unsigned int *>(
            malloc(num_points_ * sizeof(unsigned int)));
        rate_ = static_cast<T *>(malloc(num_points_ * sizeof(T)));

        for (unsigned int i = 0; i < num_points_; i++) {
            x_[i] = other.x_[i];
            y_[i] = other.y_[i];
            x_id_[i] = other.x_id_[i];
            y_id_[i] = other.y_id_[i];
            rate_[i] = other.rate_[i];
        }
    }
}

// Copy assignment operator
template <class T>
auto PointSourceList<T>::operator=(const PointSourceList &other)
    -> PointSourceList & {
    if (this != &other) {
        // Clean up existing resources
        free(x_);
        free(y_);
        free(x_id_);
        free(y_id_);
        free(rate_);

        // Copy data
        sources_ = other.sources_;
        num_points_ = other.num_points_;

        if (num_points_ > 0) {
            x_ = static_cast<double *>(malloc(num_points_ * sizeof(double)));
            y_ = static_cast<double *>(malloc(num_points_ * sizeof(double)));
            x_id_ = static_cast<unsigned int *>(
                malloc(num_points_ * sizeof(unsigned int)));
            y_id_ = static_cast<unsigned int *>(
                malloc(num_points_ * sizeof(unsigned int)));
            rate_ = static_cast<T *>(malloc(num_points_ * sizeof(T)));

            for (unsigned int i = 0; i < num_points_; i++) {
                x_[i] = other.x_[i];
                y_[i] = other.y_[i];
                x_id_[i] = other.x_id_[i];
                y_id_[i] = other.y_id_[i];
                rate_[i] = other.rate_[i];
            }
        } else {
            x_ = nullptr;
            y_ = nullptr;
            x_id_ = nullptr;
            y_id_ = nullptr;
            rate_ = nullptr;
        }
    }
    return *this;
}

// Move constructor
template <class T>
PointSourceList<T>::PointSourceList(PointSourceList &&other) noexcept
    : sources_(std::move(other.sources_)), num_points_(other.num_points_),
      x_(other.x_), y_(other.y_), x_id_(other.x_id_), y_id_(other.y_id_),
      rate_(other.rate_) {
    other.num_points_ = 0;
    other.x_ = nullptr;
    other.y_ = nullptr;
    other.x_id_ = nullptr;
    other.y_id_ = nullptr;
    other.rate_ = nullptr;
}

// Move assignment operator
template <class T>
auto PointSourceList<T>::operator=(PointSourceList &&other) noexcept
    -> PointSourceList & {
    if (this != &other) {
        // Clean up existing resources
        free(x_);
        free(y_);
        free(x_id_);
        free(y_id_);
        free(rate_);

        // Move data
        sources_ = std::move(other.sources_);
        num_points_ = other.num_points_;
        x_ = other.x_;
        y_ = other.y_;
        x_id_ = other.x_id_;
        y_id_ = other.y_id_;
        rate_ = other.rate_;

        // Reset other object
        other.num_points_ = 0;
        other.x_ = nullptr;
        other.y_ = nullptr;
        other.x_id_ = nullptr;
        other.y_id_ = nullptr;
        other.rate_ = nullptr;
    }
    return *this;
}

template <class T> void PointSourceList<T>::Scale(const T scalar) {
    for (unsigned int i = 0; i < num_points_; i++) {
        sources_[i].Scale(scalar);
    }
}

template <class T> void PointSourceList<T>::Add(const PointSource<T> &source) {
    sources_.push_back(source);
    num_points_ += 1;

    x_ = static_cast<double *>(realloc(x_, num_points_ * sizeof(double)));
    y_ = static_cast<double *>(realloc(y_, num_points_ * sizeof(double)));
    x_id_ = static_cast<unsigned int *>(
        realloc(x_id_, num_points_ * sizeof(unsigned int)));
    y_id_ = static_cast<unsigned int *>(
        realloc(y_id_, num_points_ * sizeof(unsigned int)));
    rate_ = static_cast<T *>(realloc(rate_, num_points_ * sizeof(T)));

    // Array index of the element we are adding.
    const unsigned int index = num_points_ - 1;

    x_[index] = source.x();
    y_[index] = source.y();
    x_id_[index] = 0;
    y_id_[index] = 0;
    rate_[index] = T{0};
}

template <class T>
void PointSourceList<T>::SetCoordinates(const Grid<T> &grid) {
    for (unsigned int i = 0; i < num_points_; i++) {
        const auto column = static_cast<unsigned int>(
            ((x_[i] - grid.x_lower_left()) / grid.cellsize()) +
            ROUNDING_OFFSET);
        const auto row = static_cast<unsigned int>(
            ((y_[i] - grid.y_lower_left()) / grid.cellsize()) +
            ROUNDING_OFFSET);

        x_id_[i] = column;
        y_id_[i] = row;
    }
}

template <class T> void PointSourceList<T>::Update(const T current_time) {
    for (unsigned int i = 0; i < num_points_; i++) {
        rate_[i] = sources_[i].interpolated_rate(current_time);
    }
}

template <class T> auto PointSourceList<T>::SumOfRates() -> T {
    T sum = T{0};
    for (unsigned int i = 0; i < num_points_; i++) {
        sum += rate_[i];
    }
    return sum;
}

template <class T> PointSourceList<T>::~PointSourceList() {
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
