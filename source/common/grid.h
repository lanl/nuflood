#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <string.h>
#include <string>
#include <sstream>
#include <omp.h>
#include "error.h"
#include "file.h"
#include "folder.h"
#include "precision.h"

template<class T>
class Grid {
public:
	Grid(const File& file);
	Grid(const INT_TYPE num_columns = 0, const INT_TYPE num_rows = 0,
	     const double x_lower_left = 0.0, const double y_lower_left = 0.0,
	     const double cellsize = 0.0, const double nodata_value = -9999.0,
	     const std::string name = "");
	~Grid(void);

	// Initialization and loading operations.
	void Initialize(const INT_TYPE num_columns = 0, const INT_TYPE num_rows = 0,
	                const double x_lower_left = 0.0, const double y_lower_left = 0.0,
	                const double cellsize = 0.0, const double nodata_value = -9999.0,
	                const std::string name = "");
	void Load(const File& file);
	void Copy(const Grid<T>& reference_grid);
	void Fill(const T value);

	// Data manipulation operations.
	void Normalize(void);
	void BilinearInterpolate(void);
	void AddBoundaries(void);
	void Add(const Grid<T>& a);
	void Add(const T value);
	void Scale(const T value);
	void ReplaceValue(const T a, const T b);
	void EqualsDifferenceOf(const Grid<T>& b, const Grid<T>& a);

	// Index operations.
	INT_TYPE GetXIndex(const double X) const;
	INT_TYPE GetYIndex(const double Y) const;
	INT_TYPE GetLinearIndex(const double X, const double Y) const;
	INT_TYPE GetLinearIndex(const INT_TYPE column,
	                            const INT_TYPE row) const;

	// Coordinate operations.
	double GetXCoordinate(const INT_TYPE column) const;
	double GetYCoordinate(const INT_TYPE row) const;

	// Validation operations.
	void CheckIndexValidity(const INT_TYPE column,
	                        const INT_TYPE row) const;
	bool DimensionsArePositive(void) const;
	bool DimensionsMatch(const Grid& a) const;
	bool IsEmpty(void) const;

	// Set operations.
	void Set(const T value, const INT_TYPE column, const INT_TYPE row);
	void Set(const T value, const double x, const double y);
	void set_name(const std::string name) { name_ = name; }

	// Get operations.
	T Get(const INT_TYPE column, const INT_TYPE row) const;
	T Get(const INT_TYPE linear_index) const;
	T Get(const double x, const double y) const;
	T* GetPointer(const INT_TYPE column, const INT_TYPE row) const;
	T Minimum(void) const;
	T Maximum(void) const;
	long double InnerSum(void) const;
	long double Sum(void) const;

	// Write operations.
	void Write(const Folder& folder, const float current_time) const;
	void Write(const Folder& folder) const;
	void Write(const std::string filename) const;
	void WriteResized(const Folder& folder, const float current_time) const;
	void WriteResized(const Folder& folder) const;
	void WriteWithoutBoundaries(const Folder& folder, const float current_time) const;
	void WriteWithoutBoundaries(const Folder& folder) const;

	// Accessors.
	std::string name(void) const { return name_; }
	T* data(void) const { return data_; }
	INT_TYPE num_columns(void) const { return num_columns_; }
	INT_TYPE num_rows(void) const { return num_rows_; }
	double x_lower_left(void) const { return x_lower_left_; }
	double y_lower_left(void) const { return y_lower_left_; }
	double cellsize(void) const { return cellsize_; }
	double nodata_value(void) const { return nodata_value_; }

private:
	T* data_;
	INT_TYPE num_columns_, num_rows_;
	double x_lower_left_, y_lower_left_, cellsize_, nodata_value_;
	std::string name_;
};

template<class T>
inline Grid<T>::Grid(const File& file) {
	Load(file);
}

template<class T>
inline Grid<T>::~Grid(void) {
	free(data_);
	data_ = nullptr;
}

template<class T>
inline Grid<T>::Grid(const INT_TYPE num_columns,
                     const INT_TYPE num_rows,
	                  const double x_lower_left,
                     const double y_lower_left,
	                  const double cellsize,
                     const double nodata_value,
                     const std::string name) {
	num_columns_  = num_columns;
	num_rows_ = num_rows;
	x_lower_left_ = x_lower_left;
	y_lower_left_ = y_lower_left;
	cellsize_ = cellsize;
	nodata_value_ = nodata_value;
	name_ = name;
	data_ = nullptr;

	if (DimensionsArePositive()) {
		data_ = (T*)calloc(num_columns_*num_rows_, sizeof(T));
	}
}

template<class T>
inline void Grid<T>::Initialize(const INT_TYPE num_columns,
                                const INT_TYPE num_rows,
	                              const double x_lower_left,
                                const double y_lower_left,
	                              const double cellsize,
                                const double nodata_value,
                                const std::string name) {
	num_columns_  = num_columns;
	num_rows_ = num_rows;
	x_lower_left_ = x_lower_left;
	y_lower_left_ = y_lower_left;
	cellsize_ = cellsize;
	nodata_value_ = nodata_value;
	name_ = name;

	free(data_);
	data_ = nullptr;

	if (DimensionsArePositive()) {
		data_ = (T*)calloc(num_columns_*num_rows_, sizeof(T));
	}
}

template<class T>
inline void Grid<T>::Load(const File& file) {
	std::ifstream input(file.path());
	std::string temp_string;
	
	input >> temp_string;
	input >> num_columns_;

	input >> temp_string;
	input >> num_rows_;

	input >> temp_string;
	input >> x_lower_left_;

	input >> temp_string;
	input >> y_lower_left_;

	input >> temp_string;
	input >> cellsize_;

	input >> temp_string;
	input >> nodata_value_;

	data_ = (T*)calloc(num_columns_*num_rows_, sizeof(T));

	for (INT_TYPE row = num_rows_; row-- > 0;) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			if (input.good()) {
				input >> data_[row*num_columns_+column];
			} else {
				PrintErrorAndExit("Grid '" + file.path() + "' ended prematurely upon load.");
			}
		}
	}
}

template<class T>
inline void Grid<T>::Copy(const Grid<T>& reference_grid) {
	num_columns_ = reference_grid.num_columns();
	num_rows_ = reference_grid.num_rows();
	x_lower_left_ = reference_grid.x_lower_left();
	y_lower_left_ = reference_grid.y_lower_left();
	cellsize_ = reference_grid.cellsize();
	nodata_value_ = reference_grid.nodata_value();
	name_ = reference_grid.name();

	free(data_);
	data_ = nullptr;

	if (!reference_grid.IsEmpty()) {
		data_ = (T*)calloc(num_columns_*num_rows_, sizeof(T));
		memcpy(data_, reference_grid.data(), num_columns_*num_rows_*sizeof(T));
	}
}

template<class T>
inline bool Grid<T>::IsEmpty(void) const {
	return data_ == nullptr;
}

template<class T>
inline void Grid<T>::Set(const T value, const INT_TYPE column,
                         const INT_TYPE row) {
	// If we are not fully iterating over the grid, it's safest to ensure the
	// element actually exists by using GetLinearIndex. However, reliance on
	// this function may result in performance degradation (compared to operating
	// on data_ itself).
	INT_TYPE linear_index = GetLinearIndex(column, row);
	data_[linear_index] = value;
}

template<class T>
inline void Grid<T>::Set(const T value, const double x, const double y) {
	INT_TYPE linear_index = GetLinearIndex(x, y);
	data_[linear_index] = value;
}

template<class T>
inline void Grid<T>::Fill(const T value) {
	#pragma omp parallel for
	for (INT_TYPE row = 0; row < num_rows_; row++) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			data_[row*num_columns_+column] = value;
		}
	}
}

template<class T>
inline void Grid<T>::Normalize(void) {
	T minimum = Minimum();
	#pragma omp parallel for
	for (INT_TYPE row = 0; row < num_rows_; row++) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			data_[row*num_columns_+column] -= minimum;
		}
	}
}

template<class T>
inline void Grid<T>::BilinearInterpolate(void) {
	INT_TYPE num_columns_new = num_columns_ - 1;
	INT_TYPE num_rows_new = num_rows_ - 1;

	double x_lower_left_new = x_lower_left_ + 0.5*cellsize_;
	double y_lower_left_new = y_lower_left_ + 0.5*cellsize_;

	Grid<T> temp;
	temp.Initialize(num_columns_new, num_rows_new, x_lower_left_new,
	                y_lower_left_new, cellsize_, nodata_value_, name_);

	#pragma omp parallel for
	for (INT_TYPE row = 0; row < temp.num_rows(); row++) {
		for (INT_TYPE column = 0; column < temp.num_columns(); column++) {
			T average = (T)0.25*(Get(column, row  ) + Get(column+1, row  ) +
			                     Get(column, row+1) + Get(column+1, row+1));
			temp.Set(average, column, row);
		}
	}

	Copy(temp);
}
	
template<class T>
inline void Grid<T>::Add(const Grid<T>& a) {
	if (DimensionsMatch(a)) {
		#pragma omp parallel for
		for (INT_TYPE row = 0; row < num_rows_; row++) {
			for (INT_TYPE column = 0; column < num_columns_; column++) {
				if (Get(column, row) != (T)nodata_value_ &&
				    a.Get(column, row) != (T)a.nodata_value()) {
					T sum = Get(column, row) + a.Get(column, row);
					Set(sum, column, row);
				}
			}
		}
	} else {
		PrintErrorAndExit("'" + name_ + "' and '" + a.name() + "' cannot be added. " + 
		                  "Dimensions do not match.");
	}
}

template<class T>
inline void Grid<T>::Add(const T value) {
	#pragma omp parallel for
	for (INT_TYPE row = 0; row < num_rows_; row++) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			if (data_[row*num_columns_+column] != (T)nodata_value_) {
				data_[row*num_columns_+column] += value;
			}
		}
	}
}

template<class T>
inline void Grid<T>::Scale(const T value) {
	#pragma omp parallel for
	for (INT_TYPE row = 0; row < num_rows_; row++) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			if (data_[row*num_columns_+column] != (T)nodata_value_) {;
				data_[row*num_columns_+column] *= value;
			}
		}
	}
}

template<class T>
inline void Grid<T>::ReplaceValue(const T a, const T b) {
	#pragma omp parallel for
	for (INT_TYPE row = 0; row < num_rows_; row++) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			if (data_[row*num_columns_+column] == a) {
				data_[row*num_columns_+column] = b;
			}
		}
	}
}

template<class T>
inline void Grid<T>::AddBoundaries(void) {
	if (DimensionsArePositive()) {
		INT_TYPE num_columns_new = num_columns_ + 4;
		INT_TYPE num_rows_new = num_rows_ + 4;

		double x_lower_left_new = x_lower_left_ - 2.0*cellsize_;
		double y_lower_left_new = y_lower_left_ - 2.0*cellsize_;

		Grid<T> temp;
		temp.Initialize(num_columns_new, num_rows_new, x_lower_left_new,
		                y_lower_left_new, cellsize_, nodata_value_, name_);

		for (INT_TYPE row = 2; row < temp.num_rows()-2; row++) {
			for (INT_TYPE column = 2; column < temp.num_columns()-2; column++) {
				T value = Get(column-2, row-2);
				temp.Set(value, column, row);
			}
		}

		// Fill the top and bottom boundary cells.
		for (INT_TYPE column = 2; column < temp.num_columns()-2; column++) {
			T bottom = Get(column-2, 0);
			T bottom_p1 = Get(column-2, 1);
			T top = Get(column-2, num_rows_-1);
			T top_m1 = Get(column-2, num_rows_-2);

			temp.Set(top_m1, column, temp.num_rows()-2);
			temp.Set(top, column, temp.num_rows()-1);

			temp.Set(bottom, column, 0);
			temp.Set(bottom_p1, column, 1);
		}

		for (INT_TYPE row = 2; row < temp.num_rows()-2; row++) {
			T left = Get(0, row-2);
			T left_p1 = Get(1, row-2);
			T right = Get(num_columns_-1, row-2);
			T right_m1 = Get(num_columns_-2, row-2);

			temp.Set(right_m1, temp.num_columns()-2, row);
			temp.Set(right, temp.num_columns()-1, row);
			temp.Set(left, 0, row);
			temp.Set(left_p1, 1, row);
		}

		temp.set_name(name_);

		Copy(temp);
	} else {
		PrintErrorAndExit("Cannot add boundaries to '" + name_ + "'. Grid is empty.");
	}
}

template<class T>
inline T Grid<T>::Maximum(void) const {
	T maximum = std::numeric_limits<T>::min();

	for (INT_TYPE row = 0; row < num_rows_; row++) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			if (data_[row*num_columns_+column] != (T)nodata_value_) {
				maximum = std::max(data_[row*num_columns_+column], maximum);
			}
		}
	}

	return maximum;
}

template<class T>
inline T Grid<T>::Minimum(void) const {
	T minimum = std::numeric_limits<T>::max();

	for (INT_TYPE row = 0; row < num_rows_; row++) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			if (data_[row*num_columns_+column] != (T)nodata_value_) {
				minimum = std::min(data_[row*num_columns_+column], minimum);
			}
		}
	}

	return minimum;
}

template<class T>
inline long double Grid<T>::InnerSum(void) const {
	long double sum = 0.0;

	#pragma omp parallel for reduction(+:sum)
	for (INT_TYPE row = 2; row < num_rows_-2; row++) {
		for (INT_TYPE column = 2; column < num_columns_-2; column++) {
			if (data_[row*num_columns_+column] != (T)nodata_value_) {
				sum += (long double)data_[row*num_columns_+column];
			}
		}
	}

	return sum;
}

template<class T>
inline long double Grid<T>::Sum(void) const {
	long double sum = 0.0;

	#pragma omp parallel for reduction(+:sum)
	for (INT_TYPE row = 0; row < num_rows_; row++) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			if (data_[row*num_columns_+column] != (T)nodata_value_) {
				sum += (long double)data_[row*num_columns_+column];
			}
		}
	}

	return sum;
}

template<class T>
inline void Grid<T>::CheckIndexValidity(const INT_TYPE column,
                                        const INT_TYPE row) const {
	bool index_above_bounds = (column >= num_columns_) || (row >= num_rows_);
	if (index_above_bounds) {
		PrintErrorAndExit("Cannot get linearized index for grid '" + name_ +
		                  "'. Requested coordinate is out of bounds.");
	}
}

template<class T>
inline INT_TYPE Grid<T>::GetLinearIndex(const INT_TYPE column,
                                        const INT_TYPE row) const {
#ifdef DEBUG
	CheckIndexValidity(column, row);
#endif
	return row*num_columns_ + column;
}

template<class T>
inline INT_TYPE Grid<T>::GetLinearIndex(const double X,
                                        const double Y) const {
	INT_TYPE column = (INT_TYPE)((X - x_lower_left_) / cellsize_);
	INT_TYPE row = (INT_TYPE)((Y - y_lower_left_) / cellsize_);
	return GetLinearIndex(column, row);
}

template<class T>
inline INT_TYPE Grid<T>::GetXIndex(const double X) const {
	INT_TYPE column = (INT_TYPE)((X - x_lower_left_) / cellsize_);
	return column;
}

template<class T>
inline INT_TYPE Grid<T>::GetYIndex(const double Y) const {
	INT_TYPE row = (INT_TYPE)((Y - y_lower_left_) / cellsize_);
	return row;
}

template<class T>
inline double Grid<T>::GetXCoordinate(const INT_TYPE column) const {
	return x_lower_left_ + 0.5*cellsize_ + (double)column * cellsize_;
}

template<class T>
inline double Grid<T>::GetYCoordinate(const INT_TYPE row) const {
	return y_lower_left_ + 0.5*cellsize_ + (double)row * cellsize_;
}

template<class T>
inline T Grid<T>::Get(const INT_TYPE column, const INT_TYPE row) const {
	INT_TYPE linear_index = GetLinearIndex(column, row);
	return data_[linear_index];
}

template<class T>
inline T Grid<T>::Get(const INT_TYPE linear_index) const {
	return data_[linear_index];
}

template<class T>
inline T Grid<T>::Get(const double x, const double y) const {
	INT_TYPE linear_index = GetLinearIndex(x, y);
	return data_[linear_index];
}

template<class T>
inline T* Grid<T>::GetPointer(const INT_TYPE column, const INT_TYPE row) const {
	INT_TYPE linear_index = GetLinearIndex(column, row);
	return &data_[linear_index];
}

template<class T>
inline bool Grid<T>::DimensionsArePositive(void) const {
	return (num_columns_ > 0 && num_rows_ > 0);
}

template<class T>
inline bool Grid<T>::DimensionsMatch(const Grid& reference_grid) const {
	bool columns_match = num_columns_ == reference_grid.num_columns();
	bool rows_match = num_rows_ == reference_grid.num_rows();

	if (columns_match && rows_match) {
		return true;
	} else {
		return false;
	}
}

template<class T>
inline void Grid<T>::WriteResized(const Folder& folder,
                                  const float current_time) const {
	std::ostringstream file_path;
	file_path << folder.path() << name_ << "-" << current_time << "_s.asc";

	std::ofstream output;
	output.open((file_path.str()).c_str());

	INT_TYPE num_columns_new = num_columns_ - 3;
	INT_TYPE num_rows_new = num_rows_ - 3;

	double x_lower_left_new = x_lower_left_ + 1.5*cellsize_;
	double y_lower_left_new = y_lower_left_ + 1.5*cellsize_;

	output.precision(std::numeric_limits<double>::digits10);
	output << "ncols         " << num_columns_new  << std::endl;
	output << "nrows         " << num_rows_new     << std::endl;
	output << "xllcorner     " << x_lower_left_new << std::endl;
	output << "yllcorner     " << y_lower_left_new << std::endl;
	output << "cellsize      " << cellsize_        << std::endl;
	output << "NODATA_value  " << nodata_value_    << std::endl;

	output.precision(std::numeric_limits<float>::digits10);
	for (INT_TYPE row = num_rows_new; row-- > 0;) {
		for (INT_TYPE column = 0; column < num_columns_new; column++) {
			INT_TYPE ll = (row+2)*num_columns_ + (column+2);
			INT_TYPE lr = (row+2)*num_columns_ + (column+3);
			INT_TYPE ul = (row+3)*num_columns_ + (column+2);
			INT_TYPE ur = (row+3)*num_columns_ + (column+3);

			T average = (T)(0.25)*(data_[ll] + data_[lr] + data_[ul] + data_[ur]);
			output << average;

			if (column < num_columns_new - 1) {
				output << " ";
			}
		}

		if (row > 0) {
			output << std::endl;
		}
	}

	output.close();
}

template<class T>
inline void Grid<T>::WriteResized(const Folder& folder) const {
	std::ostringstream file_path;
	file_path << folder.path() << name_ << ".asc";

	std::ofstream output;
	output.open((file_path.str()).c_str());

	INT_TYPE num_columns_new = num_columns_ - 3;
	INT_TYPE num_rows_new = num_rows_ - 3;

	double x_lower_left_new = x_lower_left_ + 1.5*cellsize_;
	double y_lower_left_new = y_lower_left_ + 1.5*cellsize_;

	output.precision(std::numeric_limits<double>::digits10);
	output << "ncols         " << num_columns_new  << std::endl;
	output << "nrows         " << num_rows_new     << std::endl;
	output << "xllcorner     " << x_lower_left_new << std::endl;
	output << "yllcorner     " << y_lower_left_new << std::endl;
	output << "cellsize      " << cellsize_        << std::endl;
	output << "NODATA_value  " << nodata_value_    << std::endl;

	output.precision(std::numeric_limits<float>::digits10);
	for (INT_TYPE row = num_rows_new; row-- > 0;) {
		for (INT_TYPE column = 0; column < num_columns_new; column++) {
			INT_TYPE ll = (row+2)*num_columns_ + (column+2);
			INT_TYPE lr = (row+2)*num_columns_ + (column+3);
			INT_TYPE ul = (row+3)*num_columns_ + (column+2);
			INT_TYPE ur = (row+3)*num_columns_ + (column+3);

			T average = (T)(0.25)*(data_[ll] + data_[lr] + data_[ul] + data_[ur]);
			output << average;

			if (column < num_columns_new - 1) {
				output << " ";
			}
		}

		if (row > 0) {
			output << std::endl;
		}
	}

	output.close();
}

template<class T>
inline void Grid<T>::Write(const std::string filename) const {
	std::ofstream output;
	output.open(filename.c_str());

	output.precision(std::numeric_limits<double>::digits10);
	output << "ncols         " << num_columns_  << std::endl;
	output << "nrows         " << num_rows_     << std::endl;
	output << "xllcorner     " << x_lower_left_ << std::endl;
	output << "yllcorner     " << y_lower_left_ << std::endl;
	output << "cellsize      " << cellsize_     << std::endl;
	output << "NODATA_value  " << nodata_value_ << std::endl;

	output.precision(std::numeric_limits<float>::digits10);
	for (INT_TYPE row = num_rows_; row-- > 0;) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			output << data_[row*num_columns_+column];

			if (column < num_columns_ - 1) {
				output << " ";
			}
		}

		if (row > 0) {
			output << std::endl;
		}
	}

	output.close();
}

template<class T>
inline void Grid<T>::Write(const Folder& folder,
                           const float current_time) const {
	std::ostringstream file_path;
	file_path << folder.path() << name_ << "-" << current_time << "_s.asc";

	std::ofstream output;
	output.open((file_path.str()).c_str());

	output.precision(std::numeric_limits<double>::digits10);
	output << "ncols         " << num_columns_  << std::endl;
	output << "nrows         " << num_rows_     << std::endl;
	output << "xllcorner     " << x_lower_left_ << std::endl;
	output << "yllcorner     " << y_lower_left_ << std::endl;
	output << "cellsize      " << cellsize_     << std::endl;
	output << "NODATA_value  " << nodata_value_ << std::endl;

	output.precision(std::numeric_limits<float>::digits10);
	for (INT_TYPE row = num_rows_; row-- > 0;) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			output << data_[row*num_columns_+column];

			if (column < num_columns_ - 1) {
				output << " ";
			}
		}

		if (row > 0) {
			output << std::endl;
		}
	}

	output.close();
}

template<class T>
inline void Grid<T>::Write(const Folder& folder) const {
	std::ostringstream file_path;
	file_path << folder.path() << name_ << ".asc";

	std::ofstream output;
	output.open((file_path.str()).c_str());

	output.precision(std::numeric_limits<double>::digits10);
	output << "ncols         " << num_columns_  << std::endl;
	output << "nrows         " << num_rows_     << std::endl;
	output << "xllcorner     " << x_lower_left_ << std::endl;
	output << "yllcorner     " << y_lower_left_ << std::endl;
	output << "cellsize      " << cellsize_     << std::endl;
	output << "NODATA_value  " << nodata_value_ << std::endl;

	output.precision(std::numeric_limits<float>::digits10);
	for (INT_TYPE row = num_rows_; row-- > 0;) {
		for (INT_TYPE column = 0; column < num_columns_; column++) {
			output << data_[row*num_columns_+column];

			if (column < num_columns_ - 1) {
				output << " ";
			}
		}

		if (row > 0) {
			output << std::endl;
		}
	}

	output.close();
}

template<class T>
inline void Grid<T>::WriteWithoutBoundaries(const Folder& folder) const {
	std::ostringstream file_path;
	file_path << folder.path() << name_ << ".asc";

	std::ofstream output;
	output.open((file_path.str()).c_str());

	output.precision(std::numeric_limits<double>::digits10);
	output << "ncols         " << num_columns_ - 4              << std::endl;
	output << "nrows         " << num_rows_ - 4                 << std::endl;
	output << "xllcorner     " << x_lower_left_ + 2.0*cellsize_ << std::endl;
	output << "yllcorner     " << y_lower_left_ + 2.0*cellsize_ << std::endl;
	output << "cellsize      " << cellsize_                     << std::endl;
	output << "NODATA_value  " << nodata_value_                 << std::endl;

	output.precision(std::numeric_limits<float>::digits10);
	for (INT_TYPE row = num_rows_ - 2; row-- > 2;) {
		for (INT_TYPE column = 2; column < num_columns_ - 2; column++) {
			output << data_[row*num_columns_+column] << " ";
		}

		output << std::endl;
	}

	output.close();
}

template<class T>
inline void Grid<T>::WriteWithoutBoundaries(const Folder& folder,
                                            const float current_time) const {
	std::ostringstream file_path;
	file_path << folder.path() << name_ << "-" << current_time << "_s.asc";

	std::ofstream output;
	output.open((file_path.str()).c_str());

	output.precision(std::numeric_limits<double>::digits10);
	output << "ncols         " << num_columns_ - 4              << std::endl;
	output << "nrows         " << num_rows_ - 4                 << std::endl;
	output << "xllcorner     " << x_lower_left_ + 2.0*cellsize_ << std::endl;
	output << "yllcorner     " << y_lower_left_ + 2.0*cellsize_ << std::endl;
	output << "cellsize      " << cellsize_                     << std::endl;
	output << "NODATA_value  " << nodata_value_                 << std::endl;

	output.precision(std::numeric_limits<float>::digits10);
	for (INT_TYPE row = num_rows_ - 2; row-- > 2;) {
		for (INT_TYPE column = 2; column < num_columns_ - 2; column++) {
			output << data_[row*num_columns_+column] << " ";
		}

		output << std::endl;
	}

	output.close();
}

template<class T>
inline void Grid<T>::EqualsDifferenceOf(const Grid<T>& b, const Grid<T>& a) {
	for (INT_TYPE j = 0; j < num_rows_; j++) {
		for (INT_TYPE i = 0; i < num_columns_; i++) {
			T value = b.Get(i, j) - a.Get(i, j);
			Set(value, i, j);
		}
	}
}
