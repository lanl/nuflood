#pragma once

#include <algorithm>
#include <iostream>
#include <cpl_conv.h>
#include <gdal_priv.h>
#include <gdalwarper.h>
#include <ogr_spatialref.h>
#include <typeinfo>
#include "precision.h"
#include "error.h"

//! Default Raster implementation.
/*! Encapsulates native GDAL functionality.
    \tparam Type of raster data. Default is double.
*/
template<class T = double>
class Raster {
public:
	// Constructors.
	Raster(void);
	Raster(std::string path, std::string name, GDALAccess access = GA_ReadOnly);
	Raster(const Raster& raster, std::string path, std::string name);
	Raster(const Raster& raster, std::string name);

	// Destructor.
	~Raster(void);

	// Functions to read and initialize raster data.
	void CopyFrom(const Raster<T>& raster);
	void Read(std::string path, GDALAccess access = GA_ReadOnly);

	// Functions that update and write raster data.
	void Update(void);
	void Write(const std::string path);

	// Functions to get pixel values and indices.
	T GetFromIndex(const int_t i) const;
	T GetFromCoordinates(const double x, const double y) const;
	T GetFromIndices(const int_t i, const int_t j) const;
	int_t index(double x, double y) const;

	// Functions to set pixel values.
	void SetAtIndex(int_t index, T value);
	void SetAtCoordinates(const double x, const double y, T value);
	void SetAtIndices(const int_t i, const int_t j, T value);

	// Functions that modify all pixel values.
	void Add(const Raster<T>& reference);
	void Subtract(const Raster<T>& reference);
	void Fill(T value);

	// Functions to check raster validity.
	bool EqualDimensions(const Raster<T>& raster) const;

	// Trivial getters.
	T* array(void) const { return array_; }
	T nodata(void) const { return nodata_; }
	GDALDataset* dataset(void) const { return dataset_; }
	int_t height(void) const { return dataset_->GetRasterBand(1)->GetYSize(); }
	int_t width(void) const { return dataset_->GetRasterBand(1)->GetXSize(); }
	int_t num_pixels(void) const { return width() * height(); }
	double cellsize_x(void) const { return geo_transform_[1]; }
	double cellsize_y(void) const { return -geo_transform_[5]; }
	std::string name(void) const { return name_; }
	std::string path(void) const { return path_; }

protected:
	T* array_ = nullptr;
	T nodata_ = (T)(-9999);
	GDALDataset* dataset_ = nullptr;
	double geo_transform_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	std::string name_ = "";
	std::string path_ = "";
};

//! Constructor for Raster.
/*! \tparam Type of raster data. Default is double. */
template<class T>
inline Raster<T>::Raster(void) {
	array_ = nullptr;
	dataset_ = nullptr;
	nodata_ = (T)(-9999);
	path_ = name_ = "";

	for (int i = 0; i < 6; i++) {
		geo_transform_[i] = 0;
	}
}

//! Constructor for Raster.
/*! \tparam Type of raster data. Default is double.
	 \param path Path to raster file.
	 \param name Name of the raster.
	 \param access <a href="http://www.gdal.org/gdal_8h.html#a045e3967c208993f70257bfd40c9f1d7">
	               Flag indicating read/write, or read-only access to raster</a>
*/
template<class T>
inline Raster<T>::Raster(std::string path, std::string name, GDALAccess access) {
	Raster<T>::Read(path, access);
	name_ = name;
}

//! Constructor for Raster.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster from which data is copied.
	 \param path Path to write Raster.
	 \param name Name of the raster.
*/
template<class T>
inline Raster<T>::Raster(const Raster& raster, std::string path, std::string name) {
	name_ = name;
	path_ = path;
	nodata_ = raster.nodata();
	GDALDataset* src = raster.dataset();
	GDALDriver* driver = src->GetDriver();
	dataset_ = driver->CreateCopy(path_.c_str(), src, FALSE, NULL, NULL, NULL);
	dataset_->GetGeoTransform(geo_transform_);
	array_ = (T*)CPLMalloc(width()*height()*sizeof(T));
	memcpy(array_, raster.array(), width()*height()*sizeof(T));
}

//! Constructor for Raster.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster from which data is copied.
	 \param name Name of the raster.
*/
template<class T>
inline Raster<T>::Raster(const Raster& raster, std::string name) {
	Raster<T>::CopyFrom(raster);
	name_ = name;
}

//! Copy another raster's data to this Raster.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster from which data is copied.
*/
template<class T>
inline void Raster<T>::CopyFrom(const Raster& raster) {
	if (array_ != nullptr) {
		CPLFree(array_);
		GDALClose(dataset_);
		dataset_ = nullptr;
	}

	name_ = raster.name();
	nodata_ = raster.nodata();
	GDALDataset* src = raster.dataset();
	GDALDriver* driver;

	if (path_.empty()) {
		driver = GetGDALDriverManager()->GetDriverByName("MEM");
		dataset_ = driver->CreateCopy("", src, FALSE, NULL, NULL, NULL);
	} else {
		driver = src->GetDriver();
		dataset_ = driver->CreateCopy(path_.c_str(), src, FALSE, NULL, NULL, NULL);
	}

	dataset_->GetGeoTransform(geo_transform_);
	array_ = (T*)CPLMalloc(width()*height()*sizeof(T));
	memcpy(array_, raster.array(), width()*height()*sizeof(T));
}

//! Reads in a raster from a file path, overwriting any existing data.
/*! \tparam Type of raster data. Default is double.
    \param path Path to raster file.
    \param access <a href="http://www.gdal.org/gdal_8h.html#a045e3967c208993f70257bfd40c9f1d7">
           Flag indicating read/write or read-only access to raster</a>
*/
template<class T>
inline void Raster<T>::Read(std::string path, GDALAccess access) {
	if (array_ != nullptr) {
		CPLFree(array_);
		GDALClose(dataset_);
		dataset_ = nullptr;
	}

	path_ = path;
	GDALAllRegister();
	dataset_ = (GDALDataset*)GDALOpen(path_.c_str(), access);

	if (dataset_ == NULL) {
		std::string error_string = "Raster dataset \"" + path_ + "\" is invalid.";
		throw std::system_error(std::error_code(), error_string);
	}

	dataset_->GetGeoTransform(geo_transform_);
	GDALRasterBand* band = dataset_->GetRasterBand(1);
	nodata_ = (T)band->GetNoDataValue();
	array_ = (T*)CPLMalloc(width()*height()*sizeof(T));
	GDALDataType gdt = typeid(T) == typeid(double) ? GDT_Float64 : GDT_Float32;
	CPLErrChk(band->RasterIO(GF_Read, 0, 0, width(), height(), array_,
	                         width(), height(), gdt, 0, 0));
}

//! Destructor for Raster.
/*! \tparam Type of raster data. */
template<class T>
inline Raster<T>::~Raster(void) {
	if (array_ != nullptr) {
		CPLFree(array_);
		GDALClose(dataset_);
	}
}

//! Returns the flattened raster pixel index for a given location.
/*! \tparam Type of raster data. Default is double.
	 \param x x-coordinate of the pixel.
	 \param y y-coordinate of the pixel.
*/
template<class T>
inline int_t Raster<T>::index(double x, double y) const {
	double transform[6], inv_transform[6];
	dataset_->GetGeoTransform(transform);
	bool success = GDALInvGeoTransform(transform, inv_transform);

	if (!success) {
		std::string error_string = "Cannot invert GeoTransform coefficients.";
		throw std::system_error(std::error_code(), error_string);
	}

	int_t i = floor(inv_transform[3] + inv_transform[4] * x + inv_transform[5] * y);
	int_t j = floor(inv_transform[0] + inv_transform[1] * x + inv_transform[2] * y);
	int_t id = i * width() + j;

	if (id >= 0 && id < num_pixels()) {
		return id;
	} else {
		std::string error_string = "Index out of bounds for \"" + name_ + "\".";
		throw std::system_error(std::error_code(), error_string);
	}
}

//! Fills all pixels of a raster with a single value.
/*! \tparam Type of raster data. Default is double.
	 \param value Value with which to populate all pixels.
*/
template<class T>
inline void Raster<T>::Fill(T value) {
	std::fill_n(array_, width() * height(), value);
}

//! Updates the raster file on disk using current data from `array_`.
/*! \tparam Type of raster data. Default is double.
*/
template<class T>
inline void Raster<T>::Update(void) {
	GDALRasterBand* band = dataset_->GetRasterBand(1);
	GDALDataType gdt = typeid(T) == typeid(double) ? GDT_Float64 : GDT_Float32;
	CPLErrChk(band->RasterIO(GF_Write, 0, 0, width(), height(), array_,
	                         width(), height(), gdt, 0, 0));
}

//! Write raster data to the specified path.
/*! \tparam Type of raster data. Default is double.
	 \param path Path to the which the raster will be written.
*/
template<class T>
inline void Raster<T>::Write(const std::string path) {
	char **options = nullptr;
	options = CSLSetNameValue(options, "COMPRESS", "LZW");
	options = CSLSetNameValue(options, "NUM_THREADS", "ALL_CPUS");
	GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
	GDALDataset* dataset = driver->CreateCopy(path.c_str(), dataset_, FALSE, NULL, NULL, NULL);
	GDALRasterBand* band = dataset->GetRasterBand(1);
	GDALDataType gdt = typeid(T) == typeid(double) ? GDT_Float64 : GDT_Float32;
	CPLErrChk(band->RasterIO(GF_Write, 0, 0, width(), height(), array_, width(), height(), gdt, 0, 0));
	GDALClose((GDALDatasetH)dataset);
	CSLDestroy(options);
}

//! Returns if the dimensions of the current raster are equivalent to another.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster used for the comparison.
*/
template<class T>
inline bool Raster<T>::EqualDimensions(const Raster<T>& raster) const {
	return raster.width() == width() && raster.height() == height();
}

//! Returns the pixel value at the specified coordinates.
/*! \tparam Type of raster data. Default is double.
	 \param x x-coordinate of the pixel.
	 \param y y-coordinate of the pixel.
*/
template<class T>
inline T Raster<T>::GetFromCoordinates(const double x, const double y) const {
	int_t i = Raster<T>::index(x, y);
	return array_[i];
}

//! Returns the pixel value at the specified indices.
/*! \tparam Type of raster data. Default is double.
	 \param i Row index of the pixel.
	 \param j Column index of the pixel.
*/
template<class T>
inline T Raster<T>::GetFromIndices(const int_t i, const int_t j) const {
	int_t ij = i * width() + j;
	return array_[ij];
}

//! Returns the pixel value at the specified flattened index.
/*! \tparam Type of raster data. Default is double.
	 \param index Flattened index of the pixel (i.e., row * width + column).
*/
template<class T>
inline T Raster<T>::GetFromIndex(const int_t index) const {
	return array_[index];
}

//! Sets the pixel value at the specified coordinates.
/*! \tparam Type of raster data. Default is double.
	 \param x x-coordinate of the pixel.
	 \param y y-coordinate of the pixel.
	 \param value New value of the pixel.
*/
template<class T>
inline void Raster<T>::SetAtCoordinates(const double x, const double y, T value) {
	int_t i = Raster<T>::index(x, y);
	array_[i] = value;
}

//! Sets the pixel value at the specified indices.
/*! \tparam Type of raster data. Default is double.
	 \param i Row index of the pixel.
	 \param j Column index of the pixel.
	 \param value New value of the pixel.
*/
template<class T>
inline void Raster<T>::SetAtIndices(const int_t i, const int_t j, T value) {
	int_t ij = i * width() + j;
	array_[ij] = value;
}

//! Sets the pixel value at the specified flattened index.
/*! \tparam Type of raster data. Default is double.
	 \param index Flattened index of the pixel (i.e., row * width + column).
	 \param value New value of the pixel.
*/
template<class T>
inline void Raster<T>::SetAtIndex(const int_t index, const T value) {
	array_[index] = value;
}

//! Elementwise addition of the raster with another.
/*! \tparam Type of raster data. Default is double.
	 \param reference Raster from which pixel values will be elementwise added.
*/
template<class T>
inline void Raster<T>::Add(const Raster<T>& reference) {
	T nodata_ref = (T)reference.nodata();

	if (Raster<T>::EqualDimensions(reference)) {
		#pragma omp parallel for
		for (int_t i = 0; i < num_pixels(); i++) {
			if (Raster<T>::GetFromIndex(i) != nodata_ && (T)reference.GetFromIndex(i) != nodata_ref) {
				array_[i] += (T)reference.GetFromIndex(i);
			}
		}
	} else {
		std::string error_string = "'" + name_ + "' and '" + reference.name() +
		                           "' cannot be added. Dimensions do not match.";
		throw std::system_error(std::error_code(), error_string);
	}
}

//! Elementwise subtraction of the raster with another.
/*! \tparam Type of raster data. Default is double.
	 \param reference Raster from which pixel values will be elementwise subtracted.
*/
template<class T>
inline void Raster<T>::Subtract(const Raster<T>& reference) {
	T nodata_ref = (T)reference.nodata();

	if (Raster<T>::EqualDimensions(reference)) {
		#pragma omp parallel for
		for (int_t i = 0; i < num_pixels(); i++) {
			if (Raster<T>::GetFromIndex(i) != nodata_ && (T)reference.GetFromIndex(i) != nodata_ref) {
				array_[i] -= (T)reference.GetFromIndex(i);
			}
		}
	} else {
		std::string error_string = "'" + name_ + "' and '" + reference.name() +
		                           "' cannot be subtracted. Dimensions do not match.";
		throw std::system_error(std::error_code(), error_string);
	}
}
