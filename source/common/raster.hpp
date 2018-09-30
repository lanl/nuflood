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

	// Functions.
	void Add(const Raster<T>& reference);
	void Subtract(const Raster<T>& reference);
	T GetFromIndex(const INT_TYPE i) const;
	T GetFromCoordinates(const double x, const double y) const;
	T GetFromIndices(const INT_TYPE i, const INT_TYPE j) const;
	void CopyFrom(const Raster<T>& raster);
	void Read(std::string path, GDALAccess access = GA_ReadOnly);
	void Fill(T value);
	void Update(void);
	void Resample(void);
	void Write(const std::string path);
	bool EqualDimensions(const Raster<T>& raster) const;

	// Setters.
	void set_name(const std::string name) { name_ = name; }
	void SetAtIndex(INT_TYPE i, T value);
	void SetAtCoordinates(const double x, const double y, T value);
	void SetAtIndices(const INT_TYPE i, const INT_TYPE j, T value);

	// Getters.
	T* array(void) const { return array_; }
	T nodata(void) const { return nodata_; }
	GDALDataset* dataset(void) const { return dataset_; }
	INT_TYPE height(void) const { return dataset_->GetRasterBand(1)->GetYSize(); }
	INT_TYPE width(void) const { return dataset_->GetRasterBand(1)->GetXSize(); }
	INT_TYPE num_pixels(void) const { return width() * height(); }
	double cellsize_x(void) const { return geo_transform_[1]; }
	double cellsize_y(void) const { return -geo_transform_[5]; }
	std::string name(void) const { return name_; }
	std::string path(void) const { return path_; }
	INT_TYPE index(double x, double y) const;

protected:
	T* array_ = nullptr;
	T nodata_ = (T)(-9999);
	GDALDataset* dataset_ = nullptr;
	double geo_transform_[6] = {0, 0, 0, 0, 0, 0};
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
	 \param access <a href="http://www.gdal.org/gdal_8h.html#a045e3967c208993f70257bfd40c9f1d7">
	               Flag indicating read/write, or read-only access to raster</a>
*/
template<class T>
inline Raster<T>::Raster(std::string path, std::string name, GDALAccess access) {
	Raster<T>::Read(path);
	name_ = name;
}

//! Constructor for Raster.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster from which data is copied.
	 \param path Path to write Raster.
	 \param access <a href="http://www.gdal.org/gdal_8h.html#a045e3967c208993f70257bfd40c9f1d7">
	               Flag indicating read/write, or read-only access to raster</a>
*/
template<class T>
inline Raster<T>::Raster(const Raster& raster, std::string path, std::string name) {
	name_ = name;
	path_ = path;
	nodata_ = raster.nodata();
	GDALDataset* src = raster.dataset();
	GDALDriver* driver = src->GetDriver();
	dataset_ = driver->CreateCopy(path_.c_str(), src, false, NULL, NULL, NULL);
	dataset_->GetGeoTransform(geo_transform_);
	array_ = (T*)CPLMalloc(width()*height()*sizeof(T));
	memcpy(array_, raster.array(), width()*height()*sizeof(T));
}

//! Constructor for Raster.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster from which data is copied.
	 \param access <a href="http://www.gdal.org/gdal_8h.html#a045e3967c208993f70257bfd40c9f1d7">
	               Flag indicating read/write, or read-only access to raster</a>
*/
template<class T>
inline Raster<T>::Raster(const Raster& raster, std::string name) {
	Raster<T>::CopyFrom(raster);
	name_ = name;
}

//! Copy another raster's data to this Raster.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster from which data is copied.
	 \param access <a href="http://www.gdal.org/gdal_8h.html#a045e3967c208993f70257bfd40c9f1d7">
	               Flag indicating read/write, or read-only access to raster</a>
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
		dataset_ = driver->CreateCopy("", src, false, NULL, NULL, NULL);
	} else {
		driver = src->GetDriver();
		dataset_ = driver->CreateCopy(path_.c_str(), src, false, NULL, NULL, NULL);
	}

	dataset_->GetGeoTransform(geo_transform_);
	array_ = (T*)CPLMalloc(width()*height()*sizeof(T));
	memcpy(array_, raster.array(), width()*height()*sizeof(T));
}

//! Reads in a raster from a file path, overwriting any existing data.
/*! \tparam Type of raster data. Default is double.
    \param path Path to raster file.
    \param access <a href="http://www.gdal.org/gdal_8h.html#a045e3967c208993f70257bfd40c9f1d7">
           Flag indicating read/write, or read-only access to raster</a>
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
	 \param x Raster from which data is copied.
	 \param y x-coordinate of the 
*/
template<class T>
inline INT_TYPE Raster<T>::index(double x, double y) const {
	double transform[6];
	dataset_->GetGeoTransform(transform);
	double inv_transform[6];
	bool success = GDALInvGeoTransform(transform, inv_transform);

	INT_TYPE i = floor(inv_transform[3] + inv_transform[4] * x + inv_transform[5] * y);
	INT_TYPE j = floor(inv_transform[0] + inv_transform[1] * x + inv_transform[2] * y);
	INT_TYPE id = i * width() + j;

	if (id >= 0 && id < num_pixels()) {
		return id;
	} else {
		std::string error_string = "Index out of bounds for \"" + name_ + "\".";
		throw std::system_error(std::error_code(), error_string);
	}
}

//! Fills all pixels of a raster with a single value.
/*! \tparam Type of raster data. Default is double.
	 \param value Value with which to populate all raster cells.
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

//! Updates the raster file on disk using current data from `array_`.
/*! \tparam Type of raster data. Default is double.
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

//! Returns if the dimensions of the current raster are equivalent to another.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster used for the comparison.
*/
template<class T>
inline T Raster<T>::GetFromCoordinates(const double x, const double y) const {
	INT_TYPE i = Raster<T>::index(x, y);
	return array_[i];
}

//! Returns if the dimensions of the current raster are equivalent to another.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster used for the comparison.
*/
template<class T>
inline T Raster<T>::GetFromIndices(const INT_TYPE i, const INT_TYPE j) const {
	INT_TYPE ij = i * width() + j;
	return array_[ij];
}

//! Returns if the dimensions of the current raster are equivalent to another.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster used for the comparison.
*/
template<class T>
inline T Raster<T>::GetFromIndex(const INT_TYPE i) const {
	return array_[i];
}

//! Returns if the dimensions of the current raster are equivalent to another.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster used for the comparison.
*/
template<class T>
inline void Raster<T>::SetAtIndex(const INT_TYPE i, const T value) {
	array_[i] = value;
}


//! Returns if the dimensions of the current raster are equivalent to another.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster used for the comparison.
*/
template<class T>
inline void Raster<T>::SetAtCoordinates(const double x, const double y, T value) {
	INT_TYPE i = Raster<T>::index(x, y);
	array_[i] = value;
}

//! Returns if the dimensions of the current raster are equivalent to another.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster used for the comparison.
*/
template<class T>
inline void Raster<T>::SetAtIndices(const INT_TYPE i, const INT_TYPE j, T value) {
	INT_TYPE ij = i * width() + j;
	array_[ij] = value;
}

template<class T>
inline void Raster<T>::Add(const Raster<T>& reference) {
	T nodata_ref = (T)reference.nodata();

	if (Raster<T>::EqualDimensions(reference)) {
		#pragma omp parallel for
		for (INT_TYPE i = 0; i < num_pixels(); i++) {
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

template<class T>
inline void Raster<T>::Subtract(const Raster<T>& reference) {
	T nodata_ref = (T)reference.nodata();

	if (Raster<T>::EqualDimensions(reference)) {
		#pragma omp parallel for
		for (INT_TYPE i = 0; i < num_pixels(); i++) {
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
