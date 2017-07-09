#pragma once

#include <algorithm>
#include <iostream>
#include <cpl_conv.h>
#include <gdal_priv.h>
#include "error.h"

//! Default Raster implementation.
/*! Encapsulates native GDAL functionality.
    \tparam Type of raster data. Default is double.
*/
template<class T = double>
class Raster {
public:
	Raster(std::string path, GDALAccess access = GA_ReadOnly);
	Raster(const Raster& raster, std::string path, GDALAccess access = GA_ReadOnly);
	~Raster(void);

	void Fill(T value);
	void Update(void) const;
	bool EqualDimensions(const Raster<T>& raster) const;

	T* array(void) const { return array_; }
	GDALDataset* dataset(void) const { return dataset_; }
	int height(void) const { return dataset_->GetRasterBand(1)->GetYSize(); }
	int width(void) const { return dataset_->GetRasterBand(1)->GetXSize(); }
	double cellsize_x(void) const { return geo_transform_[1]; }
	double cellsize_y(void) const { return -geo_transform_[5]; }
	std::string path(void) const { return path_; }
	int index(double x, double y) const;

protected:
	T* array_;
	GDALDataset* dataset_;
	double geo_transform_[6];
	std::string path_;
};

//! Constructor for Raster.
/*! \tparam Type of raster data. Default is double.
	 \param path Path to raster file.
	 \param access <a href="http://www.gdal.org/gdal_8h.html#a045e3967c208993f70257bfd40c9f1d7">
	               Flag indicating read/write, or read-only access to raster</a>
*/
template<class T>
inline Raster<T>::Raster(std::string path, GDALAccess access) {
	path_ = path;
	GDALAllRegister();
	dataset_ = (GDALDataset*)GDALOpen(path_.c_str(), access);
	dataset_->GetGeoTransform(geo_transform_);

	if (dataset_ == NULL) {
		std::string error_message = "Raster dataset \"" + path_ + "\" is invalid.";
		std::cerr << "ERROR: " << error_message << std::endl;
		std::exit(3);
	}

	GDALRasterBand* band = dataset_->GetRasterBand(1);
	array_ = (T*)CPLMalloc(width()*height()*sizeof(T));
	CPLErrChk(band->RasterIO(GF_Read, 0, 0, width(), height(), array_,
	                         width(), height(), GDT_Float32, 0, 0));
}

//! Constructor for Raster.
/*! \tparam Type of raster data. Default is double.
	 \param raster Raster from which data is copied.
	 \param path Path to write Raster.
	 \param access <a href="http://www.gdal.org/gdal_8h.html#a045e3967c208993f70257bfd40c9f1d7">
	               Flag indicating read/write, or read-only access to raster</a>
*/
template<class T>
inline Raster<T>::Raster(const Raster& raster, std::string path, GDALAccess access) {
	path_ = path;
	GDALDataset* src = raster.dataset();
	GDALDriver* driver = src->GetDriver();
	dataset_ = driver->CreateCopy(path_.c_str(), src, false, NULL, NULL, NULL);
	dataset_->GetGeoTransform(geo_transform_);
	array_ = (T*)CPLMalloc(width()*height()*sizeof(T));
	memcpy(array_, raster.array(), width()*height()*sizeof(T));
}

//! Destructor for Raster.
/*! \tparam Type of raster data. */
template<class T>
inline Raster<T>::~Raster(void) {
	CPLFree(array_);
	GDALClose(dataset_);
}

template<class T>
inline int Raster<T>::index(double x, double y) const {
	double transform[6];
	dataset_->GetGeoTransform(transform);
	double inv_transform[6];
	bool success = GDALInvGeoTransform(transform, inv_transform);

	if (!success) {
		std::string error_message = "Raster inverse geotransform for \"" + path_ + "\" failed.";
		std::cerr << "ERROR: " << error_message << std::endl;
		std::exit(4);
	}

	int row = floor(inv_transform[3] + inv_transform[4] * x + inv_transform[5] * y);
	int column = floor(inv_transform[0] + inv_transform[1] * x + inv_transform[2] * y);
	return row * width() + column;
}

template<class T>
inline void Raster<T>::Fill(T value) {
	std::fill_n(array_, width() * height(), value);
}

template<class T>
inline void Raster<T>::Update(void) const {
	GDALRasterBand* band = dataset_->GetRasterBand(1);
	CPLErrChk(band->RasterIO(GF_Write, 0, 0, width(), height(), array_,
	                         width(), height(), GDT_Float32, 0, 0));
}

template<class T>
inline bool Raster<T>::EqualDimensions(const Raster<T>& raster) const {
	if (raster.width() == Raster<T>::width() &&
	    raster.height() == Raster<T>::height()) {
		return true;
	} else {
		return false;
	}
}
