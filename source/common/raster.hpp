#pragma once

#include <algorithm>
#include <iostream>
#include <cpl_conv.h>
#include <gdal_priv.h>
#include "error.h"

template<class T>
class Raster {
public:
	Raster(std::string path, GDALAccess access = GA_ReadOnly, std::string name = "");
	Raster(const Raster& raster, std::string path, GDALAccess access = GA_ReadOnly, std::string name = "");
	~Raster(void);

	void Fill(T value);
	void Update(void) const;
	bool EqualDimensions(const Raster<T>& raster) const;

	T* get_array(void) const { return array_; }
	GDALDataset* get_dataset(void) const { return dataset_; }
	int get_height(void) const { return dataset_->GetRasterBand(1)->GetYSize(); }
	int get_width(void) const { return dataset_->GetRasterBand(1)->GetXSize(); }
	int get_index(double x, double y) const;

protected:
	T* array_;
	GDALDataset* dataset_;
	std::string name_;
};

template<class T>
inline Raster<T>::Raster(std::string path, GDALAccess access, std::string name) {
	name_ = name;

	GDALAllRegister();
	dataset_ = (GDALDataset*)GDALOpen(path.c_str(), access);

	if (dataset_ == NULL) {
		// TODO: Throw error here.
	}

	GDALRasterBand* band = dataset_->GetRasterBand(1);
	array_ = (T*)CPLMalloc(get_width()*get_height()*sizeof(T));

	CPLErrChk(band->RasterIO(GF_Read, 0, 0, get_width(), get_height(), array_,
                            get_width(), get_height(), GDT_Float32, 0, 0));
}

template<class T>
inline Raster<T>::Raster(const Raster& raster, std::string path, GDALAccess access, std::string name) {
	name_ = name;
	GDALDataset* src = raster.get_dataset();
	GDALDriver* driver = src->GetDriver();
	dataset_ = driver->CreateCopy(path.c_str(), src, false, NULL, NULL, NULL);
	array_ = (T*)CPLMalloc(get_width()*get_height()*sizeof(T));
	memcpy(array_, raster.get_array(), get_width()*get_height()*sizeof(T));
}

template<class T>
inline Raster<T>::~Raster(void) {
	CPLFree(array_);
	GDALClose(dataset_);
}

template<class T>
inline int Raster<T>::get_index(double x, double y) const {
	double transform[6];
	dataset_->GetGeoTransform(transform);
	double inv_transform[6];
	bool success = GDALInvGeoTransform(transform, inv_transform);

	int row = floor(inv_transform[3] + inv_transform[4] * x + inv_transform[5] * y);
	int column = floor(inv_transform[0] + inv_transform[1] * x + inv_transform[2] * y);
	return row * get_width() + column;
}

template<class T>
inline void Raster<T>::Fill(T value) {
	std::fill_n(array_, get_width() * get_height(), value);
}

template<class T>
inline void Raster<T>::Update(void) const {
	GDALRasterBand* band = dataset_->GetRasterBand(1);
	band->RasterIO(GF_Write, 0, 0, get_width(), get_height(), array_,
		            get_width(), get_height(), GDT_Float32, 0, 0);
}

template<class T>
inline bool Raster<T>::EqualDimensions(const Raster<T>& raster) const {
	if (raster.get_width() == Raster<T>::get_width() &&
		 raster.get_height() == Raster<T>::get_height()) {
		return true;
	} else {
		return false;
	}
}
