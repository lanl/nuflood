#pragma once

#include <iostream>
#include <cpl_conv.h>
#include <gdal_priv.h>

template<class T>
class Raster {
public:
	Raster(std::string path, GDALAccess access = GA_ReadOnly);
	Raster(const Raster& raster, std::string path, GDALAccess access = GA_ReadOnly);
	~Raster(void);

	void Update(void) const;

	T* get_array(void) const { return array; }
	GDALDataset* get_dataset(void) const { return dataset; }
	int get_height(void) const { return dataset->GetRasterBand(1)->GetYSize(); }
	int get_width(void) const { return dataset->GetRasterBand(1)->GetXSize(); }
	int get_index(double x, double y) const;

protected:
	T* array;
	GDALDataset* dataset;
};

template<class T>
inline Raster<T>::Raster(std::string path, GDALAccess access) {
	GDALAllRegister();
	dataset = (GDALDataset*)GDALOpen(path.c_str(), access);

	if (dataset == NULL) {
		// TODO: Throw error here.
	}

	GDALRasterBand* band = dataset->GetRasterBand(1);
	array = (T*)CPLMalloc(get_width()*get_height()*sizeof(T));
	band->RasterIO(GF_Read, 0, 0, get_width(), get_height(), array,
		            get_width(), get_height(), GDT_Float32, 0, 0);
}

template<class T>
inline Raster<T>::Raster(const Raster& raster, std::string path, GDALAccess access) {
	GDALDataset* src = raster.get_dataset();
	GDALDriver* driver = src->GetDriver();
	dataset = driver->CreateCopy(path.c_str(), src, false, NULL, NULL, NULL);
	array = (T*)CPLMalloc(get_width()*get_height()*sizeof(T));
	memcpy(array, raster.get_array(), get_width()*get_height()*sizeof(T));
}

template<class T>
inline Raster<T>::~Raster(void) {
	CPLFree(array);
	GDALClose(dataset);
}

template<class T>
inline int Raster<T>::get_index(double x, double y) const {
	double transform[6];
	dataset->GetGeoTransform(transform);
	double inv_transform[6];
	bool success = GDALInvGeoTransform(transform, inv_transform);

	int row = floor(inv_transform[3] + inv_transform[4] * x + inv_transform[5] * y);
	int column = floor(inv_transform[0] + inv_transform[1] * x + inv_transform[2] * y);
	return row * get_width() + column;
}

template<class T>
inline void Raster<T>::Update(void) const {
	GDALRasterBand* band = dataset->GetRasterBand(1);
	band->RasterIO(GF_Write, 0, 0, get_width(), get_height(), array,
		            get_width(), get_height(), GDT_Float32, 0, 0);
}
