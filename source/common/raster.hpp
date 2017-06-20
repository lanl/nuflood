#pragma once

#include <iostream>
#include <cpl_conv.h>
#include <gdal_priv.h>

template<class T>
class Raster {
public:
	Raster(std::string path, bool read_only = true); // Read from raster file.
	Raster(const Raster& raster, std::string path); // Copy from existing Raster.
	~Raster(void);

	void Update(void);

	T* get_array(void) const { return array; }
	GDALDataset* get_dataset(void) const { return dataset; }
	int get_height(void) const { return dataset->GetRasterBand(1)->GetYSize(); }
	int get_width(void) const { return dataset->GetRasterBand(1)->GetXSize(); }

private:
	T* array;
	GDALDataset* dataset;
};

template<class T>
inline Raster<T>::Raster(std::string path, bool read_only) {
	GDALAllRegister();
	GDALAccess access = read_only ? GA_ReadOnly : GA_Update;
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
inline Raster<T>::Raster(const Raster& raster, std::string path) {
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
