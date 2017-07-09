#pragma once

#include <memory>
#include <rapidjson/document.h>
#include <common/gpu_raster.hpp>

class KurganovPetrovaGpu {
public:
	KurganovPetrovaGpu(const rapidjson::Value& root);
	~KurganovPetrovaGpu(void);

private:
	GpuRaster<double>* topographic_elevation_;
	GpuRaster<double>* water_surface_elevation_;
	GpuRaster<double>* horizontal_discharge_;
	GpuRaster<double>* vertical_discharge_;
};
