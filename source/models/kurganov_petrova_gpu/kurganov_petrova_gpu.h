#pragma once

#include <rapidjson/document.h>
#include <common/gpu_raster.hpp>

class KurganovPetrovaGpu {
public:
	KurganovPetrovaGpu(const rapidjson::Value& root);
	~KurganovPetrovaGpu(void);

	void Run(void);

private:
	GpuRaster<double>* topographic_elevation_;
	GpuRaster<double>* water_surface_elevation_;
	GpuRaster<double>* horizontal_discharge_;
	GpuRaster<double>* vertical_discharge_;
	double time_, start_time_, end_time_, time_step_;
	double desingularization_, gravitational_acceleration_;
};
