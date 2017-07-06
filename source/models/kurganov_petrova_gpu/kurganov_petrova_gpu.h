#pragma once

#include <memory>
#include <rapidjson/document.h>
#include <common/gpu_raster.hpp>

class KurganovPetrovaGpu {
public:
	KurganovPetrovaGpu(const rapidjson::Value& root);

private:
	GpuRaster<float>* depth_;
};
