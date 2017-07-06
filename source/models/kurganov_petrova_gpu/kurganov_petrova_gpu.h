#pragma once

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/filereadstream.h>
#include <common/gpu_raster.hpp>

class KurganovPetrovaGpu {
public:
	KurganovPetrovaGpu(const rapidjson::Value& root);
};
