#pragma once

#include <common/document.h>
#include <common/isources.h>
#include "topography.h"
#include "constants.h"

class Sources : public ISources {
public:
	Sources(const rapidjson::Value& root, const Topography& topography, const Constants& constants);
	void Update(const Time& T, const Constants& C, double& volume_added);
};
