#pragma once

#include <common/iconstants.h>
#include <common/document.h>
#include "topography.h"

class Topography;

class Constants : public IConstants {
public:
	Constants(const rapidjson::Value& root, const Topography& topography);

	bool track_wet_cells(void) const { return track_wet_cells_; }
	prec_t kappa(void) const { return kappa_; }

private:
	bool track_wet_cells_;
	prec_t kappa_;
};
