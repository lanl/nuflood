#include <math.h>
#include <iostream>
#include <common/parameter.h>
#include "topography.h"
#include "constants.h"

Constants::Constants(const rapidjson::Value& root, const Topography& topography) : IConstants(root, topography) {
	kappa_ = (prec_t)sqrt((prec_t)0.01*std::max(std::max((prec_t)1.0, cellsize_x_), cellsize_y_));
	track_wet_cells_ = false;

	if (root.HasMember("constants")) {
		const rapidjson::Value& constants_json = root["constants"];
		ReadParameter(constants_json, "desingularizationConstant", kappa_);
		ReadParameter(constants_json, "trackWetCells", track_wet_cells_);
	}

	num_columns_ = topography.elevation_interpolated().num_columns();
	num_rows_ = topography.elevation_interpolated().num_rows();
	num_cells_ = (long unsigned int)num_columns_ * (long unsigned int)num_rows_;
}
