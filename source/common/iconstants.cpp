#include "iconstants.h"
#include "parameter.h"

IConstants::IConstants(void) {
	gravitational_acceleration_ = (prec_t)9.80665;
	machine_epsilon_ = std::numeric_limits<prec_t>::epsilon();
	num_cells_ = 0;
}

IConstants::IConstants(const rapidjson::Value& root, const ITopography& topography) {
	gravitational_acceleration_ = (prec_t)9.80665;
	machine_epsilon_ = std::numeric_limits<prec_t>::epsilon();

	if (root.HasMember("constants")) {
		const rapidjson::Value& json = root["constants"];
		ReadParameter(json, "gravitationalAcceleration", gravitational_acceleration_);
		ReadParameter(json, "machineEpsilon", machine_epsilon_);
	}

	if (topography.metric()) {
		cellsize_x_ = (prec_t)topography.elevation().cellsize();
		cellsize_y_ = (prec_t)topography.elevation().cellsize();
	} else {
		cellsize_x_ = (prec_t)topography.elevation().cellsize() * 6378137.0*3.1415927 / 180.0;
		cellsize_y_ = (prec_t)topography.elevation().cellsize() * 6378137.0*3.1415927 / 180.0;
	}

	num_columns_ = topography.elevation().num_columns();
	num_rows_ = topography.elevation().num_rows();
	num_cells_ = (unsigned long)topography.elevation().num_columns() *
	             (unsigned long)topography.elevation().num_rows();
}
