#include <common/parameter.h>
#include "friction.h"

Friction::Friction(const rapidjson::Value& root) {
	manning_value_ = (prec_t)0;

	if (root.HasMember("friction")) {
		const rapidjson::Value& friction_json = root["friction"];
		ReadParameter(friction_json, "manningFile", manning_file_);
		ReadParameter(friction_json, "manningValue", manning_value_);
	}

	if (!manning_file_.IsEmpty()) {
		manning_grid_.Load(manning_file_);
		manning_grid_.BilinearInterpolate();
		manning_grid_.AddBoundaries();
		manning_grid_.set_name("manningCoefficient");
		manning_value_ = (prec_t)0; // If a Manning roughness grid has been defined, reset manning_value_ to zero.
	}
}
