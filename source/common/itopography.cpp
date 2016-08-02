#include "itopography.h"
#include "file.h"
#include "parameter.h"

ITopography::ITopography(const rapidjson::Value& root) {
	metric_ = false;

	if (root.HasMember("topography")) {
		const rapidjson::Value& json = root["topography"];

		ReadParameter(root, "metric", metric_);
		if (json.HasMember("elevationFile")) {
			elevation_.Load(File(json["elevationFile"].GetString()));
		} else {
			elevation_.Initialize(256, 256, 0.0, 0.0, 1.0);
		}
	} else {
		elevation_.Initialize(256, 256, 0.0, 0.0, 1.0);
	}


	elevation_.set_name("topographicElevation");
}

void ITopography::WriteGrids(const IOutput& output) const {
	static bool printed = false;
	if (!printed) {
		output.WriteGridIfInList(elevation_);
		printed = true;
	}
}
