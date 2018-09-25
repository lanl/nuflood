#include "itopography.h"
#include "file.h"
#include "parameter.h"

ITopography::ITopography(const rapidjson::Value& root) {
	metric_ = false;

	if (root.HasMember("topography")) {
		const rapidjson::Value& json = root["topography"];

		if (json.HasMember("elevationFile")) {
			std::string elevation_path = json["elevationFile"].GetString();
			elevation_.Read(elevation_path);
			elevation_.set_name("topographicElevation");
			ReadParameter(json, "metric", metric_);
		} else {
			std::string error_message = "Path to topographic elevation not specified.";
			std::cerr << "ERROR: " << error_message << std::endl;
			std::exit(1);
		}
	} else {
		std::string error_message = "Topography parameters not specified.";
		std::cerr << "ERROR: " << error_message << std::endl;
		std::exit(1);
	}
}

void ITopography::WriteGrids(const IOutput& output) const {
	static bool printed = false;

	if (!printed) {
		//output.WriteGridIfInList(elevation_);
		printed = true;
	}
}
