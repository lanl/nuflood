#include <common/parameter.h>
#include <common/error.h>
#include "boundary_conditions.h"

boundary_t BoundaryConditions::GetBoundaryType(const std::string type_label) {
	if (type_label.compare("") == 0) {
		return NONE;
	} else if (type_label.compare("open") == 0) {
		return OPEN;
	} else if (type_label.compare("wall") == 0) {
		return WALL;
	} else if (type_label.compare("criticalDepth") == 0) {
		return CRITICAL_DEPTH;
	} else if (type_label.compare("marigram") == 0) {
		return MARIGRAM;
	} else {
		PrintErrorAndExit("'" + type_label + "' is not a recognized boundary condition.");
		return NONE;
	}
}

BoundaryConditions::BoundaryConditions(const rapidjson::Value& root) {
	std::string east_label = "";
	std::string west_label = "";
	std::string north_label = "";
	std::string south_label = "";

	if (root.HasMember("boundaryConditions")) {
		const rapidjson::Value& boundary_json = root["boundaryConditions"];
		ReadParameter(boundary_json, "east", east_label);
		ReadParameter(boundary_json, "west", west_label);
		ReadParameter(boundary_json, "north", north_label);
		ReadParameter(boundary_json, "south", south_label);
	}

	types.east = GetBoundaryType(east_label);
	types.west = GetBoundaryType(west_label);
	types.north = GetBoundaryType(north_label);
	types.south = GetBoundaryType(south_label);
}
