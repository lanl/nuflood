#pragma once

#include <common/document.h>
#include <common/grid.h>

typedef enum {
	NONE,
	OPEN,
	WALL,
	CRITICAL_DEPTH,
	MARIGRAM
} boundary_t;

class BoundaryConditions {
public:
	BoundaryConditions(const rapidjson::Value& root);
	inline boundary_t GetBoundaryType(const std::string type_label);

	struct types {
		boundary_t west;
		boundary_t east;
		boundary_t north;
		boundary_t south;
	} types;
};
