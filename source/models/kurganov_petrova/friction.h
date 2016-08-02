#pragma once

#include <common/document.h>
#include <common/grid.h>
#include <common/precision.h>

class Friction {
public:
	Friction(const rapidjson::Value& root);
	prec_t manning_value(void) const { return manning_value_; }
	const Grid<prec_t>& manning_grid(void) const { return manning_grid_; }

private:
	File manning_file_;
	Grid<prec_t> manning_grid_;
	prec_t manning_value_;
};
