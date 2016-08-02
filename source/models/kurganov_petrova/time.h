#pragma once

#include <common/document.h>
#include <common/grid.h>
#include <common/itime.h>

class Time : public ITime {
public:
	// We need to construct the max_step_ grid using the equivalently-sized
	// interpolated bathymetry grid, so we include a reference to the
	// interpolated bathymetry grid in the constructor arguments.
	Time(const rapidjson::Value& root, const Grid<prec_t>& interpolated_grid);
	const Grid<prec_t>& max_step(void) const { return max_step_; }

private:
	Grid<prec_t> max_step_;
};
