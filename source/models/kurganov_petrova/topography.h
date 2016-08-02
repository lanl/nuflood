#pragma once

#include <common/document.h>
#include <common/grid.h>
#include <common/precision.h>
#include <common/itopography.h>
#include "output.h"

class Output;

class Topography : public ITopography {
public:
	Topography(const rapidjson::Value& root);

	void WriteGrids(const Output& output) const;
	const Grid<prec_t>& elevation_interpolated(void) const { return elevation_interpolated_; }
	prec_t minimum(void) const { return minimum_; }

private:
	Grid<prec_t> elevation_interpolated_;
	prec_t minimum_;
};
