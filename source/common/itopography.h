#pragma once

#include "document.h"
#include "ioutput.h"
#include "precision.h"
#include "raster.hpp"

class ITopography {
public:
	ITopography(const rapidjson::Value& root);

	const Raster<prec_t>& elevation(void) const { return elevation_; }
	bool metric(void) const { return metric_; }
	void WriteGrids(const class IOutput& output) const;

protected:
	Raster<prec_t> elevation_;
	bool metric_;
};
