#pragma once

#include "document.h"
#include "grid.h"
#include "ioutput.h"
#include "precision.h"

class ITopography {
public:
	ITopography(const rapidjson::Value& root);

	const Grid<prec_t>& elevation(void) const { return elevation_; }
	bool metric(void) const { return metric_; }
	void WriteGrids(const class IOutput& output) const;

protected:
	Grid<prec_t> elevation_;
	bool metric_;
};
