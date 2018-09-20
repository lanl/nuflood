#pragma once

#include "document.h"
#include "grid.h"
#include "itime.h"
#include "precision.h"
#include "time_series.h"
#include "point_source.h"
#include "iconstants.h"

class ISources {
public:
	ISources(void);
	ISources(const rapidjson::Value& root, const IConstants& constants);
	void Update(const ITime& T);

	const std::vector< PointSource<prec_t> >& points(void) const { return points_; }
	unsigned int num_points(void) const { return points_.size(); }
	const Grid<prec_t>& rainfall_grid(void) const { return rainfall_grid_; }
	const TimeSeries<prec_t>& storm_curve(void) const { return storm_curve_; }
	const TimeSeries<prec_t>& marigram(void) const { return marigram_; }
	prec_t storm_curve_proportion(void) const { return storm_curve_proportion_; }

protected:
	std::vector< PointSource<prec_t> > points_;
	Grid<prec_t> rainfall_grid_;
	TimeSeries<prec_t> storm_curve_;
	prec_t storm_curve_proportion_;
	TimeSeries<prec_t> marigram_;
};
