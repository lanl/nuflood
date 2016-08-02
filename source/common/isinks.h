#pragma once

#include "document.h"
#include "precision.h"
#include "point_sink.h"
#include "iconstants.h"

class IConstants;

class ISinks {
public:
	ISinks(const rapidjson::Value& root, const IConstants& constants);
	std::vector< PointSink<prec_t> >& points(void) { return points_; }
	const unsigned int num_points(void) const { return points_.size(); }

protected:
	std::vector< PointSink<prec_t> > points_;
};
