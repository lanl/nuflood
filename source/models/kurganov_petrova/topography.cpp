#include <common/parameter.h>
#include "topography.h"

Topography::Topography(const rapidjson::Value& root) : ITopography::ITopography(root) {
	elevation_.ReplaceValue((prec_t)elevation_.nodata_value(), (prec_t)0);

	minimum_ = elevation_.Minimum();

	elevation_.AddBoundaries();
	elevation_.Normalize();

	elevation_interpolated_.Copy(elevation_);
	elevation_interpolated_.BilinearInterpolate();
	elevation_interpolated_.set_name("interpolatedTopographicElevation");
}

void Topography::WriteGrids(const Output& output) const {
	ITopography::WriteGrids(output);

	static bool printed = false;
	if (!printed) {
		output.WriteGridIfInList(elevation_interpolated_);
		printed = true;
	}
}
