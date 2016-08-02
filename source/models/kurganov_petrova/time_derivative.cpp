#include <common/parameter.h>
#include "time_derivative.h"

TimeDerivative::TimeDerivative(const rapidjson::Value& root, const Topography& topography) {
	w_value_ = (prec_t)0;
	hu_value_ = (prec_t)0;
	hv_value_ = (prec_t)0;

	if (root.HasMember("timeDerivative")) {
		const rapidjson::Value& conserved_json = root["timeDerivative"];
		ReadParameter(conserved_json, "waterSurfaceElevationFile", w_file_);
		ReadParameter(conserved_json, "waterSurfaceElevationValue", w_value_);
		ReadParameter(conserved_json, "horizontalDischargeFile", hu_file_);
		ReadParameter(conserved_json, "horizontalDischargeValue", hu_value_);
		ReadParameter(conserved_json, "verticalDischargeFile", hv_file_);
		ReadParameter(conserved_json, "verticalDischargeValue", hv_value_);
	}

	// Initialize the water surface elevation (i.e., the sum of bathymetry and
	// water depth). If the file is unspecified, assume there is no depth and
	// that the grid is equivalent to that of the interpolated bathymetry.
	if (!w_file_.IsEmpty()) {
		w_.Load(w_file_);
		w_.BilinearInterpolate();
		w_.AddBoundaries();
	} else {
		w_.Copy(topography.elevation_interpolated());
	}

	// Name the water surface elevation grid.
	w_.set_name("waterSurfaceElevationTimeDerivative");

	// Initialize the horizontal discharge (i.e., the product of depth and
	// horizontal velocity). If the file is unspecified, assume all horizontal
	// velocities are zero.
	if (!hu_file_.IsEmpty()) {
		hu_.Load(hu_file_);
		hu_.BilinearInterpolate();
		hu_.AddBoundaries();
	} else {
		hu_.Copy(topography.elevation_interpolated());
		hu_.Fill((prec_t)0);
	}

	// Add the user-defined constant horizontal discharge.
	hu_.Add(hu_value_);

	// Name the horizontal discharge grid.
	hu_.set_name("horizontalDischargeTimeDerivative");

	// Initialize the vertical discharge (i.e., the product of depth and
	// vertical velocity). If the file is unspecified, assume all vertical
	// velocities are zero.
	if (!hv_file_.IsEmpty()) {
		hv_.Load(hv_file_);
		hv_.BilinearInterpolate();
		hv_.AddBoundaries();
	} else {
		hv_.Copy(topography.elevation_interpolated());
		hv_.Fill((prec_t)0);
	}

	// Add the user-defined constant vertical discharge.
	hv_.Add(hu_value_);

	// Name the vertical discharge grid.
	hv_.set_name("verticalDischargeTimeDerivative");
}

void TimeDerivative::WriteGrids(const Output& output, const prec_t current_time) {
	output.WriteGridIfInList(current_time, w_);
	output.WriteGridIfInList(current_time, hu_);
	output.WriteGridIfInList(current_time, hv_);
}
