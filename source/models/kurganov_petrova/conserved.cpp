#include <common/parameter.h>
#include "conserved.h"

Conserved::Conserved(const rapidjson::Value& root, const Topography& topography,
                     const Constants& C, const Output& output) {
	w_value_ = hu_value_ = hv_value_ = h_value_ = (prec_t)0;
	volume_computed_ = volume_added_ = (prec_t)0;

	if (root.HasMember("conserved")) {
		const rapidjson::Value& conserved_json = root["conserved"];
		ReadParameter(conserved_json, "waterSurfaceElevationFile", w_file_);
		ReadParameter(conserved_json, "waterSurfaceElevationValue", w_value_);
		ReadParameter(conserved_json, "horizontalDischargeFile", hu_file_);
		ReadParameter(conserved_json, "horizontalDischargeValue", hu_value_);
		ReadParameter(conserved_json, "verticalDischargeFile", hv_file_);
		ReadParameter(conserved_json, "verticalDischargeValue", hv_value_);
		ReadParameter(conserved_json, "depthFile", h_file_);
		ReadParameter(conserved_json, "depthValue", h_value_);
	}

	// Initialize the water surface elevation (i.e., the sum of bathymetry and
	// water depth). If the file is unspecified, assume there is no depth and
	// that the grid is equivalent to that of the interpolated bathymetry.
	if (!w_file_.IsEmpty()) {
		w_.Load(w_file_);
		w_.Add(-topography.minimum());
		w_.BilinearInterpolate();
		w_.AddBoundaries();
	} else {
		w_.Copy(topography.elevation_interpolated());
	}

	// If specified, add a depth grid to the water surface elevation.
	if (!h_file_.IsEmpty()) {
		Grid<prec_t> h(h_file_);
		h.BilinearInterpolate();
		h.AddBoundaries();
		w_.Add(h);
	}

	volume_added_ += w_.InnerSum() - topography.elevation_interpolated().InnerSum();
	volume_added_ *= C.cellsize_x() * C.cellsize_y();

	// If specified, add a constant depth value to the water surface elevation.
	if (h_value_ > (prec_t)0) {
		w_.Add(h_value_);
	}

	// Name the water surface elevation grid.
	w_.set_name("waterSurfaceElevation");

	// Copy this initial water surface elevation grid to the grid which describes
	// water surface elevation at the previous timestep.
	w_old_.Copy(w_);
	w_old_.set_name("previousWaterSurfaceElevation");

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
	hu_.set_name("horizontalDischarge");

	// Copy this initial horizontal discharge grid to the grid which describes
	// horizontal discharge at the previous timestep.
	hu_old_.Copy(hu_);
	hu_old_.set_name("previousHorizontalDischarge");

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
	hv_.set_name("verticalDischarge");

	// Copy this initial vertical discharge grid to the grid which describes
	// vertical discharge at the previous timestep.
	hv_old_.Copy(hv_);
	hv_old_.set_name("previousVerticalDischarge");

	// Ensure the depth and/or maximum depth grids are initialized if the user
	// wishes to output temporal depth or maximum depth data.
	if (output.GridInList("depth") || output.GridInList("maxDepth")) {
		h_.Copy(w_);
		h_.EqualsDifferenceOf(w_, topography.elevation_interpolated());
		h_.set_name("depth");

		if (output.GridInList("maxDepth")) {
			h_max_.Copy(h_);
			h_max_.set_name("maxDepth");
		}
	}

	// Similarly, if the user wishes to output unit discharge data (i.e.,
	// sqrt((hu)^2 + (hv)^2)), ensure the appropriate grids are initialized.
	if (output.GridInList("unitDischarge") || output.GridInList("maxUnitDischarge")) {
		q_.Copy(w_);
		q_.Fill(0.0f);
		q_.set_name("unitDischarge");

		if (output.GridInList("maxUnitDischarge")) {
			q_max_.Copy(q_);
			q_max_.set_name("maxUnitDischarge");
		}
	}

	sum_topography_heights_ = (double)topography.elevation_interpolated().InnerSum();
}

double Conserved::volume_computed(const Constants& C, const Infiltration& I) {
	double cell_area = C.cellsize_x() * C.cellsize_y();
	double depth_computed_ = (double)w_.InnerSum() - sum_topography_heights_;

	if (I.F().data() != nullptr) {
		depth_computed_ += (double)I.F().InnerSum();
	}

	volume_computed_ = depth_computed_ * cell_area;
	return volume_computed_;
}

void Conserved::WriteGrids(const Output& output, const prec_t current_time) {
	output.WriteGridIfInList(current_time, w_);
	output.WriteGridIfInList(current_time, w_old_);
	output.WriteGridIfInList(current_time, hu_);
	output.WriteGridIfInList(current_time, hu_old_);
	output.WriteGridIfInList(current_time, hv_);
	output.WriteGridIfInList(current_time, hv_old_);
	output.WriteGridIfInList(current_time, h_);
	output.WriteGridIfInList(current_time, q_);
	output.WriteGridIfInList(h_max_);
	output.WriteGridIfInList(q_max_);
}
