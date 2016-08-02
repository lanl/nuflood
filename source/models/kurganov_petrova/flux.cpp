#include "flux.h"

Flux::Flux(const Grid<prec_t>& interpolated_grid) {
	wx_.Copy(interpolated_grid);
	wx_.Fill((prec_t)0);
	wx_.set_name("waterSurfaceElevationHorizontalFlux");
	wy_.Copy(wx_);
	wy_.set_name("waterSurfaceElevationVerticalFlux");

	hux_.Copy(wx_);
	hux_.set_name("horizontalDischargeHorizontalFlux");
	huy_.Copy(wx_);
	huy_.set_name("horizontalDischargeVerticalFlux");

	hvx_.Copy(wx_);
	hvx_.set_name("verticalDischargeHorizontalFlux");
	hvy_.Copy(wx_);
	hvy_.set_name("verticalDischargeVerticalFlux");
}

void Flux::WriteGrids(const Output& output, const prec_t current_time) {
	output.WriteGridIfInList(current_time, wx_);
	output.WriteGridIfInList(current_time, wy_);
	output.WriteGridIfInList(current_time, hux_);
	output.WriteGridIfInList(current_time, huy_);
	output.WriteGridIfInList(current_time, hvx_);
	output.WriteGridIfInList(current_time, hvy_);
}
