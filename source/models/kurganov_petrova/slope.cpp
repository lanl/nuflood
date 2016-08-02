#include "slope.h"

Slope::Slope(const Grid<prec_t>& interpolated_grid) {
	wx_.Copy(interpolated_grid);
	wx_.Fill((prec_t)0);
	wx_.set_name("waterSurfaceElevationHorizontalSlope");
	wy_.Copy(wx_);
	wy_.set_name("waterSurfaceElevationVerticalSlope");

	hux_.Copy(wx_);
	hux_.set_name("horizontalDischargeHorizontalSlope");
	huy_.Copy(wx_);
	huy_.set_name("horizontalDischargeVerticalSlope");

	hvx_.Copy(wx_);
	hvx_.set_name("verticalDischargeHorizontalSlope");
	hvy_.Copy(wx_);
	hvy_.set_name("verticalDischargeVerticalSlope");
}
