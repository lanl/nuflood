#include <iostream>
#include <common/parameter.h>
#include "sources.h"

Sources::Sources(const rapidjson::Value& root, const Topography& topography,
                 const Constants& constants) : ISources(root, constants) {
	if (rainfall_grid_.data() != nullptr) {
		rainfall_grid_.BilinearInterpolate();
		rainfall_grid_.AddBoundaries();
	}
}

void Sources::Update(const Time& T, const Constants& C, double& volume_added) {
	ISources::Update(T);
	static prec_t cell_area = C.cellsize_x() * C.cellsize_y();

	for (INT_TYPE k = 0; k < points_.size(); k++) {
		volume_added += points_[k].interpolated_rate(T.current()) * T.step() * cell_area;
	}

	if (rainfall_grid_.data() != nullptr) {
		static prec_t cell_area = C.cellsize_x() * C.cellsize_y();
		static prec_t rainfall_grid_inner_sum = rainfall_grid_.InnerSum();
		volume_added += rainfall_grid_inner_sum * storm_curve_proportion_ * cell_area;
	}
}
