#pragma once

#include <common/precision.h>
#include <common/grid.h>
#include "output.h"

class Flux {
public:
	Flux(const Grid<prec_t>& interpolated_grid);
	void WriteGrids(const Output& output, const prec_t current_time);

	const Grid<prec_t>& wx(void) const { return wx_; }
	const Grid<prec_t>& wy(void) const { return wy_; }
	const Grid<prec_t>& hux(void) const { return hux_; }
	const Grid<prec_t>& huy(void) const { return huy_; }
	const Grid<prec_t>& hvx(void) const { return hvx_; }
	const Grid<prec_t>& hvy(void) const { return hvy_; }

private:
	Grid<prec_t> wx_,  wy_;
	Grid<prec_t> hux_, huy_;
	Grid<prec_t> hvx_, hvy_;
};
