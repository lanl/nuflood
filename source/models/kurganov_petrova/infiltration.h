#pragma once

#include <common/document.h>
#include <common/grid.h>
#include <common/precision.h>
#include "topography.h"

class Topography;

class Infiltration {
public:
	Infiltration(const rapidjson::Value& root, const Topography& topography);

	const Grid<prec_t>& K_grid(void) const { return K_grid_; }
	const Grid<prec_t>& psi_grid(void) const { return psi_grid_; }
	const Grid<prec_t>& dtheta_grid(void) const { return dtheta_grid_; }

	prec_t K_value(void) const { return K_value_; }
	prec_t psi_value(void) const { return psi_value_; }
	prec_t dtheta_value(void) const { return dtheta_value_; }

	const Grid<prec_t>& F(void) const { return F_; }
	const Grid<prec_t>& F_old(void) const { return F_old_; }
	const Grid<prec_t>& dF(void) const { return dF_; }

private:
	File K_file_, psi_file_, dtheta_file_;
	Grid<prec_t> K_grid_, psi_grid_, dtheta_grid_;
	Grid<prec_t> F_, F_old_, dF_;
	prec_t K_value_, psi_value_, dtheta_value_;
};
