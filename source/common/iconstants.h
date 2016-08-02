#pragma once

#include "document.h"
#include "precision.h"
#include "itopography.h"

class ITopography;

class IConstants {
public:
	IConstants(void);
	IConstants(const rapidjson::Value& root, const ITopography& topography);

	prec_t gravitational_acceleration(void) const { return gravitational_acceleration_; }
	prec_t machine_epsilon(void) const { return machine_epsilon_; }
	prec_t cellsize_x(void) const { return cellsize_x_; }
	prec_t cellsize_y(void) const { return cellsize_y_; }
	INT_TYPE num_columns(void) const { return num_columns_; }
	INT_TYPE num_rows(void) const { return num_rows_; }
	long unsigned int num_cells(void) const { return num_cells_; }

protected:
	prec_t gravitational_acceleration_;
	prec_t machine_epsilon_;
	prec_t cellsize_x_;
	prec_t cellsize_y_;
	INT_TYPE num_columns_;
	INT_TYPE num_rows_;
	long unsigned int num_cells_;
};
