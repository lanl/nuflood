#pragma once

#include <common/index_table.h>
#include <common/raster.hpp>
#include "input.hpp"

class FloodFill {
public:
	FloodFill(const Input& input);
	void Run(void);

private:
	void Grow(void);
	void UpdateWetCells(void);
	void WriteResults(void);

	const Input* input_;
	INT_TYPE num_iterations_;
	INT_TYPE num_seeds_, num_wet_;
	Raster<prec_t> B_, w_, h_;
	IndexTable wet_, seed_, seed_holder_;
};
