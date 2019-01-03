#pragma once

#include <common/index_table.h>
#include <common/raster.hpp>
#include "input.hpp"

class CellularAutomata {
public:
	CellularAutomata(const Input& input);
	void Run(void);
	void Grow(void);
	void UpdateWetCells(void);
	void WriteResults(void);

private:
	const Input* input_;
	int_t num_seeds_, num_wet_;
	Raster<prec_t> B_, w_, h_;
	IndexTable wet_, seed_, seed_holder_;
};
