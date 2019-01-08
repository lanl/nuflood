#pragma once

#include <common/index_table.h>
#include <common/raster.hpp>
#include "cellular_automata_input.h"

class CellularAutomata {
public:
	CellularAutomata(const CellularAutomataInput& input);
	void Run(void);
	void Grow(void);
	void UpdateWetCells(void);
	void WriteResults(void);

private:
	const CellularAutomataInput* input_;
	int_t num_seeds_, num_wet_;
	Raster<prec_t> B_, w_, h_;
	IndexTable wet_, seed_, seed_holder_;
};
