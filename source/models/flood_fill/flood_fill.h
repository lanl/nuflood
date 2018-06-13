#pragma once

#include <common/grid.h>
#include <common/document.h>
#include <common/index_table.h>

class FloodFill {
public:
	FloodFill(const rapidjson::Value& root);
	void Run(void);

private:
	void Grow(void);
	void UpdateWetCells(void);
	void FillCorner(INT_TYPE column, INT_TYPE row);
	void FillCorners(void);

	Folder output_folder_;
	INT_TYPE num_seeds_, num_wet_;
	INT_TYPE num_iterations_;
	Grid<float> B_, w_, h_;
	IndexTable wet_, seed_, seed_holder_;
};
