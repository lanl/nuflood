#pragma once

#include <common/precision.h>
#include <common/index_table.h>
#include <common/grid.h>
#include "sources.h"
#include "constants.h"

class ActiveCells {
public:
	ActiveCells(const Grid<prec_t>& w, const Grid<prec_t>& B, const Sources& R, const Constants& C);

	void Update(const Grid<prec_t>& w, const Grid<prec_t>& B);
	void SetTracking(const bool value);
	bool Tracking(void) const;
	bool Contains(const INT_TYPE i, const INT_TYPE j) const;
	bool ContainsWet(const INT_TYPE i, const INT_TYPE j) const;
	bool Insert(const INT_TYPE i, const INT_TYPE j);
	bool InsertWet(const INT_TYPE i, const INT_TYPE j);

	IndexTable active_, wet_;

private:
	bool tracking_;
};
