#include "active_cells.h"

ActiveCells::ActiveCells(const Grid<prec_t>& w, const Grid<prec_t>& B,
                         const Sources& R, const Constants& C) {
	for (INT_TYPE k = 0; k < R.points().size(); k++) {
		INT_TYPE i = R.points()[k].x_index(B);
		INT_TYPE j = R.points()[k].y_index(B);
		active_.Insert(i, j);
		active_.Insert(i+1, j);
		active_.Insert(i-1, j);
		active_.Insert(i, j+1);
		active_.Insert(i, j-1);
	}

	#pragma omp parallel for
	for (INT_TYPE j = 2; j < w.num_rows()-2; j++) {
		for (INT_TYPE i = 2; i < w.num_columns()-2; i++) {
			if (w.Get(i, j) > B.Get(i, j)) {
				#pragma omp critical
				{
					active_.Insert(i, j);
					active_.Insert(i+1, j);
					active_.Insert(i-1, j);
					active_.Insert(i, j+1);
					active_.Insert(i, j-1);
				}
			}
		}
	}

	tracking_ = C.track_wet_cells();
}

void ActiveCells::Update(const Grid<prec_t>& w, const Grid<prec_t>& B) {
	if (tracking_) {
		for (Map::const_iterator it = wet_.begin(); it != wet_.end(); ++it) {
			for (const INT_TYPE& i: it->second) {
				const INT_TYPE j = it->first;
				active_.Insert(i+1, j);
				active_.Insert(i-1, j);
				active_.Insert(i, j+1);
				active_.Insert(i, j-1);
			}
		}

		wet_.Clear();
	}
}

void ActiveCells::SetTracking(const bool value) {
	tracking_ = value;
}

bool ActiveCells::Tracking(void) const {
	return tracking_;
}

bool ActiveCells::Contains(const INT_TYPE i, const INT_TYPE j) const {
	return active_.Contains(i, j); 
}

bool ActiveCells::Insert(const INT_TYPE i, const INT_TYPE j) {
	return active_.Insert(i, j);
}

bool ActiveCells::InsertWet(const INT_TYPE i, const INT_TYPE j) {
	return wet_.Insert(i, j);
}

bool ActiveCells::ContainsWet(const INT_TYPE i, const INT_TYPE j) const {
	return wet_.Contains(i, j);
}
