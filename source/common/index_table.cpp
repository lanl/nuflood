#include "index_table.h"

//! Constructor for IndexTable.
IndexTable::IndexTable(void) {
	num_ = 0;
}

//! Inserts a pair of indices into the IndexTable map.
/*! \return True if the pair of indices was inserted, false if not. */
bool IndexTable::Insert(const int_t i, const int_t j) {
	Map::iterator row_found = map_.find(i);

	if (row_found == map_.end()) {
		num_ += 1;
		map_.insert(std::make_pair(i, std::unordered_set<int_t>({j})));
		return true;
	} else {
		if (row_found->second.find(j) == row_found->second.end()) {
			num_ += 1;
			row_found->second.insert(j);
			return true;
		} else {
			return false;
		}
	}
}

//! Inserts all possible pairs of two-dimensional indices for the given dimensions into the IndexTable map.
/*! \param num_rows Size of the vertical dimension.
    \param num_columns Size of the horizontal dimension. */
void IndexTable::InsertAll(const int_t num_rows, const int_t num_columns) {
	for (int_t i = 0; i < num_rows; i++) {
		for (int_t j = 0; j < num_columns; j++) {
			Insert(i, j);
		}
	}
}

//! Checks whether or not the IndexTable map contains the given pair of indices.
/*! \param i Vertical (row) index.
    \param j Horizontal (column) index.
    \return True if the pair of indices is present, false if not. */
bool IndexTable::Contains(const int_t i, const int_t j) const {
	Map::const_iterator row_found = map_.find(i);

	if (row_found != map_.end()) {
		return row_found->second.find(j) != row_found->second.end();
	} else {
		return false;
	}
}

//! Removes all indices from the IndexTable map.
void IndexTable::Clear(void) {
	map_.clear();
	num_ = 0;
}
