#include "index_table.h"

IndexTable::IndexTable(void) {
	num_ = 0;
}

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

void IndexTable::InsertAll(const int_t num_columns, const int_t num_rows) {
	for (int_t i = 0; i < num_rows; i++) {
		for (int_t j = 0; j < num_columns; j++) {
			Insert(i, j);
		}
	}
}

bool IndexTable::Contains(const int_t i, const int_t j) const {
	Map::const_iterator row_found = map_.find(i);

	if (row_found != map_.end()) {
		return row_found->second.find(j) != row_found->second.end();
	} else {
		return false;
	}
}

void IndexTable::Clear(void) {
	map_.clear();
	num_ = 0;
}
