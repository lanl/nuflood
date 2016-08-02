#include "index_table.h"

IndexTable::IndexTable(void) {
	num_ = 0;
}

bool IndexTable::Insert(const INT_TYPE i, const INT_TYPE j) {
	Map::iterator row_found = map_.find(j);
	if (row_found == map_.end()) {
		map_.insert(std::make_pair(j, std::unordered_set<INT_TYPE>({i})));
		return true;
	} else {
		if (row_found->second.find(i) == row_found->second.end()) {
			row_found->second.insert(i);
			return true;
		} else {
			return false;
		}
	}
}

void IndexTable::InsertAll(const INT_TYPE num_columns,
                           const INT_TYPE num_rows) {
	for (INT_TYPE j = 0; j < num_rows; j++) {
		for (INT_TYPE i = 0; i < num_columns; i++) {
			Insert(i, j);
		}
	}
}

bool IndexTable::Contains(const INT_TYPE i, const INT_TYPE j) const {
	Map::const_iterator row_found = map_.find(j);
	if (row_found != map_.end()) {
		return row_found->second.find(i) != row_found->second.end();
	} else {
		return false;
	}
}

void IndexTable::Clear(void) {
	map_.clear();
	num_ = 0;
}

INT_TYPE IndexTable::num_elements(void) const {
	return num_;
}

Map::iterator IndexTable::begin(void) {
	return map_.begin();
}

Map::iterator IndexTable::end(void) {
	return map_.end();
}
