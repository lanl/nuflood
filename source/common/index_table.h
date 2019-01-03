#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "precision.h"

typedef std::unordered_map<int_t, std::unordered_set<int_t> > Map;

class IndexTable {
public:
	IndexTable(void);
	bool Insert(const int_t i, const int_t j);
	void InsertAll(const int_t num_columns, const int_t num_rows);
	bool Contains(const int_t i, const int_t j) const;
	void Clear(void);

	Map::iterator begin(void) { return map_.begin(); }
	Map::iterator end(void) { return map_.end(); }
	int_t num(void) { return num_; }

private:
	int_t num_;
	Map map_;
};
