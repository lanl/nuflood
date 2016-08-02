#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include "precision.h"

typedef std::unordered_map<INT_TYPE, std::unordered_set<INT_TYPE> > Map;

class IndexTable {
public:
	IndexTable(void);
	bool Insert(const INT_TYPE i, const INT_TYPE j);
	void InsertAll(const INT_TYPE num_columns, const INT_TYPE num_rows);
	bool Contains(const INT_TYPE i, const INT_TYPE j) const;
	void Clear(void);
	INT_TYPE num_elements(void) const;

	Map::iterator begin(void);
	Map::iterator end(void);

	INT_TYPE num(void) { return num_; }

private:
	INT_TYPE num_;
	Map map_;
};
