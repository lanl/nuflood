#pragma once

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "precision.h"

typedef std::unordered_map<int_t, std::unordered_set<int_t> > Map;

//! Convenience class for storing two-dimensional integer indices.
class IndexTable {
public:
	IndexTable(void);
	bool Insert(const int_t i, const int_t j);
	void InsertAll(const int_t num_columns, const int_t num_rows);
	bool Contains(const int_t i, const int_t j) const;
	void Clear(void);

	//! Returns the beginning of the IndexTable map iterator.
	/*! \return Beginning of the IndexTable map iterator. */
	Map::iterator begin(void) { return map_.begin(); }

	//! Returns the end of the IndexTable map iterator.
	/*! \return End of the IndexTable map iterator. */
	Map::iterator end(void) { return map_.end(); }

	//! Returns the number of index pairs in the IndexTable.
	/*! \return Number of index pairs in the IndexTable. */
	int_t num(void) { return num_; }

private:
	int_t num_;
	Map map_;
};
