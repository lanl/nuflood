#include <common/index_table.h>
#include "gtest/gtest.h"

TEST(IndexTable, Constructor) {
	IndexTable index_table;
	EXPECT_EQ(index_table.num(), 0);
}

TEST(IndexTable, InsertNew) {
	IndexTable index_table;
	bool inserted = index_table.Insert(0, 0);
	EXPECT_EQ(inserted, true);
}

TEST(IndexTable, InsertRepeated) {
	IndexTable index_table;
	index_table.Insert(0, 0);
	bool inserted = index_table.Insert(0, 0);
	EXPECT_EQ(inserted, false);
}

TEST(IndexTable, InsertNewColumn) {
	IndexTable index_table;
	index_table.Insert(0, 0);
	bool inserted = index_table.Insert(0, 1);
	EXPECT_EQ(inserted, true);
}

TEST(IndexTable, InsertAll) {
	IndexTable index_table;
	index_table.InsertAll(2, 2);
	EXPECT_EQ(index_table.num(), 4);
}

TEST(IndexTable, ContainsTrue) {
	IndexTable index_table;
	index_table.Insert(0, 0);
	bool contains = index_table.Contains(0, 0);
	EXPECT_EQ(contains, true);
}

TEST(IndexTable, ContainsFalse) {
	IndexTable index_table;
	index_table.Insert(0, 0);
	bool contains = index_table.Contains(1, 0);
	EXPECT_EQ(contains, false);
}

TEST(IndexTable, Clear) {
	IndexTable index_table;
	index_table.Insert(0, 0);
	index_table.Clear();
	EXPECT_EQ(index_table.num(), 0);
}
