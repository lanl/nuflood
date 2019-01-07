#include <common/raster.hpp>
#include "gtest/gtest.h"

TEST(Raster, ConstructorVoid) {
	Raster<double> raster;
}

TEST(Raster, ConstructorRead) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
}

TEST(Raster, ConstructorCopyToMemory) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	Raster<double> raster_copy(raster, "rasterCopy");
}

TEST(Raster, ConstructorCopyToFile) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	Raster<double> copy(raster, "../test/resources/tmp.tif", "rasterCopy");
}

TEST(Raster, Read) {
	Raster<double> raster;
	raster.Read("../test/resources/elevation.tif");
}

TEST(Raster, ReadTwice) {
	Raster<double> raster;
	raster.Read("../test/resources/elevation.tif");
	raster.Read("../test/resources/depth.tif");
}

TEST(Raster, ReadInvalid) {
	Raster<double> raster;
	EXPECT_THROW(raster.Read("../test/resources/invalid.tif"), std::system_error);
}

TEST(Raster, ReadIncomplete) {
	Raster<double> raster;
	EXPECT_THROW(raster.Read("../test/resources/incomplete.asc"), std::system_error);
}

TEST(Raster, CopyOverwrite) {
	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
	Raster<double> raster_2("../test/resources/elevation.tif", "elevationCopy");
	raster_1.CopyFrom(raster_2);
}

TEST(Raster, Index) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	int_t index = raster.index(8.0, 8.0);
	EXPECT_EQ(index, 136);
}

TEST(Raster, IndexFailure) {
	Raster<double> raster("../test/resources/uninvertible.tif", "raster");
	EXPECT_THROW(raster.index(0.0, 0.0), std::system_error);
}

TEST(Raster, IndexInvalid) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	EXPECT_THROW(raster.index(256.0, 256.0), std::system_error);
}

TEST(Raster, Fill) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	raster.Fill(0.0);
}

TEST(Raster, Update) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	Raster<double> copy(raster, "../test/resources/tmp.tif", "rasterCopy");
	copy.Fill(0.0);
	copy.Update();
}

TEST(Raster, Write) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	Raster<double> copy;
	copy.CopyFrom(raster);
	copy.Write("../test/resources/tmp.tif");
}

TEST(Raster, EqualDimensions) {
	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
	Raster<double> raster_2("../test/resources/depth.tif", "depth");
	raster_1.EqualDimensions(raster_2);
}

TEST(Raster, GetFromCoordinates) {
	Raster<double> raster("../test/resources/depth.tif", "raster");
	double value = raster.GetFromCoordinates(8.0, 8.0);
	EXPECT_EQ(value, 1.0);
}

TEST(Raster, GetFromIndices) {
	Raster<double> raster("../test/resources/depth.tif", "raster");
	int_t i = 8, j = 8;
	double value = raster.GetFromIndices(i, j);
	EXPECT_EQ(value, 1.0);
}

TEST(Raster, GetFromIndex) {
	Raster<double> raster("../test/resources/depth.tif", "raster");
	int_t index = raster.index(8.0, 8.0);
	double value = raster.GetFromIndex(index);
	EXPECT_EQ(value, 1.0);
}

TEST(Raster, SetAtIndex) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	int_t index = raster.index(8.0, 8.0);
	raster.SetAtIndex(index, 10.0);
}

TEST(Raster, SetAtCoordinates) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	raster.SetAtCoordinates(8.0, 8.0, 10.0);
}

TEST(Raster, SetAtIndices) {
	Raster<double> raster("../test/resources/elevation.tif", "raster");
	int_t i = 8, j = 8;
	raster.SetAtIndices(i, j, 10.0);
}

TEST(Raster, Add) {
	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
	Raster<double> raster_2("../test/resources/depth.tif", "depth");
	raster_1.Add(raster_2);
}

TEST(Raster, AddInvalid) {
	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
	Raster<double> raster_2("../test/resources/mismatch.tif", "mismatch");
	EXPECT_THROW(raster_1.Add(raster_2), std::system_error);
}

TEST(Raster, Subtract) {
	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
	Raster<double> raster_2("../test/resources/depth.tif", "depth");
	raster_1.Subtract(raster_2);
}

TEST(Raster, SubtractInvalid) {
	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
	Raster<double> raster_2("../test/resources/mismatch.tif", "mismatch");
	EXPECT_THROW(raster_1.Subtract(raster_2), std::system_error);
}
