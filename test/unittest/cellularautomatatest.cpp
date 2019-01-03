#include <models/cellular_automata/input.hpp>
#include <models/cellular_automata/cellular_automata.h>
#include "gtest/gtest.h"

TEST(CellularAutomata, Constructor1) {
	Input input("../test/resources/cellular_automata_1.json");
	CellularAutomata cellular_automata(input);
}

TEST(CellularAutomata, Constructor2) {
	Input input("../test/resources/cellular_automata_2.json");
	CellularAutomata cellular_automata(input);
}

TEST(CellularAutomata, Constructor3) {
	Input input("../test/resources/cellular_automata_3.json");
	CellularAutomata cellular_automata(input);
}

TEST(CellularAutomata, ConstructorInvalidJson) {
	std::string input_path = "../test/resources/cellular_automata_invalid_json.json";
	EXPECT_THROW(Input input(input_path), std::system_error);
}

TEST(CellularAutomata, ConstructorInvalidSchema) {
	std::string input_path = "../test/resources/cellular_automata_invalid_schema.json";
	EXPECT_THROW(Input input(input_path), std::system_error);
}

TEST(CellularAutomata, Grow) {
	Input input("../test/resources/cellular_automata_1.json");
	CellularAutomata cellular_automata(input);
	cellular_automata.Grow();
}

TEST(CellularAutomata, UpdateWetCells) {
	Input input("../test/resources/cellular_automata_1.json");
	CellularAutomata cellular_automata(input);
	cellular_automata.Grow();
	cellular_automata.UpdateWetCells();
}

//
//TEST(FloodFill, ConstructorRead) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//}
//
//TEST(FloodFill, ConstructorCopyToMemory) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	Raster<double> raster_copy(raster, "rasterCopy");
//}
//
//TEST(FloodFill, ConstructorCopyToFile) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	Raster<double> copy(raster, "../test/resources/tmp.tif", "rasterCopy");
//}
//
//TEST(FloodFill, Read) {
//	Raster<double> raster;
//	raster.Read("../test/resources/elevation.tif");
//}
//
//TEST(FloodFill, ReadTwice) {
//	Raster<double> raster;
//	raster.Read("../test/resources/elevation.tif");
//	raster.Read("../test/resources/depth.tif");
//}
//
//TEST(FloodFill, ReadInvalid) {
//	Raster<double> raster;
//	EXPECT_THROW(raster.Read("../test/resources/invalid.tif"), std::system_error);
//}
//
//TEST(FloodFill, ReadIncomplete) {
//	Raster<double> raster;
//	EXPECT_THROW(raster.Read("../test/resources/incomplete.asc"), std::system_error);
//}
//
//TEST(FloodFill, CopyOverwrite) {
//	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
//	Raster<double> raster_2("../test/resources/depth.tif", "depth");
//	raster_1.CopyFrom(raster_2);
//}
//
//TEST(FloodFill, Index) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	INT_TYPE index = raster.index(8.0, 8.0);
//	EXPECT_EQ(index, 136);
//}
//
//TEST(FloodFill, IndexInvalid) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	EXPECT_THROW(raster.index(256.0, 256.0), std::system_error);
//}
//
//TEST(FloodFill, Fill) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	raster.Fill(0.0);
//}
//
//TEST(FloodFill, Update) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	Raster<double> copy(raster, "../test/resources/tmp.tif", "rasterCopy");
//	copy.Fill(0.0);
//	copy.Update();
//}
//
//TEST(FloodFill, Write) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	Raster<double> copy;
//	copy.CopyFrom(raster);
//	copy.Write("../test/resources/tmp.tif");
//}
//
//TEST(FloodFill, EqualDimensions) {
//	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
//	Raster<double> raster_2("../test/resources/depth.tif", "depth");
//	raster_1.EqualDimensions(raster_2);
//}
//
//TEST(FloodFill, GetFromCoordinates) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	double value = raster.GetFromCoordinates(8.0, 8.0);
//	EXPECT_EQ(value, 1.0);
//}
//
//TEST(FloodFill, GetFromIndices) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	INT_TYPE i = 8, j = 8;
//	double value = raster.GetFromIndices(i, j);
//	EXPECT_EQ(value, 1.0);
//}
//
//TEST(FloodFill, GetFromIndex) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	INT_TYPE index = raster.index(8.0, 8.0);
//	double value = raster.GetFromIndex(index);
//	EXPECT_EQ(value, 1.0);
//}
//
//TEST(FloodFill, SetAtIndex) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	INT_TYPE index = raster.index(8.0, 8.0);
//	raster.SetAtIndex(index, 10.0);
//}
//
//TEST(FloodFill, SetAtCoordinates) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	raster.SetAtCoordinates(8.0, 8.0, 10.0);
//}
//
//TEST(FloodFill, SetAtIndices) {
//	Raster<double> raster("../test/resources/elevation.tif", "raster");
//	INT_TYPE i = 8, j = 8;
//	raster.SetAtIndices(i, j, 10.0);
//}
//
//TEST(FloodFill, Add) {
//	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
//	Raster<double> raster_2("../test/resources/depth.tif", "depth");
//	raster_1.Add(raster_2);
//}
//
//TEST(FloodFill, AddInvalid) {
//	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
//	Raster<double> raster_2("../test/resources/mismatch.tif", "mismatch");
//	EXPECT_THROW(raster_1.Add(raster_2), std::system_error);
//}
//
//TEST(FloodFill, Subtract) {
//	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
//	Raster<double> raster_2("../test/resources/depth.tif", "depth");
//	raster_1.Subtract(raster_2);
//}
//
//TEST(FloodFill, SubtractInvalid) {
//	Raster<double> raster_1("../test/resources/elevation.tif", "elevation");
//	Raster<double> raster_2("../test/resources/mismatch.tif", "mismatch");
//	EXPECT_THROW(raster_1.Subtract(raster_2), std::system_error);
//}

//int main(int argc, char **argv) {
//	::testing::InitGoogleTest(&argc, argv);
//	return RUN_ALL_TESTS();
//}
