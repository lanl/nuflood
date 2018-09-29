#include <common/raster.hpp>
#include "gtest/gtest.h"

TEST(Raster, Read) {
	Raster<double> raster;
	raster.Read("../test/resources/elevation.asc");
}

TEST(Raster, Write) {
	Raster<double> raster;
	raster.Read("../test/resources/elevation.asc");
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
