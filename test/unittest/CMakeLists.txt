ADD_EXECUTABLE(rastertest rastertest.cpp)
TARGET_LINK_LIBRARIES(rastertest Common gtest gtest_main)
ADD_TEST(NAME "common/raster.hpp" COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rastertest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
	INCLUDE(CodeCoverage)
	SETUP_TARGET_FOR_COVERAGE(${PROJECT_NAME}_coverage rastertest coverage)
endif()
