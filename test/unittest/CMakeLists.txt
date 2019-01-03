ADD_EXECUTABLE(unittest cellularautomatatest.cpp indextabletest.cpp
	            rastertest.cpp unittest.cpp)
TARGET_LINK_LIBRARIES(unittest CellularAutomata Common gtest gtest_main)
ADD_TEST(NAME "unittest" COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/unittest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
	INCLUDE(CodeCoverage)
	SETUP_TARGET_FOR_COVERAGE(coverage unittest coverage)
endif()
