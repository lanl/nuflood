SET(UNITTEST_SOURCES
    rastertest.cpp)

ADD_EXECUTABLE(unittest ${UNITTEST_SOURCES})
TARGET_LINK_LIBRARIES(unittest gtest gtest_main)
TARGET_LINK_LIBRARIES(unittest Common)
ADD_TEST(NAME unittest COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/unittest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
	include(CodeCoverage)
	SETUP_TARGET_FOR_COVERAGE(${PROJECT_NAME}_coverage unittest coverage)
	SET(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage -std=c++11")
	SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
endif()
