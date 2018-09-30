project(nuflood)
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)

add_definitions(-DUSE_DOUBLE=1)
add_definitions(-DBIG_GRID=0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
include(${PROJECT_SOURCE_DIR}/CMakeModules/bin2h.cmake)
include(${PROJECT_SOURCE_DIR}/CMakeModules/text2h.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
	string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/output)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/output)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/output)
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

# Set GDAL build variables.
find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIR})
link_directories(${GDAL_LIBRARY})
list(APPEND LIBS ${GDAL_LIBRARY})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories("libraries/rapidjson/include")
include_directories(${CMAKE_SOURCE_DIR}/source)
add_subdirectory(libraries/googletest)
add_subdirectory(source)

option(NUFLOOD_BUILD_DOC "Build Nuflood documentation." ON)
if(NUFLOOD_BUILD_DOC)
	add_subdirectory(doc)
endif()

option(BUILD_TESTS "Build all tests." ON)
if(BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()
