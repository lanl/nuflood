add_definitions(-DUSE_DOUBLE=1)
add_definitions(-DBIG_GRID=0)

find_program (INTEL_COMPILER icc)

if(INTEL_COMPILER)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11; -O3; -parallel; -openmp;
	                     -lpthreads; -Wall; -g; -shared-intel;
	                     -debug inline-debug-info")
	if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
	endif()
elseif(CMAKE_COMPILER_IS_GNUCXX)
	IF(NOT(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Coverage"))
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -fopenmp -Wall")
		if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
			set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")
			set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
		endif()
	ELSE()
		include(CodeCoverage)
		setup_target_for_coverage(${PROJECT_NAME}_coverage kurganov_petrova coverage "../test/coverage.json")
		set(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage -std=c++11")
		SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
	ENDIF()
endif()

if(APPLE)
	set(CMAKE_MACOSX_RPATH 1)
endif()

if(WIN32)
	add_definitions(-DUNICODE -D_UNICODE)
	set(TARGET_ARCHITECTURE "x86")
endif()

find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIR})
link_directories(${GDAL_LIBRARY})
list(APPEND LIBS ${GDAL_LIBRARY})

find_package(CUDA)
if(CUDA_FOUND)
	include_directories(${CUDA_INCLUDE_DIRS})
	link_directories(${CUDA_LIBRARIES})
	list(APPEND LIBS ${CUDA_LIBRARIES})
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	IF(NOT(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "Coverage"))
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11; -O3; -use_fast_math;
									-ftz=true; -prec-div=false; -prec-sqrt=false;
									-Xptxas -dlcm=ca; -lineinfo;
									-Wno-deprecated-gpu-targets")
	ELSE()
		set(CUDA_NVCC_FLAGS "-std=c++11 -g -O0")
	ENDIF()
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(common)
add_subdirectory(models)
