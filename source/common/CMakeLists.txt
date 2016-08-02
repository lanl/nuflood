set(SOURCE_FILES
    iconstants.cpp document.cpp error.cpp file.cpp gpu_error.cpp folder.cpp
    isinks.cpp isources.cpp index_table.cpp itime.cpp name_list.cpp
    timer.cpp itopography.cpp ioutput.cpp)
set(HEADER_FILES
    iconstants.h document.h error.h file.h folder.h gpu_error.h gpu_grid.h
    gpu_vector.h grid.h isinks.h isources.h index_table.h itime.h
    name_list.h parameter.h point_sink.h point_source.h point_source_list.h
    time_series.h timer.h itopography.h ioutput.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(Common STATIC ${SOURCE_FILES} ${HEADER_FILES})

INSTALL(FILES ${HEADER_FILES} DESTINATION "include/nuflood/common")
