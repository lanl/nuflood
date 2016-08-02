set(SOURCE_FILES
    active_cells.cpp boundary_conditions.cpp compute_flux.cpp
    compute_timestep.cpp conserved.cpp constants.cpp flux.cpp friction.cpp
    infiltration.cpp integrate.cpp kurganov_petrova.cpp output.cpp slope.cpp
    sources.cpp time.cpp time_derivative.cpp topography.cpp
    update_boundaries.cpp)
set(HEADER_FILES
    active_cells.h boundary_conditions.h compute_flux.h compute_timestep.h
    conserved.h constants.h infiltration.h flux.h friction.h integrate.h
    kurganov_petrova.h output.h slope.h sources.h time.h time_derivative.h
    topography.h update_boundaries.h)

add_library(KurganovPetrova STATIC ${SOURCE_FILES} ${HEADER_FILES})

set(NAME "kurganov_petrova")
set(EXECUTABLE "${NAME}")

if(WIN32)
	add_executable(${EXECUTABLE} WIN32 ${SOURCE_FILES})
elseif(APPLE)
	add_executable(${EXECUTABLE} MACOSX_BUNDLE ${SOURCE_FILES})
else()
	add_executable(${EXECUTABLE} ${SOURCE_FILES})
endif()

target_link_libraries(${EXECUTABLE} KurganovPetrova Common)
