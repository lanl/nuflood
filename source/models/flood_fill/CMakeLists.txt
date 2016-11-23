set(SOURCE_FILES
    flood_fill.cpp)
set(HEADER_FILES
    flood_fill.h)

add_library(FloodFill STATIC ${SOURCE_FILES} ${HEADER_FILES})

set(NAME "flood_fill")
set(EXECUTABLE "${NAME}")

if(WIN32)
	add_executable(${EXECUTABLE} WIN32 ${SOURCE_FILES})
elseif(APPLE)
	add_executable(${EXECUTABLE} MACOSX_BUNDLE ${SOURCE_FILES})
else()
	add_executable(${EXECUTABLE} ${SOURCE_FILES})
endif()

target_link_libraries(${EXECUTABLE} FloodFill Common)
