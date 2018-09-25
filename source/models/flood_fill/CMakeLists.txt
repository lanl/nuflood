set(SOURCE_FILES
    flood_fill.cpp)
set(HEADER_FILES
    flood_fill.h input.hpp)

# Convert the Flood Fill schema document to a byte array in a header file.
textfile2h(SOURCE_FILE ${PROJECT_SOURCE_DIR}/resources/floodFill.schema.json
           HEADER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/schema.h
		     VARIABLE_NAME "FLOODFILL_SCHEMA")

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
