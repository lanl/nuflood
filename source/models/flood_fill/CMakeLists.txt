SET(SOURCE_FILES flood_fill.cpp)
SET(HEADER_FILES flood_fill.h input.hpp point_source.h)

# Convert the Flood Fill schema document to a byte array in a header file.
textfile2h(SOURCE_FILE ${PROJECT_SOURCE_DIR}/resources/floodFill.schema.json
           HEADER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/schema.h
		     VARIABLE_NAME "FLOODFILL_SCHEMA")

ADD_LIBRARY(FloodFill STATIC ${SOURCE_FILES} ${HEADER_FILES})
SET(NAME "flood_fill")
SET(EXECUTABLE "${NAME}")

if(WIN32)
	ADD_EXECUTABLE(${EXECUTABLE} WIN32 ${SOURCE_FILES})
elseif(APPLE)
	ADD_EXECUTABLE(${EXECUTABLE} MACOSX_BUNDLE ${SOURCE_FILES})
else()
	ADD_EXECUTABLE(${EXECUTABLE} ${SOURCE_FILES})
endif()

TARGET_LINK_LIBRARIES(${EXECUTABLE} FloodFill Common)
