SET(SOURCE_FILES cellular_automata.cpp cellular_automata_input.cpp)
SET(HEADER_FILES cellular_automata.h cellular_automata_input.h point_source.h)

# Convert the Flood Fill schema document to a byte array in a header file.
textfile2h(SOURCE_FILE ${PROJECT_SOURCE_DIR}/resources/cellularAutomata.schema.json
           HEADER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/schema.h
           VARIABLE_NAME "CELLULARAUTOMATA_SCHEMA")

ADD_LIBRARY(CellularAutomata STATIC ${SOURCE_FILES} ${HEADER_FILES})
SET(NAME "cellular_automata")
SET(EXECUTABLE "${NAME}")

ADD_EXECUTABLE(${EXECUTABLE} ${SOURCE_FILES} cellular_automata_cli.cpp)
TARGET_LINK_LIBRARIES(${EXECUTABLE} CellularAutomata Common)
