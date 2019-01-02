SET(SOURCE_FILES error.cpp index_table.cpp)
SET(HEADER_FILES error.h index_table.h raster.hpp)

ADD_LIBRARY(Common STATIC ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(Common ${LIBS})
INSTALL(FILES ${HEADER_FILES} DESTINATION "include/nuflood/common")
