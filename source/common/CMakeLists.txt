set(SOURCE_FILES
    error.cpp)
set(HEADER_FILES
    error.h raster.hpp)

add_library(Common STATIC ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(Common ${LIBS})

INSTALL(FILES ${HEADER_FILES} DESTINATION "include/nuflood/common")
