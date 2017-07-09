find_package(Doxygen)

IF(NOT DOXYGEN_FOUND)
    MESSAGE(STATUS "Doxygen not found. Documentation will not be built.")
ELSE()
	SET(SOURCE_FILES
	    ${CMAKE_CURRENT_LIST_DIR}/../source/common/raster.hpp
	    ${CMAKE_CURRENT_LIST_DIR}/../source/common/gpu_raster.hpp
	    ${CMAKE_CURRENT_LIST_DIR}/../source/models/kurganov_petrova_gpu/kurganov_petrova_gpu.h
	    ${CMAKE_CURRENT_LIST_DIR}/../source/models/kurganov_petrova_gpu/kurganov_petrova_gpu.cpp
	    ${CMAKE_CURRENT_LIST_DIR}/../source/models/kurganov_petrova_gpu/compute_time_step.h
	    ${CMAKE_CURRENT_LIST_DIR}/../source/models/kurganov_petrova_gpu/compute_time_step.cu)
	FILE(GLOB MARKDOWN_DOC ${CMAKE_CURRENT_LIST_DIR}/../doc/*.md)
	LIST(APPEND MARKDOWN_DOC ${CMAKE_CURRENT_LIST_DIR}/../README.md)

	CONFIGURE_FILE(Doxyfile.in Doxyfile @ONLY)
	ADD_CUSTOM_COMMAND(OUTPUT html
		COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/html
		DEPENDS ${MARKDOWN_DOC} ${SOURCE_FILES} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile*
		WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../)

	add_custom_target(doc ALL DEPENDS html)
ENDIF()
