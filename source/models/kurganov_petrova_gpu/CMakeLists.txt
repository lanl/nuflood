set(SOURCE_FILES
    kurganov_petrova_gpu.cpp)
set(HEADER_FILES
    kurganov_petrova_gpu.h)

add_library(KurganovPetrovaGpu STATIC ${SOURCE_FILES} ${HEADER_FILES})

set(NAME "kurganov_petrova_gpu")
set(EXECUTABLE "${NAME}")

if(WIN32)
	add_executable(${EXECUTABLE} WIN32 ${SOURCE_FILES})
elseif(APPLE)
	add_executable(${EXECUTABLE} MACOSX_BUNDLE ${SOURCE_FILES})
else()
	add_executable(${EXECUTABLE} ${SOURCE_FILES})
endif()

target_link_libraries(${EXECUTABLE} ${LIBS} KurganovPetrovaGpu Common)
