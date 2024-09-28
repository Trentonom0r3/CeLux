cmake_minimum_required(VERSION 3.15)

# Use vcpkg toolchain if on Windows
if (WIN32)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/tjerf/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
endif()

project(FFMPy)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find FFmpeg package
find_package(ffmpeg REQUIRED)

# Include directories for FFmpeg and your custom headers
include_directories(${FFMPEG_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include/FFMPY)

# Link directories for FFmpeg
link_directories(${FFMPEG_LIBRARY_DIRS})

# Add source files for the main project
file(GLOB_RECURSE SOURCES "src/FFMPY/*.cpp")

# Create a library target for the main project
add_library(FFMPyLib ${SOURCES})

# Specify include directories for FFMPyLib
target_include_directories(FFMPyLib PRIVATE ${CMAKE_SOURCE_DIR}/include/FFMPY)

# Link FFmpeg libraries to the main project
if (WIN32)
    # Windows-specific libraries
    target_link_libraries(FFMPyLib ${FFMPEG_LIBRARIES} swresample  ws2_32 bcrypt secur32 mfplat)
else()
    # Linux-specific libraries (if any, like pthread)
    target_link_libraries(FFMPyLib ${FFMPEG_LIBRARIES} swresample pthread)
endif()

# Create the main executable target
add_executable(FFMPy src/main.cpp) # Assuming you have a main.cpp file for the main executable

# Link the main executable with the library
target_link_libraries(FFMPy FFMPyLib)

# Add individual test executables
# Test for MediaFormat
add_executable(TestMediaFormat tests/test_media_format.cpp)
target_include_directories(TestMediaFormat PRIVATE ${CMAKE_SOURCE_DIR}/include/FFMPY)
if (WIN32)
    target_link_libraries(TestMediaFormat FFMPyLib ${FFMPEG_LIBRARIES} ws2_32 bcrypt secur32 mfplat)
else()
    target_link_libraries(TestMediaFormat FFMPyLib ${FFMPEG_LIBRARIES} pthread)
endif()

# Test for Decoder (example)
add_executable(TestDecoder tests/test_decoder.cpp)
target_include_directories(TestDecoder PRIVATE ${CMAKE_SOURCE_DIR}/include/FFMPY)
if (WIN32)
    target_link_libraries(TestDecoder FFMPyLib ${FFMPEG_LIBRARIES} ws2_32 bcrypt secur32 mfplat)
else()
    target_link_libraries(TestDecoder FFMPyLib ${FFMPEG_LIBRARIES} pthread)
endif()

# Add a new target for the resampler test
add_executable(FFMPyResamplerTest tests/test_resampler.cpp)

# Specify include directories for the test target
target_include_directories(FFMPyResamplerTest PRIVATE ${CMAKE_SOURCE_DIR}/include/FFMPY)

# Link libraries for the resampler test
if (WIN32)
    target_link_libraries(FFMPyResamplerTest FFMPyLib ${FFMPEG_LIBRARIES} ws2_32 bcrypt secur32 mfplat)
else()
    target_link_libraries(FFMPyResamplerTest FFMPyLib ${FFMPEG_LIBRARIES} pthread)
endif()



# Set properties for C++ standard
set_property(TARGET FFMPyLib PROPERTY CXX_STANDARD 17)
set_property(TARGET FFMPy PROPERTY CXX_STANDARD 17)
set_property(TARGET TestMediaFormat PROPERTY CXX_STANDARD 17)
set_property(TARGET TestDecoder PROPERTY CXX_STANDARD 17)
