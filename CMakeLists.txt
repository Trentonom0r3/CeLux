cmake_minimum_required(VERSION 3.15)

# ------------------------------
# 1. Project Setup
# ------------------------------

# Option to enable or disable CUDA support
option(ENABLE_CUDA "Enable CUDA support" ON)

# Use vcpkg toolchain if on Windows
if (WIN32)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/tjerf/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Conditionally include CUDA in the project languages
if(ENABLE_CUDA)
    project(CeLux LANGUAGES CXX CUDA)
else()
    project(CeLux LANGUAGES CXX)
endif()

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------
# 2. Specify CUDA Paths Directly
# ------------------------------

if(ENABLE_CUDA)
    # Define CUDA toolkit root path
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")

    # Include CUDA directories
    include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
    link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")

    # Include NPP headers
    include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")

    # List of NPP libraries
    set(NPP_LIBRARIES
        nppicc.lib   # Core functions for image processing
    )
endif()

# ------------------------------
# 3. Manually Specify FFmpeg Static Libraries
# ------------------------------

# Define the vcpkg installation path for FFmpeg static libraries
set(FFMPEG_VCPKG_STATIC_DIR "C:/Users/tjerf/vcpkg/installed/x64-windows-static")

# Specify the path to FFmpeg include directories
set(FFMPEG_INCLUDE_DIRS "${FFMPEG_VCPKG_STATIC_DIR}/include")

# Add WebP, x264, and x265 libraries to the list of FFmpeg dependencies
set(FFMPEG_DEPENDENCIES
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libwebp.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libwebpdecoder.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libwebpdemux.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libwebpmux.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libx264.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/x265-static.lib" #libsharpyuv.lib
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/libsharpyuv.lib"
)

# List the static FFmpeg libraries you want to link against
set(FFMPEG_STATIC_LIBS
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avutil.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avcodec.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avformat.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avdevice.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/swscale.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/swresample.lib"
    "${FFMPEG_VCPKG_STATIC_DIR}/lib/avfilter.lib"
    ${FFMPEG_DEPENDENCIES}
)

# ------------------------------
# 4. Keep Torch/CUDA as Dynamic Libraries (B)
# ------------------------------

# Set the path to the root of the libtorch installation
if(ENABLE_CUDA)
    set(TORCH_INSTALL_PREFIX "C:/Users/tjerf/Downloads/libtorch-win-shared-with-deps-2.4.0+cu121/libtorch")
else()
    set(TORCH_INSTALL_PREFIX "C:/Users/tjerf/Downloads/libtorch-win-shared-with-deps-2.4.0+cpu/libtorch")
endif()
set(CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}")

# Find Torch
find_package(Torch REQUIRED)

# ------------------------------
# 5. Find Windows Media Foundation Libraries
# ------------------------------

# Define the Windows SDK lib path (adjust the version as needed)
set(WINDOW_SDK_LIB_PATH "C:/Program Files (x86)/Windows Kits/10/Lib")

find_library(MF_LIB NAMES mf PATHS "${WINDOW_SDK_LIB_PATH}/*/um/x64" NO_DEFAULT_PATH)
if(NOT MF_LIB)
    message(FATAL_ERROR "mf.lib not found. Ensure the Windows SDK is installed.")
endif()

find_library(MFUUUID_LIB NAMES mfuuid PATHS "${WINDOW_SDK_LIB_PATH}/*/um/x64" NO_DEFAULT_PATH)
if(NOT MFUUUID_LIB)
    message(FATAL_ERROR "mfuuid.lib not found. Ensure the Windows SDK is installed.")
endif()

find_library(OLE32_LIB NAMES ole32 PATHS "${WINDOW_SDK_LIB_PATH}/*/um/x64" NO_DEFAULT_PATH)
if(NOT OLE32_LIB)
    message(FATAL_ERROR "ole32.lib not found. Ensure the Windows SDK is installed.")
endif()

# ------------------------------
# 6. Manually Specify Python 3.12 Paths (C)
# ------------------------------

set(PYTHON_EXECUTABLE "C:/Users/tjerf/AppData/Local/Programs/Python/Python312/python.exe")
set(PYTHON_INCLUDE_DIR "C:/Users/tjerf/AppData/Local/Programs/Python/Python312/include")
set(PYTHON_LIBRARY "C:/Users/tjerf/AppData/Local/Programs/Python/Python312/libs/python312.lib")

if(NOT EXISTS "${PYTHON_EXECUTABLE}")
    message(FATAL_ERROR "Python executable not found at ${PYTHON_EXECUTABLE}")
endif()

if(NOT EXISTS "${PYTHON_INCLUDE_DIR}")
    message(FATAL_ERROR "Python include directory not found at ${PYTHON_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${PYTHON_LIBRARY}")
    message(FATAL_ERROR "Python library not found at ${PYTHON_LIBRARY}")
endif()

# ------------------------------
# 7. Find pybind11 using vcpkg
# ------------------------------

find_package(pybind11 REQUIRED)

# ------------------------------
# 8. Find Torch Python Library
# ------------------------------

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT TORCH_PYTHON_PATH)
    message(FATAL_ERROR "Could not find torch in the current Python environment.")
endif()

find_library(
    TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS "${TORCH_PYTHON_PATH}/lib" "${TORCH_PYTHON_PATH}"
    NO_DEFAULT_PATH
)

if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Torch Python library not found.")
endif()

# ------------------------------
# 9. Add Source Files
# ------------------------------

file(GLOB_RECURSE SOURCES_COMMON
    "src/CeLux/*.cpp"
    "src/CeLux/conversion/*.cpp"
    "src/CeLux/backends/*.cpp"
    "src/CeLux/conversion/cpu/*.cpp"
    "src/CeLux/filters/*.cpp"
)

if(ENABLE_CUDA)
    file(GLOB_RECURSE SOURCES_CUDA
        "src/CeLux/conversion/gpu/cuda*.cpp"
        "src/CeLux/cuda/*.cu"
    )
    set(SOURCES ${SOURCES_COMMON} ${SOURCES_CUDA})
else()
    set(SOURCES ${SOURCES_COMMON})
endif()

add_library(CeLuxLib STATIC ${SOURCES})

if(ENABLE_CUDA)
    target_compile_definitions(CeLuxLib PUBLIC CUDA_ENABLED)
endif()
find_package(spdlog CONFIG REQUIRED)
message(STATUS "spdlog found: ${spdlog_VERSION}")
   find_package(fmt CONFIG REQUIRED)
   message(STATUS "fmt found: ${fmt_VERSION}")
target_include_directories(CeLuxLib PUBLIC
    ${CMAKE_SOURCE_DIR}/include/CeLux
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/cpu
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/gpu
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/gpu/cuda
    ${CMAKE_SOURCE_DIR}/include/CeLux/error
    ${CMAKE_SOURCE_DIR}/include/CeLux/backends
    ${CMAKE_SOURCE_DIR}/include/CeLux/filters
    ${CMAKE_SOURCE_DIR}/magic_enum/include
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
    ${FFMPEG_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${spdlog_INCLUDE_DIRS} #- this does not work
    ${fmt_INCLUDE_DIRS}
)

if(ENABLE_CUDA)
    target_include_directories(CeLuxLib PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/include")
endif()

set_target_properties(CeLuxLib PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)

if(ENABLE_CUDA)
    set_target_properties(CeLuxLib PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

target_link_libraries(CeLuxLib PUBLIC
    ws2_32
    bcrypt
    secur32
    mfplat
    strmiids
    ${MFUUUID_LIB}
    ${MF_LIB}
    ${OLE32_LIB}
    ${FFMPEG_STATIC_LIBS}
    ${TORCH_LIBRARIES}
    spdlog::spdlog_header_only
    fmt::fmt-header-only
)

if(ENABLE_CUDA)
    target_link_libraries(CeLuxLib PUBLIC
        CUDA::cudart
        "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib"
        ${NPP_LIBRARIES}
    )
endif()

# ------------------------------
# 10. Build the Python Module
# ------------------------------

add_library(celux MODULE  ${CMAKE_SOURCE_DIR}/src/CeLux/Python/Bindings.cpp)

target_compile_options(celux PRIVATE /O2 /Ob2 /Oi /Ot /GL)
if(ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
endif()

target_include_directories(celux PUBLIC
    ${CMAKE_SOURCE_DIR}/include/CeLux
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/cpu
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/gpu
    ${CMAKE_SOURCE_DIR}/include/CeLux/conversion/gpu/cuda
    ${CMAKE_SOURCE_DIR}/include/CeLux/error
    ${CMAKE_SOURCE_DIR}/include/CeLux/backends
    ${CMAKE_SOURCE_DIR}/include/CeLux/python
    ${CMAKE_SOURCE_DIR}/include/CeLux/filters
        ${CMAKE_SOURCE_DIR}/magic_enum/include
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
    ${FFMPEG_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${spdlog_INCLUDE_DIRS} #- this does not work
    ${fmt_INCLUDE_DIRS}
)

set_target_properties(celux PROPERTIES
    PREFIX ""  
    SUFFIX ".pyd"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Ensure the output directory exists
if(ENABLE_CUDA)
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/celux_cuda")
    set(LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/celux_cuda")
    set(RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/celux_cuda")
    set(ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib_cuda")
else()
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/celux")
    set(LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/celux")
    set(RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/celux")
    set(ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

set_target_properties(celux PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY}
    RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_DIRECTORY}
)

target_link_libraries(celux PUBLIC
    CeLuxLib
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    ${FFMPEG_STATIC_LIBS}
    ${PYTHON_LIBRARY}
    spdlog::spdlog_header_only
    fmt::fmt-header-only
)

if(ENABLE_CUDA)
    target_link_libraries(celux PUBLIC CUDA::cudart)
endif()

# ------------------------------
# 11. Optional: Clean Up Other Output Directories
# ------------------------------

set_target_properties(CeLuxLib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${ARCHIVE_OUTPUT_DIRECTORY}"
)
